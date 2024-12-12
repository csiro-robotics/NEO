"""
Neo pipeline.
"""
import os
import re
import torch
import typing

from dataclasses import dataclass, field
from typing import Literal, Optional, Type
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor
from diffusers import AutoPipelineForInpainting

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.utils.rich_utils import CONSOLE, status
from nerfstudio.utils.scripts import run_command

from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)

from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)

from neo.neo_model import NeoModel, NeoModelConfig
from neo.utils.sparse_dataset import generate_sparse_dataset


@dataclass
class NeoPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: NeoPipeline)
    """target class to instantiate"""
    bbox_data_path: Path = None
    """path to file with bounding boxes"""
    datamanager: DataManagerConfig = VanillaDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = NeoModelConfig()
    """specifies the model config"""
    sam_checkpoint: str = "sam_vit_h_4b8939.pth"
    """name of the SAM checkpoint"""
    sam_model_type: str = "vit_h"
    """SAM model type, vit_h by default"""
    log_inpaint_images: bool = True
    """If set to True images used for inpainting will be logged."""
    inpainting_model: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    """Model used for inpainting"""

    # TODO Add functionality described below
    loopback_model: bool = True
    """Load the model after editing for further editing"""


class NeoPipeline(VanillaPipeline):
    """Neo Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """

    def __init__(
        self,
        config: NeoPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)

        # Obtain example dataparser output. Used for mask generation
        self.dataparser_output = self.datamanager.dataparser._generate_dataparser_outputs()

        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        assert self.config.bbox_data_path is not None, "No bbox data please specify --pipeline.bbox_data_path 'PATH_TO_BBOX_JSON'"
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
            bbox_dir=self.config.bbox_data_path,
            apply_changes_hook=self.apply_changes_hook,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                NeoModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])

        # Initialize SAM predictor
        sam_checkpoint = f"ckpts/{self.config.sam_checkpoint}"
        sam = sam_model_registry[self.config.sam_model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.sam_predictor = SamPredictor(sam)

        # Initialize inpainting model
        self.inpainting_pipeline = AutoPipelineForInpainting.from_pretrained(
            self.config.inpainting_model, torch_dtype=torch.float16, variant="fp16"
        )
        self.inpainting_pipeline.enable_model_cpu_offload()
        # If inpainted images are black uncomment below lines but it may produce NSFW content
        # self.inpainting_pipeline.safety_checker = None
        # self.inpainting_pipeline.requires_safety_checker = False

    def apply_changes_hook(self):

        verbose = True # If set to True prints training progress of separate models
        dry_run = False

        if not dry_run:
            self.model.eval()
            generate_sparse_dataset(model=self.model, 
                                    sam_predictor=self.sam_predictor,
                                    inpainting_pipeline=self.inpainting_pipeline,
                                    data_path=self.config.datamanager.data, 
                                    bboxes_path=self.config.bbox_data_path, 
                                    dataparser_config=self.config.datamanager.dataparser, 
                                    dataparser_transform=self.dataparser_output.dataparser_transform, 
                                    dataparser_scale=self.dataparser_output.dataparser_scale,
                                    log_inpaint_images=self.config.log_inpaint_images,
                                    )
            self.model.train()

        path = self.config.bbox_data_path
        dataset_path = path / "cache"
        checkpoint_paths = (path / "nerfstudio_models").glob('*.ckpt')
        checkpoint_filenames = [path for path in checkpoint_paths if path.is_file()]

        timestamp_pattern = r'\d{4}-\d{2}-\d{2}_\d{6}'
        if re.match(timestamp_pattern, path.name) or path.name == "latest":
            timestamp = path.name
        else:
            timestamp = ''

        method_name = "sparse-guided-nerf"

        train_sparse_guided_nerf_cmd = [
            f"ns-train {method_name}",
            f"--data {dataset_path}",
            "--experiment_name ''",
            f"--timestamp {timestamp}",
            f"--method-name {method_name}",
            f"--neo-model-path {checkpoint_filenames[0]}",
            f"--pipeline.model.bbox-path {path}/bbox.json",
            "--viewer.quit-on-train-completion True",
            f"--output-dir {str(path).replace('neo/', '').replace(timestamp, '')}",
            f"--pipeline.model.camera-optimizer.mode {self.config.model.camera_optimizer.mode}",
            "--viewer.websocket-port 7008"
        ]
        train_sparse_guided_nerf_cmd = " ".join(train_sparse_guided_nerf_cmd)

        if not dry_run:
            with status(msg="[bold green]Training Sparse Guided NeRF", spinner="moon", verbose=verbose):
                run_command(train_sparse_guided_nerf_cmd, verbose=verbose)

        else:
            print("----------------------------------------------------------------------------------------------------------------")
            print("Train command:")
            print(train_sparse_guided_nerf_cmd)
            print("----------------------------------------------------------------------------------------------------------------")

        method_name = "replace-nerf"

        train_replace_nerf_cmd = [
            f"ns-train {method_name}",
            f"--data {dataset_path}",
            "--experiment_name ''",
            f"--timestamp {timestamp}",
            f"--method-name {method_name}",
            f"--pipeline.model.field-path {checkpoint_filenames[0]}",
            f"--pipeline.model.replace-field-path {str(path).replace('neo', 'sparse-guided-nerf')}/nerfstudio_models/step-000029999.ckpt",
            f"--pipeline.model.bbox-path {path}/bbox.json",
            "--pipeline.datamanager.pixel-sampler.ignore-mask True",
            "--viewer.quit-on-train-completion True",
            f"--output-dir {str(path).replace('neo/', '').replace(timestamp, '')}",
            f"--pipeline.model.camera-optimizer.mode {self.config.model.camera_optimizer.mode}",
            "--viewer.websocket-port 7009"
        ]

        train_replace_nerf_cmd = " ".join(train_replace_nerf_cmd)


        if not dry_run:
            with status(msg="[bold green]Replacing objects", spinner="moon", verbose=verbose):
                run_command(train_replace_nerf_cmd, verbose=verbose)

        else:
            print("----------------------------------------------------------------------------------------------------------------")
            print("Train command:")
            print(train_replace_nerf_cmd)
            print("----------------------------------------------------------------------------------------------------------------")

        CONSOLE.print("[bold green]Editing finished!")