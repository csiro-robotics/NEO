"""
Neo pipeline.
"""
import os
import typing
import time
import torch

from dataclasses import dataclass, field
from typing import Literal, Optional, Type
from pathlib import Path

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
from neo.utils.sparse_dataset import SparseDataset, SparseDatasetConfig
from neo.utils.config_utils import validate_transforms


@dataclass
class NeoPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: NeoPipeline)
    """target class to instantiate"""
    bbox_data_path: Optional[Path] = None
    """path to file with bounding boxes"""
    datamanager: DataManagerConfig = VanillaDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = NeoModelConfig()
    """specifies the model config"""
    sparse_dataset: SparseDatasetConfig = SparseDatasetConfig()
    """specifies the sparse dataset config"""
    max_num_iterations_sparse_guided_nerf: int = 5000
    """max number of iterations for sparse guided NeRF"""
    max_num_iterations_replace_nerf: int = 2000
    """max number of iterations for replace NeRF"""
    start_inpainting_strength_sparse_guided_nerf: float = 0.8
    """Starting inpainting strength"""
    strength_step_sparse_guided_nerf: float = 0.2
    """Inpainting strength step"""
    inpainting_start_iter_sparse_guided_nerf: float = 1000
    """Inpainting start iteration"""
    inpainting_repeat_iter_sparse_guided_nerf: float = 1000
    """Inpainting repeat iteration"""
    num_images_in_group_sparse_guided_nerf: int = 32
    """Number of images in group"""
    proposal_warmup_sparse_guided_nerf: int = 500
    """Scales n from 1 to proposal_update_every over this many steps"""
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

        # Initialze datamanager
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        # Validate transforms
        validate_transforms(config.datamanager.data)

        # Obtain example dataparser output. Used for mask generation
        self.dataparser_output = self.datamanager.dataparser._generate_dataparser_outputs()

        # Initialize model
        assert self.config.bbox_data_path is not None, "No bbox data please specify --pipeline.bbox_data_path 'PATH_TO_BBOX_JSON'"
        self.grad_scaler = grad_scaler
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

        # Initialize Sparse Dataset
        self.sparse_dataset: SparseDataset = self.config.sparse_dataset.setup(
            output_path=self.config.bbox_data_path,
            split_fraction=self.config.datamanager.dataparser.train_split_fraction,
            device=device
        )

    def apply_changes_hook(self):

        path = self.config.bbox_data_path
        dataset_path = path / "cache"
        logs_path = path / "logs"

        if os.path.exists(logs_path):
            total_start_time = time.time()
        
        verbose = True # If set to True prints training progress of separate models
        dry_run = False

        if not dry_run:
            sparse_start_time = time.time()
            self.model.eval()
            self.sparse_dataset.generate(model=self.model, 
                         data_path=self.config.datamanager.data, 
                         dataparser_config=self.config.datamanager.dataparser, 
                         dataparser_transform=self.dataparser_output.dataparser_transform, 
                         dataparser_scale=self.dataparser_output.dataparser_scale)
            self.model.train()
            sparse_end_time = time.time()
            sparse_elapsed_time = sparse_end_time - sparse_start_time

        checkpoint_paths = (path / "nerfstudio_models").glob('*.ckpt')
        checkpoint_filenames = [path for path in checkpoint_paths if path.is_file()]

        method_name = "sparse-guided-nerf"

        train_sparse_guided_nerf_cmd = [
            f"ns-train {method_name}",
            f"--data {dataset_path}",
            "--experiment_name ''",
            f"--timestamp ''",
            f"--method-name {method_name}",
            f"--max-num-iterations {self.config.max_num_iterations_sparse_guided_nerf}",
            f"--neo-model-path {checkpoint_filenames[0]}",
            f"--pipeline.model.bbox-path {path}/bbox.json",
            "--viewer.quit-on-train-completion True",
            f"--output-dir {path}",
            f"--pipeline.model.camera-optimizer.mode off",
            f"--pipeline.model.start-inpainting-strength {self.config.start_inpainting_strength_sparse_guided_nerf}",
            f"--pipeline.model.strength-step {self.config.strength_step_sparse_guided_nerf}",
            f"--pipeline.model.inpainting-start-iter {self.config.inpainting_start_iter_sparse_guided_nerf}",
            f"--pipeline.model.inpainting-repeat-iter {self.config.inpainting_repeat_iter_sparse_guided_nerf}",
            f"--pipeline.model.num-images-in-group {self.config.num_images_in_group_sparse_guided_nerf}",
            f"--pipeline.model.proposal-warmup {self.config.proposal_warmup_sparse_guided_nerf}",
            f"--pipeline.inpainter.inpainting-model {self.config.sparse_dataset.inpainter.inpainting_model}",
            f"--pipeline.inpainter.inpainting-images-base-size {' '.join(map(str, self.config.sparse_dataset.inpainter.inpainting_images_base_size))}",
            f"--pipeline.inpainter.inpaint-batch-size {self.config.sparse_dataset.inpainter.inpaint_batch_size}",
            "--viewer.websocket-port 7008",
            "neo-data",
            "--eval-mode filename"
        ]
        train_sparse_guided_nerf_cmd = " ".join(train_sparse_guided_nerf_cmd)

        if not dry_run:
            train_sparse_start_time = time.time()
            with status(msg="[bold green]Training Sparse Guided NeRF", spinner="moon", verbose=verbose):
                run_command(train_sparse_guided_nerf_cmd, verbose=verbose)
            train_sparse_end_time = time.time()
            train_sparse_elapsed_time = train_sparse_end_time - train_sparse_start_time

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
            f"--timestamp ''",
            f"--method-name {method_name}",
            f"--max-num-iterations {self.config.max_num_iterations_replace_nerf}",
            f"--pipeline.model.camera-optimizer.mode off",
            f"--pipeline.model.field-path {checkpoint_filenames[0]}",
            f"--pipeline.model.replace-field-path {path}/sparse-guided-nerf/nerfstudio_models/step-{str(self.config.max_num_iterations_sparse_guided_nerf - 1).zfill(9)}.ckpt",
            f"--pipeline.model.bbox-path {path}/bbox.json",
            "--pipeline.datamanager.pixel-sampler.ignore-mask True",
            "--viewer.quit-on-train-completion True",
            f"--output-dir {path}",
            "--viewer.websocket-port 7009",
            "neo-data",
            "--eval-mode filename"
        ]

        train_replace_nerf_cmd = " ".join(train_replace_nerf_cmd)

        if not dry_run:
            train_replace_start_time = time.time()
            with status(msg="[bold green]Replacing objects", spinner="moon", verbose=verbose):
                run_command(train_replace_nerf_cmd, verbose=verbose)
            train_replace_end_time = time.time()
            train_replace_elapsed_time = train_replace_end_time - train_replace_start_time

        else:
            print("----------------------------------------------------------------------------------------------------------------")
            print("Train command:")
            print(train_replace_nerf_cmd)
            print("----------------------------------------------------------------------------------------------------------------")

        CONSOLE.print("[bold green]Editing finished!")

        if os.path.exists(logs_path):
            total_end_time = time.time()
            total_elapsed_time = total_end_time - total_start_time
            log_filename = f"logfile.log"
            with open(logs_path / log_filename, "a") as log_file:
                sparse_minutes = int(sparse_elapsed_time // 60)
                sparse_seconds = int(sparse_elapsed_time % 60)
                train_sparse_minutes = int(train_sparse_elapsed_time // 60)
                train_sparse_seconds = int(train_sparse_elapsed_time % 60)
                train_replace_minutes = int(train_replace_elapsed_time // 60)
                train_replace_seconds = int(train_replace_elapsed_time % 60)
                total_minutes = int(total_elapsed_time // 60)
                total_seconds = int(total_elapsed_time % 60)
                log_file.write(f"[{time.strftime('%Y%m%d_%H%M%S')}] Sparse dataset generation time: {sparse_minutes} min {sparse_seconds} s\n")
                log_file.write(f"[{time.strftime('%Y%m%d_%H%M%S')}] Training Sparse Guided NeRF time: {train_sparse_minutes} min {train_sparse_seconds} s\n")
                log_file.write(f"[{time.strftime('%Y%m%d_%H%M%S')}] Replacing objects time: {train_replace_minutes} min {train_replace_seconds} s\n")
                log_file.write(f"[{time.strftime('%Y%m%d_%H%M%S')}] Total editing time: {total_minutes} min {total_seconds} s\n")

        # Loopback the model
        if self.config.loopback_model:
            edited_model_path = f"{path}/replace-nerf/nerfstudio_models/step-{str(self.config.max_num_iterations_replace_nerf - 1).zfill(9)}.ckpt"
            loaded_state_dict = torch.load(edited_model_path, map_location=self.model.device, weights_only=False)["pipeline"]

            model_state = {}
            for key, value in loaded_state_dict.items():
                if any(x in key for x in ["copy", "replace"]):
                    continue

                if "camera_optimizer.pose_adjustment" in key:
                    if value.shape != self.model.state_dict()[key[len("_model."):]].shape:
                        value = value[-self.model.state_dict()[key[len("_model."):]].shape[0]:]
                    model_state[key[len("_model.") :]] = value

                if key.startswith("_model."):
                    model_state[key[len("_model.") :]] = value

            self.model.load_state_dict(model_state)