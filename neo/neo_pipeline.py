"""
Neo pipeline.
"""
import os
import re
import typing
import time

from dataclasses import dataclass, field
from typing import Literal, Optional, Type, Tuple
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

        # Initialze datamanager
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        # Validate transforms
        validate_transforms(config.datamanager.data)

        # Obtain example dataparser output. Used for mask generation
        self.dataparser_output = self.datamanager.dataparser._generate_dataparser_outputs()

        # Initialize model
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

        # Initialize Sparse Dataset
        self.sparse_dataset: SparseDataset = self.config.sparse_dataset.setup(
            device=device,
            output_path=self.config.bbox_data_path,
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

        timestamp_pattern = r'\d{4}-\d{2}-\d{2}_\d{6}'
        if re.match(timestamp_pattern, path.name) or path.name == "latest":
            timestamp = path.name
        else:
            timestamp = ''

        max_num_iterations = 5000

        method_name = "sparse-guided-nerf"

        # TODO This shouldnt be hardcoded!
        neo_variant = "neo-opt"

        train_sparse_guided_nerf_cmd = [
            f"ns-train {method_name}",
            f"--data {dataset_path}",
            "--experiment_name ''",
            f"--timestamp {timestamp}",
            f"--method-name {method_name}",
            f"--max-num-iterations {max_num_iterations}",
            f"--neo-model-path {checkpoint_filenames[0]}",
            f"--pipeline.model.bbox-path {path}/bbox.json",
            "--viewer.quit-on-train-completion True",
            f"--output-dir {str(path).replace(f'{neo_variant}/', '').replace(timestamp, '')}",
            f"--pipeline.model.camera-optimizer.mode {self.config.model.camera_optimizer.mode}",
            "--viewer.websocket-port 7008",
            "nerfstudio-data",
            f"--orientation-method {self.config.datamanager.dataparser.orientation_method}",
            f"--center-method {self.config.datamanager.dataparser.center_method}",
            f"--auto-scale-poses {self.config.datamanager.dataparser.auto_scale_poses}",
            f"--train-split-fraction {self.config.datamanager.dataparser.train_split_fraction}"
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
            f"--timestamp {timestamp}",
            f"--method-name {method_name}",
            f"--pipeline.model.field-path {checkpoint_filenames[0]}",
            f"--pipeline.model.replace-field-path {str(path).replace(neo_variant, 'sparse-guided-nerf')}/nerfstudio_models/step-{str(max_num_iterations - 1).zfill(9)}.ckpt",
            f"--pipeline.model.bbox-path {path}/bbox.json",
            "--pipeline.datamanager.pixel-sampler.ignore-mask True",
            "--viewer.quit-on-train-completion True",
            f"--output-dir {str(path).replace(f'{neo_variant}/', '').replace(timestamp, '')}",
            f"--pipeline.model.camera-optimizer.mode {self.config.model.camera_optimizer.mode}",
            "--viewer.websocket-port 7009",
            "nerfstudio-data",
            f"--orientation-method {self.config.datamanager.dataparser.orientation_method}",
            f"--center-method {self.config.datamanager.dataparser.center_method}",
            f"--auto-scale-poses {self.config.datamanager.dataparser.auto_scale_poses}",
            f"--train-split-fraction {self.config.datamanager.dataparser.train_split_fraction}"
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