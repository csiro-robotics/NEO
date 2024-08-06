"""
Nerfstudio Sparse Guided NeRF Pipeline
"""

import typing
import yaml

from dataclasses import dataclass, field, fields
from typing import Literal, Optional, Type

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)

from neo.sparse_guided_nerf_model import SparseGuidedNeRFModel, SparseGuidedNeRFModelConfig


@dataclass
class SparseGuidedNeRFPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: SparseGuidedNeRFPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = ParallelDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = SparseGuidedNeRFModelConfig()
    """specifies the model config"""


class SparseGuidedNeRFPipeline(VanillaPipeline):
    """Sparse Guided NeRF Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """

    def __init__(
        self,
        config: SparseGuidedNeRFPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        
        self.config = config

        # Merge confgis
        nerfacto_config_path = self.config.model.bbox_path.parent / "config.yml"
        nerfacto_config = yaml.load(nerfacto_config_path.read_text(), Loader=yaml.Loader)
        assert isinstance(nerfacto_config, TrainerConfig), "Config is not the instance of TrainerConfig and thus cannot be converted."

        # Copy model settings
        for field in fields(NerfstudioDataParserConfig):
            setattr(self.config.datamanager.dataparser, field.name, getattr(nerfacto_config.pipeline.datamanager.dataparser, field.name))

        for field in fields(NerfactoModelConfig):
            if field.name != '_target':
                setattr(self.config.model, field.name, getattr(nerfacto_config.pipeline.model, field.name))

        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)

        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                SparseGuidedNeRFModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])