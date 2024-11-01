"""
Nerfstudio Sparse Guided NeRF Pipeline
"""

import typing
import torch
import yaml
import torch.distributed as dist

from dataclasses import dataclass, field, fields
from typing import Literal, Optional, Type
from segment_anything import sam_model_registry, SamPredictor
from diffusers import AutoPipelineForInpainting

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
    sam_checkpoint: str = "sam_vit_h_4b8939.pth"
    """name of the SAM checkpoint"""
    sam_model_type: str = "vit_h"
    """SAM model type, vit_h by default"""
    log_inpaint_images: bool = True
    """If set to True images used for inpainting will be logged."""
    inpainting_model: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    """Model used for inpainting"""


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
            if field.name != '_target' and field.name != 'near_plane' and field.name != 'far_plane':
                setattr(self.config.model, field.name, getattr(nerfacto_config.pipeline.model, field.name))

        self.test_mode = test_mode
        self.datamanager: DataManager = self.config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)
        
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        # Initialize SAM predictor
        sam_checkpoint = f"ckpts/{self.config.sam_checkpoint}"
        sam = sam_model_registry[self.config.sam_model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        sam_predictor = SamPredictor(sam)

        # Initialize inpainting model
        inpainting_pipeline = AutoPipelineForInpainting.from_pretrained(
            self.config.inpainting_model, torch_dtype=torch.float16, variant="fp16"
        )
        inpainting_pipeline.enable_model_cpu_offload()
        # If inpainted images are black uncomment below lines but it may produce NSFW content
        # inpainting_pipeline.safety_checker = None
        # inpainting_pipeline.requires_safety_checker = False

        self._model = self.config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            dataparser_config=self.config.datamanager.dataparser,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
            sam_predictor=sam_predictor,
            inpainting_pipeline=inpainting_pipeline,
            reload_data_fn=self.datamanager.setup_train
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                SparseGuidedNeRFModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])