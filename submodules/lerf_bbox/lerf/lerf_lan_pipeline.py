import typing
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal, Type, Optional

import torch
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.configs import base_config as cfg
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)

from lerf.data.lerf_datamanager import (
    LERFDataManager,
    LERFDataManagerConfig,
)
from lerf.lerf_lan import LERFLanModel, LERFLanModelConfig
from lerf.encoders.image_encoder import BaseImageEncoderConfig, BaseImageEncoder


@dataclass
class LERFLanPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: LERFLanPipeline)
    """target class to instantiate"""
    datamanager: LERFDataManagerConfig = LERFDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = LERFLanModelConfig()
    """specifies the model config"""
    network: BaseImageEncoderConfig = BaseImageEncoderConfig()
    """specifies the vision-language network config"""
    output_path: Optional[str] = ""
    """output path for the pipeline"""
    model_type: Optional[str] = "neo"
    """type of the model to use"""

class LERFLanPipeline(VanillaPipeline):
    def __init__(
        self,
        config: LERFLanPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
        output_path: Optional[str] = ""    
    ):
        
        super(VanillaPipeline, self).__init__()
        self.config = config

        # TODO HACKY!!!!!!!
        if len(output_path) > 0:
            self.config.output_path = output_path

        self.test_mode = test_mode
        self.image_encoder: BaseImageEncoder = config.network.setup()

        self.datamanager: LERFDataManager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            image_encoder=self.image_encoder,
            output_path=self.config.output_path,
        )

        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            image_encoder=self.image_encoder,
            grad_scaler=grad_scaler,
            output_folder=self.config.output_path
        )
        self.model.to(device)

        print("-------------------------------------------------------------------------------------------------------------------------")
        print("Loading Nerfacto Model")

        if test_mode == "val":
            model_path = "replace-nerf/nerfstudio_models"
        elif test_mode == "test":
            model_path = "nerfstudio_models"

        lerf_state_dict = self.model.state_dict()
        nerfacto_state_dict = torch.load(next((Path(self.config.output_path.replace("lerf-lite-lan", self.config.model_type)) / model_path).glob("*.ckpt")), weights_only=False)
        nerfacto_state_dict = {
            (key[len("_model.") :] if key.startswith("_model.") else key): value for key, value in nerfacto_state_dict["pipeline"].items() if key.startswith("_model.")
        }

        common_keys = set(lerf_state_dict.keys()).intersection(set(nerfacto_state_dict.keys()))
        common_keys = sorted(common_keys)

        for key in common_keys:
            if not key.startswith("camera"):
                lerf_state_dict[key] = nerfacto_state_dict[key]
                print(key)

        self.model.load_state_dict(lerf_state_dict)

        print("-------------------------------------------------------------------------------------------------------------------------")

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(LERFLanModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])
