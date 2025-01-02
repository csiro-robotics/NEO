"""
Code to train Neo model.
"""

from __future__ import annotations

import torch
import functools

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Type

from nerfstudio.utils import profiler
from nerfstudio.engine.trainer import TrainerConfig, Trainer
from nerfstudio.engine.callbacks import TrainingCallback
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.pipelines.base_pipeline import VanillaPipeline

TRAIN_INTERATION_OUTPUT = Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]


@dataclass
class SparseGuidedNeRFTrainerConfig(TrainerConfig):
    """Configuration for training regimen"""

    _target: Type = field(default_factory=lambda: SparseGuidedNeRFTrainer)
    """target class to instantiate"""
    neo_model_path: Optional[Path] = None
    """Path to Neo model"""


class SparseGuidedNeRFTrainer(Trainer):
    """Trainer class

    Args:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.

    Attributes:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.
        device: The device to run the training on.
        pipeline: The pipeline object.
        optimizers: The optimizers object.
        callbacks: The callbacks object.
        training_state: Current model training state.
    """

    pipeline: VanillaPipeline
    optimizers: Optimizers
    callbacks: List[TrainingCallback]

    def __init__(self, config: TrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:

        super().__init__(config, local_rank, world_size)

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val") -> None:

        super().setup(test_mode)

        # Load camera optimizers for camera view consistency
        load_path: Path = self.config.neo_model_path
        assert load_path is not None, "Please specify --neo-model-path"
        assert load_path.exists(), f"Checkpoint {load_path} does not exist"
        loaded_state = torch.load(load_path, map_location="cpu")
        if self.config.pipeline.model.camera_optimizer.mode != "off":
            camera_opt = {"camera_opt": loaded_state["optimizers"]["camera_opt"]}
            self.optimizers.load_optimizers(camera_opt)