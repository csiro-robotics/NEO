"""
Neo configuration file.
"""

from __future__ import annotations

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig

from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)

from neo.neo_trainer import NeoTrainerConfig
from neo.neo_model import NeoModelConfig
from neo.neo_pipeline import NeoPipelineConfig
from neo.sparse_guided_nerf_trainer import SparseGuidedNeRFTrainerConfig
from neo.sparse_guided_nerf_pipeline import SparseGuidedNeRFPipelineConfig
from neo.sparse_guided_nerf_model import SparseGuidedNeRFModelConfig
from neo.replace_nerf_model import ReplaceNeRFModelConfig
from neo.replace_nerf_pipeline import ReplaceNeRFPipelineConfig
from neo.data.pixel_samplers import NeoPixelSamplerConfig
from neo.data.neo_dataparser import NeoDataParserConfig
from neo.utils.sparse_dataset import SparseDatasetConfig
from neo.utils.inpainting import InpainterConfig


neo_method = MethodSpecification(
    config=NeoTrainerConfig(
        method_name="neo",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=NeoPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=NeoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                average_init_density=0.01,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Neo method for fast nerf editing. Optimized for working with 24GB GPUs.",
)

neo_big_method = MethodSpecification(
    config=NeoTrainerConfig(
        method_name="neo-big",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=NeoPipelineConfig(
            max_num_iterations_sparse_guided_nerf=30000,
            max_num_iterations_replace_nerf=10000,
            strength_step_sparse_guided_nerf=0.1,
            inpainting_start_iter_sparse_guided_nerf=5000,
            inpainting_repeat_iter_sparse_guided_nerf=2500,
            proposal_warmup_sparse_guided_nerf=5000,
            datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=NeoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                average_init_density=0.01,
            ),
            sparse_dataset=SparseDatasetConfig(
                inpainter=InpainterConfig(
                    inpainting_model="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                    inpainting_images_base_size=(1024, 1024),
                    inpaint_batch_size=4,
                ),
                hemisphera_camera_intrisnsics=(488, 488, 256, 256, 512, 512),
            )
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Neo method for fast nerf editing. Optimized for best quality.",
)

sparse_guided_nerf_method = MethodSpecification(
    config=SparseGuidedNeRFTrainerConfig(
        method_name="sparse-guided-nerf",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=SparseGuidedNeRFPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=NeoDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                pixel_sampler=NeoPixelSamplerConfig(),
            ),
            model=SparseGuidedNeRFModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                average_init_density=0.01,
                background_color='random',
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Sparse-Guided NeRF model utilized by Neo for editing purposes.",
)

replace_nerf_method = MethodSpecification(
    config=TrainerConfig(
        method_name="replace-nerf",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=2000,
        mixed_precision=True,
        pipeline=ReplaceNeRFPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=NeoDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                pixel_sampler=NeoPixelSamplerConfig(),
            ),
            model=ReplaceNeRFModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                average_init_density=0.01,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Replace NeRF model utilized by Neo for editing purposes.",
)