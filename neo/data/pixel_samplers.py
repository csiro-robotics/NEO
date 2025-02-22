import torch

from dataclasses import dataclass, field
from typing import Type, Dict

from nerfstudio.data.pixel_samplers import PixelSamplerConfig, PixelSampler


@dataclass
class NeoPixelSamplerConfig(PixelSamplerConfig):
    """Config dataclass for PatchPixelSampler."""

    _target: Type = field(default_factory=lambda: NeoPixelSampler)
    """Target class to instantiate."""


class NeoPixelSampler(PixelSampler):

    config: NeoPixelSamplerConfig

    def __init__(self, config: NeoPixelSamplerConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        
    def sample(self, image_batch: Dict):
        """Sample an image batch and return a pixel batch.

        Args:
            image_batch: batch of images to sample from
        """
        if isinstance(image_batch["image"], list):
            image_batch["image"] = torch.nested.to_padded_tensor(torch.nested.nested_tensor(image_batch["image"]), padding=0.0)
            image_batch["mask"] = torch.nested.to_padded_tensor(torch.nested.nested_tensor(image_batch["mask"]), padding=0.0)

        if isinstance(image_batch["image"], torch.Tensor):
            pixel_batch = self.collate_image_dataset_batch(
                image_batch, self.num_rays_per_batch, keep_full_image=self.config.keep_full_image
            )
        else:
            raise ValueError("image_batch['image'] must be a list or torch.Tensor")
        return pixel_batch