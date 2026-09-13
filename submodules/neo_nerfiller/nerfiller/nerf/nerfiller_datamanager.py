# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Instruct-NeRF2NeRF Datamanager.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.progress import Console
import torch
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.utils.dataloaders import CacheDataloader
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfiller.utils.typing import *

import json
from typing import Dict

import numpy as np
from PIL import Image
from rich.progress import track

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.model_components import losses
from nerfstudio.utils.misc import torch_compile

from dataclasses import dataclass, field
from functools import cached_property
from typing import (
    Dict,
    ForwardRef,
    Tuple,
    Type,
    cast,
    get_args,
    get_origin,
)

from typing_extensions import TypeVar

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.utils.misc import get_orig_class

TDataset = TypeVar("TDataset", bound=InputDataset, default=InputDataset)
CONSOLE = Console(width=120)


class PatchedDepthDataset(DepthDataset):
    """Dataset that returns images and depths. If no depths are found, then we generate them with Zoe Depth.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        InputDataset.__init__(self, dataparser_outputs, scale_factor)  # Call __init__ from InputDataset explicitly
        # if there are no depth images than we want to generate them all with zoe depth

        if len(dataparser_outputs.image_filenames) > 0 and (
            "depth_filenames" not in dataparser_outputs.metadata.keys()
            or dataparser_outputs.metadata["depth_filenames"] is None
        ):
            device = "cpu"
            CONSOLE.print("[bold yellow] No depth data found! Generating pseudodepth...")
            losses.FORCE_PSEUDODEPTH_LOSS = True
            CONSOLE.print("[bold red] Using psueodepth: forcing depth loss to be ranking loss.")
            cache = dataparser_outputs.image_filenames[0].parent / "depths.npy"
            # Note: this should probably be saved to disk as images, and then loaded with the dataparser.
            #  That will allow multi-gpu training.
            if cache.exists():
                CONSOLE.print("[bold yellow] Loading pseudodata depth from cache!")
                # load all the depths
                self.depths = np.load(cache)
                self.depths = torch.from_numpy(self.depths).to(device)
            else:
                depth_tensors = []
                transforms = self._find_transform(dataparser_outputs.image_filenames[0])
                data = dataparser_outputs.image_filenames[0].parent
                if transforms is not None:
                    meta = json.load(open(transforms, "r"))
                    frames = meta["frames"]
                    filenames = [data / frames[j]["file_path"].split("/")[-1] for j in range(len(frames))]
                else:
                    meta = None
                    frames = None
                    filenames = dataparser_outputs.image_filenames

                zoedepth_folder = Path(__file__).parents[2] / "ZoeDepth"
                self.zoe = torch.compile(torch.hub.load(str(zoedepth_folder), "ZoeD_NK", source="local", pretrained=True).to(device))

                for i in track(range(len(filenames)), description="Generating depth images"):
                    image_filename = filenames[i]
                    pil_image = Image.open(image_filename)
                    image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
                    if len(image.shape) == 2:
                        image = image[:, :, None].repeat(3, axis=2)
                    image = torch.from_numpy(image.astype("float32") / 255.0)

                    with torch.no_grad():
                        image = torch.permute(image, (2, 0, 1)).unsqueeze(0).to(device)
                        if image.shape[1] == 4:
                            image = image[:, :3, :, :]
                        depth_tensor = self.zoe.infer(image).squeeze().unsqueeze(-1)

                    depth_tensors.append(depth_tensor)

                self.depths = torch.stack(depth_tensors)
                np.save(cache, self.depths.cpu().numpy())
            dataparser_outputs.metadata["depth_filenames"] = None
            dataparser_outputs.metadata["depth_unit_scale_factor"] = 1.0
            self.metadata["depth_filenames"] = None
            self.metadata["depth_unit_scale_factor"] = 1.0

        self.depth_filenames = self.metadata["depth_filenames"]
        self.depth_unit_scale_factor = self.metadata["depth_unit_scale_factor"]


@dataclass
class NeRFillerDataManagerConfig(VanillaDataManagerConfig):
    """Configuration for the InstructNeRF2NeRFDataManager."""

    _target: Type = field(default_factory=lambda: NeRFillerDataManager)
    patch_size: int = 32
    """Size of patch to sample from. If >1, patch-based sampling will be used."""
    sample_everywhere: bool = False
    """If true, sample any pixel regardless of masking."""


class NeRFillerDataManager(VanillaDataManager[PatchedDepthDataset]):
    """Data manager for InstructNeRF2NeRF."""

    config: NeRFillerDataManagerConfig

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=[],
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_ray_generator = RayGenerator(self.train_dataset.cameras.to(self.device))

        # pre-fetch the image batch (how images are replaced in dataset)
        self.image_batch = next(self.iter_train_image_dataloader)

        # sort the image batch based on self.image_batch['image_idx']
        i_sorted = torch.sort(self.image_batch["image_idx"]).indices.cpu()
        self.image_batch["image_idx"] = self.image_batch["image_idx"][i_sorted]
        self.image_batch["image"] = self.image_batch["image"][i_sorted]
        self.image_batch["mask"] = self.image_batch["mask"][i_sorted]
        self.image_batch["depth_image"] = self.image_batch["depth_image"][i_sorted]

        # keep a copy of the original image batch
        self.original_image_batch = {}
        self.original_image_batch["image_idx"] = self.image_batch["image_idx"].clone()
        self.original_image_batch["image"] = self.image_batch["image"].clone()
        self.original_image_batch["mask"] = self.image_batch["mask"].clone().float()
        self.original_image_batch["depth_image"] = self.image_batch["depth_image"].clone()

        if self.config.sample_everywhere:
            self.image_batch["mask"] = torch.ones_like(self.image_batch["mask"])

        # if set, sample from only these image indices
        self.sample_image_indices = None

    def create_train_dataset(self):
        """Sets up the data loaders for training"""
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self):
        """Sets up the data loaders for evaluation"""
        return self.dataset_type(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        assert self.train_pixel_sampler is not None

        # possibly choose a subset of images to sample from
        temp_batch = {}
        if self.sample_image_indices:
            for key in self.image_batch.keys():
                temp_batch[key] = self.image_batch[key][self.sample_image_indices]
                temp_batch["original_mask"] = self.original_image_batch["mask"][self.sample_image_indices]
        else:
            temp_batch = self.image_batch
            temp_batch["original_mask"] = self.original_image_batch["mask"]

        batch = self.train_pixel_sampler.sample(temp_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)

        return ray_bundle, batch
    
    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[VanillaDataManager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is VanillaDataManager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is VanillaDataManager:
            return get_args(orig_class)[0]

        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is VanillaDataManager:
                for value in get_args(base):
                    if isinstance(value, ForwardRef):
                        if value.__forward_evaluated__:
                            value = value.__forward_value__
                        elif value.__forward_module__ is None:
                            value.__forward_module__ = type(self).__module__
                            value = getattr(value, "_evaluate")(None, None, set())
                    assert isinstance(value, type)
                    if issubclass(value, InputDataset):
                        return cast(Type[TDataset], value)
        return default
