import json
import os
import shutil
import torch

import numpy as np
import torch.nn as nn

from dataclasses import dataclass, field
from typing import Type, Dict, Any, Tuple, Optional
from pathlib import Path

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.models.base_model import Model
from nerfstudio.data.utils.dataparsers_utils import get_train_eval_split_fraction
from nerfstudio.utils.rich_utils import CONSOLE

from neo.render.camera_paths import generate_hemisphere_transforms
from neo.utils.cameras import get_cameras
from neo.utils.collisions import get_collision_groups
from neo.utils.inpainting import InpainterConfig, Inpainter
from neo.utils.mask_generator import generate_masks


@dataclass
class SparseDatasetConfig(InstantiateConfig):
    
    _target: Type = field(default_factory=lambda: SparseDataset)
    """target class to instantiate"""
    inpainter: InpainterConfig = InpainterConfig()
    """specifies the inpainter config"""
    log_inpaint_images: bool = True
    """log inpaint images"""
    hemisphere_radius: float = -1
    """radius of the hemisphere, set to -1 for adaptive radius"""
    hemisphere_num_layers: int = 8
    """number of layers in the hemisphere"""
    hemisphere_img_per_layer: int = 4
    """number of images per layer in the hemisphere"""
    hemisphera_camera_intrisnsics: Tuple[float, ...] = (246, 246, 128, 128, 256, 256)
    """camera intrinsics for hemisphere camera"""


class SparseDataset(nn.Module):

    def __init__(self, config: SparseDatasetConfig, output_path: Path, split_fraction: Optional[int] = None, device: str = "cpu") -> None:
        super().__init__()
        self.config = config
        self.output_path = output_path
        self.device = device
        self.split_fraction = split_fraction

        # Setup inpainter
        self.inpainter: Inpainter = self.config.inpainter.setup(device=device,
                                                                output_path=output_path)

    def prepare(self, data_path: Path) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Path, Path]:

        # Create output path for generating dataset
        output_path = self.output_path / "cache"

        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        os.makedirs(output_path)

        # Copy dataset to output
        shutil.copytree(data_path / "images", output_path / "images")
        shutil.copy(data_path / "transforms.json", output_path / "transforms.json")
        shutil.copy(self.output_path / "bbox.json", output_path / "bbox.json")

        # Read image names
        if self.split_fraction is not None:
            image_names = []
            for image in os.listdir(data_path / "images"):
                if "train" not in image and "eval" not in image:
                    image_names.append(image)

            image_names = sorted(image_names)

            i_train, i_eval = get_train_eval_split_fraction(image_names, self.split_fraction)

        # Rename images based on train and eval indices
        for idx in i_train:
            old_name = image_names[idx]
            new_name = f"train_{old_name}"
            os.rename(output_path / "images" / old_name, output_path / "images" / new_name)

        for idx in i_eval:
            old_name = image_names[idx]
            new_name = f"eval_{old_name}"
            os.rename(output_path / "images" / old_name, output_path / "images" / new_name)

        # Load camera data and generate masks
        with open(output_path / "transforms.json", "r") as file:
            camera_data = json.load(file)

        # Sort frames data by image number
        filepaths = []
        image_names = np.array(image_names)
        for frame in camera_data["frames"]:
            filename = Path(frame["file_path"]).name
            if filename in image_names[i_eval]:
                new_filename = f"eval_{filename}"
            else:
                new_filename = f"train_{filename}"
            filepaths.append(new_filename)
            frame["file_path"] = f"images/{new_filename}"
            frame["mask_path"] = f"masks/{new_filename}"
        inds = np.argsort([int(''.join(filter(str.isdigit, Path(filepath).stem))) for filepath in filepaths])
        camera_data["frames"] = [camera_data["frames"][ind] for ind in inds]

        # Add masks paths to the transforms file
        with open(output_path / "transforms.json", "w") as file:
            json.dump(camera_data, file, indent=4)

        # Load bboxes
        with open(output_path / "bbox.json", "r") as file:
            bboxes = json.load(file)['bboxes']

        return camera_data, bboxes, output_path

    def generate(self,
                 model: Model,
                 data_path: Path, 
                 dataparser_config: Dict[str, Any],
                 dataparser_transform: torch.Tensor, 
                 dataparser_scale: torch.Tensor) -> None:

        CONSOLE.rule("[bold green]Editing NeRF!")
        CONSOLE.print("[bold green]Generating Sparse Dataset")

        # Copy necessary files
        camera_data, bboxes, output_path = self.prepare(data_path)

        # Get collision groups
        collision_groups = get_collision_groups(bboxes)

        # Get dataparser transformation matrix
        dataparser_transform = dataparser_transform.cpu().detach().numpy()
        if "applied_transform" in camera_data:
            dataparser_transform = dataparser_transform @ np.linalg.inv(np.concatenate([np.array(camera_data["applied_transform"]), np.array([[0, 0, 0, 1]])], axis=0))   
        
        # Select views for inpainting
        selected_views = []
        for i, group in enumerate(collision_groups):
            boxes_centre = np.mean(np.vstack([box["data"]["position"] for box in bboxes if box['idx'] in group]), axis=0)

            # Generate adaptive hemisphere radius
            if self.config.hemisphere_radius == -1:
                self.config.hemisphere_radius = np.max([box["data"]["dimensions"] for box in bboxes if box['idx'] in group]) * 2.5

            dedicated_transforms = generate_hemisphere_transforms(self.config.hemisphere_radius, 
                                                                  self.config.hemisphere_num_layers, 
                                                                  boxes_centre, 
                                                                  dataparser_transform, 
                                                                  dataparser_scale, 
                                                                  num_per_layer=self.config.hemisphere_img_per_layer)

            group_views = []
            for j, dedicated_transform in enumerate(dedicated_transforms):
                num = str((j + 1) + (len(dedicated_transforms) * i))
                while len(num) < 5:
                    num = '0' + num 

                frame = {}
                frame["file_path"] = f"images/dedicated_train_{num}.png"
                frame["mask_path"] = f"masks/dedicated_train_{num}.png"
                frame["transform_matrix"] = dedicated_transform.tolist()
                frame["fl_x"] = self.config.hemisphera_camera_intrisnsics[0]
                frame["fl_y"] = self.config.hemisphera_camera_intrisnsics[1]
                frame["cx"] = self.config.hemisphera_camera_intrisnsics[2]
                frame["cy"] = self.config.hemisphera_camera_intrisnsics[3]
                frame["w"] = self.config.hemisphera_camera_intrisnsics[4]
                frame["h"] = self.config.hemisphera_camera_intrisnsics[5]
                    
                camera_data["frames"].append(frame)
                group_views.append(f"dedicated_train_{num}.png")

            selected_views.append(group_views) 

        generate_masks(camera_data=camera_data, 
                       bboxes=bboxes, 
                       output_path=output_path, 
                       dataparser_transform=dataparser_transform, 
                       dataparser_scale=dataparser_scale)

        # Update transforms.json with dedicated frames for training the sparse NeRF network
        with open(output_path / "transforms.json", "w") as file:
            json.dump(camera_data, file, indent=4)

        # Get cameras for rendering
        dataparser_config.data = output_path
        dataparser_config.eval_mode = "filename"
        cameras, image_names = get_cameras(dataparser_config)

        # Pair cameras with appriopriate indices
        selected_frames = []
        for view_group in selected_views:
            frames = []
            for view in view_group:
                frames.append((view, image_names.index(view)))
            selected_frames.append(frames)

        # Inpaint selected images
        self.inpainter.inpaint_images_in_grids(model, selected_frames, cameras, output_path, initial_inpaint=True, refine_masks=False)

        # Remove temporary masks
        # shutil.rmtree(output_path / "temp_masks")