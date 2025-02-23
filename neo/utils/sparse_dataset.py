import cv2
import json
import os
import shutil
import torch

import numpy as np
import torch.nn as nn

from dataclasses import dataclass, field
from diffusers import AutoPipelineForInpainting
from typing import Type, Dict, Any, Tuple
from pathlib import Path
from rich.progress import track
from segment_anything import SamPredictor

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.models.base_model import Model
from nerfstudio.utils.rich_utils import CONSOLE

from neo.render.camera_paths import generate_hemisphere_transforms
from neo.utils.cameras import get_cameras
from neo.utils.collisions import get_collision_groups
from neo.utils.inpainting import inpaint_images_in_grids
from neo.utils.sam_patch import sam_model_registry


@dataclass
class SparseDatasetConfig(InstantiateConfig):
    
    _target: Type = field(default_factory=lambda: SparseDataset)
    """target class to instantiate"""
    log_inpaint_images: bool = True
    """log inpaint images"""
    sam_checkpoint: str = "sam_vit_h_4b8939.pth"
    """name of the SAM checkpoint"""
    sam_model_type: str = "vit_h"
    """SAM model type, vit_h by default"""
    # inpainting_model: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    inpainting_model: str = "runwayml/stable-diffusion-inpainting"
    """model used for inpainting"""
    inpainting_images_base_size: Tuple[int, int] = (512, 512)
    """base size of inpainting images"""
    hemisphere_radius = 0.5
    """radius of the hemisphere"""
    hemisphere_num_layers = 8
    """number of layers in the hemisphere"""
    hemisphere_img_per_layer = 4
    """number of images per layer in the hemisphere"""


class SparseDataset(nn.Module):

    def __init__(self, config: SparseDatasetConfig, device: str = "cpu"):
        super().__init__()
        self.config = config
        self.device = device

        # Initialize SAM predictor
        sam_checkpoint = f"ckpts/{self.config.sam_checkpoint}"
        sam = sam_model_registry[self.config.sam_model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.sam_predictor = SamPredictor(sam)

        # Initialize inpainting model
        self.inpainting_pipeline = AutoPipelineForInpainting.from_pretrained(
            self.config.inpainting_model, torch_dtype=torch.float16, variant="fp16"
        )
        self.inpainting_pipeline.enable_model_cpu_offload()
        # If inpainted images are black uncomment below lines but it may produce NSFW content
        # self.inpainting_pipeline.safety_checker = None
        # self.inpainting_pipeline.requires_safety_checker = False

    def prepare(self, data_path: Path, bboxes_path: Path) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Path, Path]:

        # Create log paths or replace existing logs
        logs_path = None
        if self.config.log_inpaint_images:
            logs_path = bboxes_path / "logs/inpainting"

            if os.path.exists(logs_path):
                shutil.rmtree(logs_path)

            os.makedirs(logs_path)

        # Create output path for generating dataset
        output_path = bboxes_path / "cache"

        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        os.makedirs(output_path)

        # Copy dataset to output
        shutil.copytree(data_path / "images", output_path / "images")
        shutil.copy(data_path / "transforms.json", output_path / "transforms.json")
        shutil.copy(bboxes_path / "bbox.json", output_path / "bbox.json")

        # Load camera data and generate masks
        with open(output_path / "transforms.json", "r") as file:
            camera_data = json.load(file)

        # Sort frames data by image name
        filepaths = []
        for frame in camera_data["frames"]:
            filename = Path(frame["file_path"]).name
            filepaths.append(filename)
            frame["mask_path"] = f"masks/{filename}"
        inds = np.argsort(filepaths)
        camera_data["frames"] = [camera_data["frames"][ind] for ind in inds]

        # Add masks paths to the transforms file
        with open(output_path / "transforms.json", "w") as file:
            json.dump(camera_data, file, indent=4)

        # Load bboxes
        with open(bboxes_path / "bbox.json", "r") as file:
            bboxes = json.load(file)['bboxes']

        return camera_data, bboxes, output_path, logs_path

    def generate(self,
                 model: Model,
                 data_path: Path, 
                 bboxes_path: Path, 
                 dataparser_config: Dict[str, Any],
                 dataparser_transform: torch.Tensor, 
                 dataparser_scale: torch.Tensor) -> None:

        CONSOLE.rule("[bold green]Editing NeRF!")
        CONSOLE.print("[bold green]Generating Sparse Dataset")

        # Copy necessary files
        camera_data, bboxes, output_path, logs_path = self.prepare(data_path, bboxes_path)

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
            dedicated_transforms = generate_hemisphere_transforms(self.config.hemisphere_radius, 
                                                                  self.config.hemisphere_num_layers, 
                                                                  boxes_centre, 
                                                                  dataparser_transform, 
                                                                  dataparser_scale, 
                                                                  num_per_layer=self.config.hemisphere_img_per_layer)

            group_views = []
            for j, dedicated_transform in enumerate(dedicated_transforms):
                num = str((j + 1) + (len(dedicated_transforms) * i))
                while len(num) < 4:
                    num = '0' + num 

                frame = {}
                frame["file_path"] = f"images/dedicated_{num}.png"
                frame["mask_path"] = f"masks/dedicated_{num}.png"
                frame["transform_matrix"] = dedicated_transform.tolist()

                if not "fl_x" in camera_data:
                    frame["fl_x"] = 488
                    frame["fl_y"] = 488
                    frame["cx"] = 256
                    frame["cy"] = 256
                    frame["w"] = 512
                    frame["h"] = 512
                    
                camera_data["frames"].append(frame)
                group_views.append(f"dedicated_{num}.png")

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
        cameras, image_names = get_cameras(dataparser_config)

        # Pair cameras with appriopriate indices
        selected_frames = []
        for view_group in selected_views:
            frames = []
            for view in view_group:
                frames.append((view, image_names.index(view)))
            selected_frames.append(frames)

        # Inpaint selected images
        inpaint_images_in_grids(model, self.sam_predictor, self.inpainting_pipeline, selected_frames, cameras, output_path, log_inpaint_images=self.config.log_inpaint_images, logs_path=logs_path)

        # Remove temporary masks
        # shutil.rmtree(output_path / "temp_masks")


def generate_masks(camera_data: Dict[str, float], 
                   bboxes: Dict[str, np.ndarray], 
                   output_path: Path,
                   dataparser_transform: torch.Tensor, 
                   dataparser_scale: torch.Tensor):
    
    # Create masks folder
    os.makedirs(output_path / "masks", exist_ok=True)
    os.makedirs(output_path / "temp_masks", exist_ok=True)

    # Add boxes to visualization
    CONSOLE.print("Reading bbox data")

    boxes_points = []
    for idx, bbox in enumerate(bboxes):
        bbox_data = bbox['data']
        offset = 0.0

        # Define box vertices for each box
        bbox_w, bbox_h, bbox_d  = np.array(bbox_data['dimensions']) - offset
        mesh_box = np.array([[-bbox_w / 2, -bbox_h / 2, -bbox_d / 2],
                             [bbox_w / 2, -bbox_h / 2, -bbox_d / 2],
                             [-bbox_w / 2, -bbox_h / 2, bbox_d / 2],
                             [bbox_w / 2, -bbox_h / 2, bbox_d / 2],
                             [-bbox_w / 2, bbox_h / 2, -bbox_d / 2],
                             [bbox_w / 2, bbox_h / 2, -bbox_d / 2],
                             [-bbox_w / 2, bbox_h / 2, bbox_d / 2],
                             [bbox_w / 2, bbox_h / 2, bbox_d / 2]])

        # Rotate the box
        alpha, beta, gamma = bbox_data['euler']

        rot_roll = np.array([[1, 0, 0],
                            [0, np.cos(alpha), -np.sin(alpha)],
                            [0, np.sin(alpha), np.cos(alpha)]])
        
        rot_pitch = np.array([[np.cos(beta), 0, np.sin(beta)],
                            [0, 1, 0],
                            [-np.sin(beta), 0, np.cos(beta)]])
        
        rot_yaw = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                            [np.sin(gamma), np.cos(gamma), 0],
                            [0, 0, 1]])
        
        R = rot_roll @ rot_pitch @ rot_yaw
        mesh_box = mesh_box @ R
        
        # Translate the box
        position = np.array(bbox_data['position'])
        mesh_box += position

        boxes_points.append(mesh_box)

    # Iterate through frames
    for i, frame in enumerate(track(camera_data["frames"], description="Generating masks", transient=False)):

        num = str(i)
        while len(num) < 4:
            num = '0' + num

        # Extract transformation matrix
        transform_matrix = np.array(frame["transform_matrix"])
        transform_matrix = np.concatenate([dataparser_transform, np.array([[0, 0, 0, 1]])], axis=0) @ transform_matrix
        transform_matrix[:3, 3] = transform_matrix[:3, 3] * dataparser_scale

        # Convert transformation matrix to camera extrinsic parameters
        extrinsic = np.linalg.inv(transform_matrix)
        transform_matrix = extrinsic @ np.diag([-1, -1, -1, -1])

        # Extract camera parameters
        if "fl_x" in camera_data:
            fl_x = camera_data["fl_x"]
            fl_y = camera_data["fl_y"]
            cx = camera_data["cx"]
            cy = camera_data["cy"]
            width = camera_data["w"]
            height = camera_data["h"]

        else:
            fl_x = frame["fl_x"]
            fl_y = frame["fl_y"]
            cx = frame["cx"]
            cy = frame["cy"]
            width = frame["w"]
            height = frame["h"]

        # Camera matrix
        camera_matrix = np.array([[fl_x, 0.0, cx],
                                  [0.0, fl_y, cy],
                                  [0.0, 0.0, 1.0]])

        mask = np.zeros((height, width))
        for box_points in boxes_points:

            uv_points = []
            for world_point in box_points:

                cam_point = transform_matrix[:3, :3] @ world_point[..., None] + transform_matrix[:3, 3:]

                uv_point = camera_matrix @ cam_point
                uv_point = (uv_point[:2] / uv_point[2])[:, 0]
                uv_point[0] = (2 * cx) - uv_point[0]

                uv_points.append(uv_point)

            top = [uv_points[2], uv_points[3], uv_points[7], uv_points[6]]
            bottom = [uv_points[0], uv_points[1], uv_points[5], uv_points[4]]
            left = [uv_points[4], uv_points[5], uv_points[7], uv_points[6]]
            right = [uv_points[0], uv_points[1], uv_points[3], uv_points[2]]
            front = [uv_points[0], uv_points[2], uv_points[6], uv_points[4]]
            back = [uv_points[1], uv_points[3], uv_points[7], uv_points[5]]

            walls = [top, bottom, left, right, front, back]
            for wall in walls:
                cv2.drawContours(mask, np.int32([wall]), -1, (255, 255, 255), -1)

        # Save images
        img_name = Path(frame['file_path']).name
        cv2.imwrite(f"{output_path}/temp_masks/{img_name}", mask)

        if "dedicated" not in img_name:
            inv_mask = 255 - mask
        
        else:
            inv_mask = mask

        cv2.imwrite(f"{output_path}/masks/{img_name}", inv_mask)