import cv2
import json
import os
import shutil
import torch

# import open3d as o3d
import numpy as np
import numpy.typing as npt

from typing import Dict, List, Any
from pathlib import Path
from rich.progress import track
from segment_anything import SamPredictor
from scipy.spatial.transform import Rotation

from nerfstudio.models.base_model import Model
from nerfstudio.utils.rich_utils import CONSOLE

from neo.utils.cameras import get_cameras
from neo.utils.collisions import get_collision_groups
from neo.utils.inpainting import inpaint_images_in_grids


def generate_sparse_dataset(model: Model,
                            sam_predictor: SamPredictor,
                            inpainting_pipeline: Any,
                            data_path: Path, 
                            bboxes_path: Path, 
                            dataparser_config: Dict[str, Any],
                            dataparser_transform: torch.Tensor, 
                            dataparser_scale: torch.Tensor,
                            log_inpaint_images: bool = False) -> None:

    CONSOLE.rule("[bold green]Editing NeRF!")
    CONSOLE.print("[bold green]Generating Sparse Dataset")

    # Create log paths or replace existing logs
    if log_inpaint_images:
        logs_path = Path("logs") / bboxes_path / "inpainting"

        if os.path.exists(logs_path):
            shutil.rmtree(logs_path)

        os.makedirs(logs_path)

    # Create cache path for generating dataset
    cache_path = Path("cache") / bboxes_path

    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)

    os.makedirs(cache_path)

    # Copy dataset to cache
    shutil.copytree(data_path / "images", cache_path / "images")
    shutil.copy(data_path / "transforms.json", cache_path / "transforms.json")
    shutil.copy(bboxes_path / "bbox.json", cache_path / "bbox.json")

    # Load camera data and generate masks
    with open(cache_path / "transforms.json", "r") as file:
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
    with open(cache_path / "transforms.json", "w") as file:
        json.dump(camera_data, file, indent=4)
    
    # Load bboxes
    with open(bboxes_path / "bbox.json", "r") as file:
        bboxes = json.load(file)['bboxes']

    collision_groups = get_collision_groups(bboxes)

    # Select views for inpainting
    radius = 0.5
    num_layers = 8
    
    selected_views = []
    for i, group in enumerate(collision_groups):
        boxes_centre = np.mean(np.vstack([box["data"]["position"] for box in bboxes if box['idx'] in group]), axis=0)
        dedicated_transforms = generate_hemisphere_vectors(radius, num_layers, boxes_centre)

        group_views = []
        for j, dedicated_transform in enumerate(dedicated_transforms):
            num = str((j + 1) + (len(dedicated_transforms) * i))
            while len(num) < 4:
                num = '0' + num 

            frame = {}
            frame["file_path"] = f"images/dedicated_{num}.png"
            frame["mask_path"] = f"masks/dedicated_{num}.png"
            frame["transform_matrix"] = dedicated_transform.tolist()

            camera_data["frames"].append(frame)
            group_views.append(f"dedicated_{num}.png")

        selected_views.append(group_views) 

    generate_masks(camera_data=camera_data, 
                   bboxes=bboxes, 
                   output_path=cache_path, 
                   dataparser_transform=dataparser_transform, 
                   dataparser_scale=dataparser_scale)

    # Update transforms.json with dedicated frames for training the sparse NeRF network
    with open(cache_path / "transforms.json", "w") as file:
        json.dump(camera_data, file, indent=4)

    # Get cameras for rendering
    dataparser_config.data = cache_path
    cameras, image_names = get_cameras(dataparser_config)

    # Pair cameras with appriopriate indices
    selected_frames = []
    for view_group in selected_views:
        frames = []
        for view in view_group:
            frames.append((view, image_names.index(view)))
        selected_frames.append(frames)

    # # Inpaint selected images
    inpaint_images_in_grids(model, sam_predictor, inpainting_pipeline, selected_frames, cameras, cache_path, log_inpaint_images=True, logs_path=logs_path)

    # Remove temporary masks
    # shutil.rmtree(cache_path / "temp_masks")


def generate_hemisphere_vectors(radius: float, num_layers: int, origin: np.ndarray) -> List[npt.NDArray]:
    transformation_matrices = []
    
    thetas = np.linspace(0.0, np.pi / 2, num_layers + 2)
    phi_offsets = np.linspace(0.0, np.pi / 2, num_layers, endpoint=False)
    for i, theta in enumerate(thetas[1:-1]):
        for j in range(4):
            
            phi = ((np.pi / 2) * j) + phi_offsets[i] 

            x = (np.cos(theta) * np.sin(-phi) * radius) + origin[0]
            y = (np.cos(theta) * np.cos(-phi) * radius) + origin[1]
            z = (np.sin(theta) * radius) + origin[2]

            rot_1 = Rotation.from_euler('XYZ', [-np.pi / 2, -phi, 0.0], degrees=False)
            rot_2 = Rotation.from_euler('XYZ', [theta, 0.0, 0.0], degrees=False)
            rot_3 = Rotation.from_euler('XYZ', [0.0, 0.0, np.pi], degrees=False)
            R = (rot_1 * rot_2 * rot_3).as_matrix()

            trans_mat = np.eye(4)
            trans_mat[:3, 3:] = np.array([[x, y, z]]).T
            trans_mat[:3, :3] = R

            transformation_matrices.append(trans_mat)
    
    return transformation_matrices


def generate_masks(camera_data: Dict[str, float], 
                   bboxes: Dict[str, np.ndarray], 
                   output_path: Path,
                   dataparser_transform: torch.Tensor, 
                   dataparser_scale: torch.Tensor):
    
    # Create masks folder
    os.makedirs(output_path / "masks", exist_ok=True)
    os.makedirs(output_path / "temp_masks", exist_ok=True)

    # Extract camera parameters
    fl_x = camera_data["fl_x"]
    fl_y = camera_data["fl_y"]
    cx = camera_data["cx"]
    cy = camera_data["cy"]
    width = camera_data["w"]
    height = camera_data["h"]
    
    if "applied_transform" in camera_data:
        dataparser_transform = dataparser_transform.cpu().detach().numpy() @ np.linalg.inv(np.concatenate([np.array(camera_data["applied_transform"]), np.array([[0, 0, 0, 1]])], axis=0))

    else:
        dataparser_transform = dataparser_transform.cpu().detach().numpy()

    # Camera matrix
    camera_matrix = np.array([[fl_x, 0.0, cx],
                              [0.0, fl_y, cy],
                              [0.0, 0.0, 1.0]])

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

        uv_points = []
        for world_point in boxes_points[0]:

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

        mask = np.zeros((height, width))
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

    print(a)