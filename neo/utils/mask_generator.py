import cv2
import os
import torch

import numpy as np

from typing import Dict
from pathlib import Path
from rich.progress import track

from nerfstudio.utils.rich_utils import CONSOLE

from neo.utils.sam_patch import sam_model_registry


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
