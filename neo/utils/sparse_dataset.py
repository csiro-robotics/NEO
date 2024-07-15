import cv2
import json
import os
import torch
import open3d as o3d
import numpy as np

from PIL import Image
from typing import Dict, Any
from pathlib import Path

from nerfstudio.models.base_model import Model
from nerfstudio.cameras.cameras import Cameras

from neo.utils.cameras import get_cameras


def generate_sparse_dataset(model: Model,
                            dataparser_config: Dict[str, Any],
                            data_path: Path, 
                            bboxes_path: Path, 
                            dataparser_transform: torch.Tensor, 
                            dataparser_scale: torch.Tensor) -> None:

    print("Generating Sparse Data!")

    # Load camera data
    with open(data_path / "transforms.json", "r") as file:
        camera_data = json.load(file)
    
    generate_masks(camera_data, bboxes_path, dataparser_transform, dataparser_scale)
    print(a)

    cameras = get_cameras(dataparser_config)
    inpaint_images(model, cameras)



def inpaint_images(model: Model, cameras: Cameras) -> None:

    print("Inpainting images")

    for camera in cameras:
        with torch.no_grad():
            outputs = model.get_outputs_for_camera(camera)

        print(outputs)

        print(a)

    print(outputs)


def generate_masks(camera_data: Dict[str, float], 
                   bboxes_path: Path, 
                   dataparser_transform: torch.Tensor, 
                   dataparser_scale: torch.Tensor) -> None:
    
    # Load bbox data
    with open(bboxes_path, "r") as file:
        bboxes = json.load(file)

    # Create masks folder
    os.makedirs(bboxes_path.parent / "masks", exist_ok=True)

    # Extract camera parameters
    fl_x = camera_data["fl_x"]
    fl_y = camera_data["fl_y"]
    cx = camera_data["cx"]
    cy = camera_data["cy"]
    width = camera_data["w"]
    height = camera_data["h"]

    # Create PinholeCameraIntrinsic object
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fl_x, fl_y, cx, cy)

    # Create camera parameters
    camera_params = o3d.camera.PinholeCameraParameters()
    camera_params.intrinsic = intrinsic

    # Create window
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.get_render_option().background_color = np.asarray([0, 0, 0])

    # Add boxes to visualization
    for idx, bbox in enumerate(bboxes['bboxes']):
        bbox_data = bbox['data']
        offset = 0.0
        bbox_w, bbox_h, bbox_d  = np.array(bbox_data['dimensions']) - offset

        mesh_box = o3d.geometry.TriangleMesh.create_box(width=bbox_w,
                                                        height=bbox_h,
                                                        depth=bbox_d)
        mesh_box.translate(np.array([-bbox_w / 2, -bbox_h / 2, -bbox_d / 2]))
        color = (idx + 10) / 255
        mesh_box.paint_uniform_color((color, color, color))

        position = np.array(bbox_data['position']) - dataparser_transform[:, 3].cpu().detach().numpy()

        mesh_box.translate(position)

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
        
        mesh_box.rotate(R.T)

        vis.add_geometry(mesh_box)

    # Iterate through frames
    for i, frame in enumerate(camera_data["frames"]):

        num = str(i)
        while len(num) < 4:
            num = '0' + num

        # Extract transformation matrix
        transform_matrix = np.array(frame["transform_matrix"])

        # Convert transformation matrix to camera extrinsic parameters
        extrinsic = np.linalg.inv(transform_matrix)
        camera_params.extrinsic = extrinsic @ np.diag([-1, -1, -1, -1])

        # Set camera parameters to view control of the visualizer
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(camera_params, True)

        vis.poll_events()
        vis.update_renderer()

        # Save images
        img = vis.capture_screen_float_buffer(do_render=True)
        img = (255.0 * np.asarray(img)).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, thr = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        cnt, _ = cv2.findContours(thr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

        print(i, Path(frame['file_path']).name, np.unique(gray), len(cnt))
        cv2.imwrite(f"{bboxes_path.parent}/masks/{Path(frame['file_path']).name}", thr)

    vis.destroy_window()