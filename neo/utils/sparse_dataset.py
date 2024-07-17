import cv2
import json
import os
import torch
import random
import itertools
import open3d as o3d
import numpy as np

from PIL import Image
from typing import Dict, List, Any
from collections import defaultdict
from pathlib import Path

from nerfstudio.models.base_model import Model
from nerfstudio.cameras.cameras import Cameras

from neo.utils.cameras import get_cameras
from neo.utils.collisions import OBB, check_obb_collision


def generate_sparse_dataset(model: Model,
                            data_path: Path, 
                            bboxes_path: Path, 
                            dataparser_config: Dict[str, Any],
                            dataparser_transform: torch.Tensor, 
                            dataparser_scale: torch.Tensor) -> None:

    print("Generating Sparse Data!")

    # Load camera data
    with open(data_path / "transforms.json", "r") as file:
        camera_data = json.load(file)
    
    cameras, image_names = get_cameras(dataparser_config)
    collision_groups, masks_metadata = generate_masks(camera_data, image_names, bboxes_path, dataparser_transform, dataparser_scale)

    # Select random frames for inpainting
    random_images = []
    for group in collision_groups:
        frames = []
        threshold = 0
        while len(frames) < 4 and threshold < 1000:
            random_frame = random.choice(list(masks_metadata.items()))
            
            visible_boxes = random_frame[1]
            if len(visible_boxes) > 0:
                random.shuffle(visible_boxes)
            else:
                continue

            for box in visible_boxes:
                if box in group and random_frame[0] not in frames:
                    frames.append(image_names.index(random_frame[0]))
                    break

            threshold += 1

        random_images.append(frames)

        if threshold == 1000:
            raise TimeoutError("Object designated for removal is not visible in at least four frames!")

    render_preview_images(model, cameras, random_images)

    print(a)


def render_preview_images(model: Model, cameras: Cameras, random_images: List[int]) -> None:

    print("Inpainting images")

    for images_group in random_images:
        for image_idx in images_group:
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(cameras[image_idx])

            rgb_preview = cv2.cvtColor((outputs['rgb_preview'].cpu().detach().numpy() * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"rgb_preview_{image_idx}.png", rgb_preview)


def generate_masks(camera_data: Dict[str, float], 
                   image_names: List[str],
                   bboxes_path: Path, 
                   dataparser_transform: torch.Tensor, 
                   dataparser_scale: torch.Tensor) -> None:
    
    # Load bbox data
    with open(bboxes_path / "bbox.json", "r") as file:
        bboxes = json.load(file)

    # Create masks folder
    os.makedirs(bboxes_path / "masks", exist_ok=True)

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
    obbs = []
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

        obb = OBB(center=position, axes=R, half_sizes=np.array([-bbox_w / 2, -bbox_h / 2, -bbox_d / 2]))
        obbs.append(obb)
        vis.add_geometry(mesh_box)

    # Detect collisions between bounding boxes
    combinations = list(itertools.combinations(np.arange(len(obbs)), 2))

    collision_graph = defaultdict(set)
    for comb in combinations:
        collision = check_obb_collision(obbs[comb[0]], obbs[comb[1]])
        if collision:
            collision_graph[comb[0]].add(comb[1])
            collision_graph[comb[1]].add(comb[0])

    def find_connected_components(graph):
        visited = set()
        components = []

        def dfs(node, component):
            stack = [node]
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    component.append(current)
                    stack.extend(graph[current] - visited)

        for node in graph:
            if node not in visited:
                component = []
                dfs(node, component)
                components.append(component)

        return components

    # Get the groups of colliding bounding boxes
    collision_groups = find_connected_components(collision_graph)

    for bbox_idx in range(len(obbs)):
        skip = False
        for group in collision_groups:
            if bbox_idx in group:
                skip = True

        if not skip:
            collision_groups.append([bbox_idx])

    # Iterate through frames
    masks_metadata = {}
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

        cv2.imwrite(f"{bboxes_path}/masks/{Path(frame['file_path']).name}", thr)

        if Path(frame['file_path']).name not in image_names:
            continue

        masks_metadata[Path(frame['file_path']).name] = (np.array(np.unique(gray)[1:]) - 10).tolist()

    vis.destroy_window()

    return collision_groups, masks_metadata