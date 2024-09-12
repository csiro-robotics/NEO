import cv2
import json
import os
import shutil
import torch
import random
import itertools
import open3d as o3d
import numpy as np
import numpy.typing as npt

from PIL import Image
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from pathlib import Path
from rich.progress import track
from segment_anything import SamPredictor

from nerfstudio.models.base_model import Model
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.rich_utils import CONSOLE

from neo.utils.cameras import get_cameras
from neo.utils.collisions import OBB, check_obb_collision


def generate_sparse_dataset(model: Model,
                            sam_predictor: SamPredictor,
                            inpainting_pipeline: Any,
                            data_path: Path, 
                            bboxes_path: Path, 
                            dataparser_config: Dict[str, Any],
                            dataparser_transform: torch.Tensor, 
                            dataparser_scale: torch.Tensor,
                            log_inpaint_images: bool = False,
                            full_random: bool = False) -> None:

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

    all_image_names = [Path(frame['file_path']).name for frame in camera_data["frames"]]
    cameras, image_names = get_cameras(dataparser_config)
    print("IMAGE NAMES")
    print(image_names)
    invalid_cameras = [idx for idx, img_name in enumerate(all_image_names) 
                       if img_name not in image_names]
    
    collision_groups, masks_metadata, cam_to_meta = generate_masks(camera_data=camera_data, 
                                                      image_names=image_names, 
                                                      bboxes_path=bboxes_path, 
                                                      output_path=cache_path, 
                                                      dataparser_transform=dataparser_transform, 
                                                      dataparser_scale=dataparser_scale)

    # Find the centre of the bbox and points 5 units (arbitrary) in each direction
    with open(bboxes_path / "bbox.json", "r") as file:
        bboxes = json.load(file)['bboxes']
    
    # Select random frames for inpainting
    random_images = []
    for group in collision_groups:
        frames = []
        threshold = 0
        if not full_random:
            # Take mean of all bboxes present in the file. Best of worlds
            boxes_centre = np.mean(np.vstack([box["data"]["position"] for box in bboxes if box['idx'] in group]), axis=0)

            # Define axes we want cameras to be near
            # TODO remove hard coding
            forward = boxes_centre + [0,5,0]
            top = boxes_centre + [0,0,5]
            left = boxes_centre + [10,5,5]
            right = boxes_centre + [-10,5,5]

            # Calculate distances of all frames to all axes
            # TODO check sign of cam_positions (should be negative what is here?)
            # TODO should replace with cameras not camera_data?
            cam_positions = np.vstack([np.array(frame["transform_matrix"])[:3,3].flatten() for frame in camera_data["frames"]])
            print(cam_positions.shape)
            # TODO validate this
            forward_diffs = np.linalg.norm(np.cross(forward - boxes_centre, boxes_centre - cam_positions), axis=1)/np.linalg.norm(forward - boxes_centre)
            forward_diffs[invalid_cameras] = 1000
            best_forwards = np.argsort(forward_diffs)
            top_diffs = np.linalg.norm(np.cross(top - boxes_centre, boxes_centre - cam_positions), axis=1)/np.linalg.norm(top - boxes_centre)
            top_diffs[invalid_cameras] = 1000
            best_tops = np.argsort(top_diffs)
            left_diffs = np.linalg.norm(np.cross(left - boxes_centre, boxes_centre - cam_positions), axis=1)/np.linalg.norm(left - boxes_centre)
            left_diffs[invalid_cameras] = 1000
            best_lefts = np.argsort(left_diffs)
            right_diffs = np.linalg.norm(np.cross(right - boxes_centre, boxes_centre - cam_positions), axis=1)/np.linalg.norm(right - boxes_centre)
            right_diffs[invalid_cameras] = 1000
            best_rights = np.argsort(right_diffs)
            
            bests_lists = [best_forwards, best_tops, best_lefts, best_rights]
            print("BESTS LISTS")
            print(bests_lists)
            current_bests_idxs = [0,0,0,0]
        print("MASKED METADATA")
        print(list(masks_metadata.items()))
        print(len(list(masks_metadata.items())))
        while len(frames) < 4 and threshold < 1000:
            if full_random:
                chosen_frame = random.choice(list(masks_metadata.items()))
                print(f"chosen_frame: {chosen_frame}")
            else:
                # Get the idx of the closest frame to the current considered viewing direction
                numf = len(frames)
                best_camera_data_idx = bests_lists[numf][current_bests_idxs[numf]]
                best_meta_idx = cam_to_meta[best_camera_data_idx]
                print("BEST CAMERA DATA IDX")
                print(best_camera_data_idx)
                chosen_frame = list(masks_metadata.items())[best_meta_idx]
                print(f"chosen_frame: {chosen_frame}")
                current_bests_idxs[numf] += 1

            visible_boxes = chosen_frame[1]
            if len(visible_boxes) > 0:
                random.shuffle(visible_boxes)
            else:
                continue

            for box in visible_boxes:
                if box in group and chosen_frame[0] not in frames:
                    frames.append((chosen_frame[0], image_names.index(chosen_frame[0])))
                    break

            threshold += 1

        random_images.append(frames)

        if threshold == 1000:
            raise TimeoutError("Object designated for removal is not visible in at least four frames!")
    
    # Inpaint selected images
    for idx, images_group in enumerate(random_images):
        if log_inpaint_images:
            group_logs_path = logs_path / f"{idx + 1}"
            os.makedirs(group_logs_path)

        images, masks = [], []
        image_names = []
        for random_image in track(images_group, description=f"Rendering images for group {idx + 1}", transient=False):
            image = render_preview_image(model, cameras, random_image)
            mask = cv2.imread(str(cache_path / f"temp_masks/{random_image[0]}"), cv2.IMREAD_GRAYSCALE)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.dilate(mask, kernel, iterations=2)
            downscale_factor = mask.shape[0] / image.shape[0]
            mask = cv2.resize(mask, (int(mask.shape[1] / downscale_factor), int(mask.shape[0] / downscale_factor)), cv2.INTER_MAX)
            refined_mask = refine_mask(mask, image, sam_predictor)

            images.append(image)
            masks.append(refined_mask)
            image_names.append(random_image[0])

            mask = cv2.resize(mask, (int(mask.shape[1] * downscale_factor), int(mask.shape[0] * downscale_factor)), cv2.INTER_MAX)
            refined_mask = cv2.resize(refined_mask, (int(refined_mask.shape[1] * downscale_factor), int(refined_mask.shape[0] * downscale_factor)), cv2.INTER_MAX)
            cv2.imwrite(str(cache_path / f"masks/{random_image[0]}"), mask)

            if log_inpaint_images:
                cv2.imwrite(str(group_logs_path / random_image[0]), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                cv2.imwrite(str(group_logs_path / f"mask_{random_image[0]}"), mask)
                cv2.imwrite(str(group_logs_path / f"refined_mask_{random_image[0]}"), refined_mask)

        images_grid = get_image_grid(images)
        masks_grid = get_image_grid(masks)

        img_old_size = (int(images_grid.shape[1] * downscale_factor), int(images_grid.shape[0] * downscale_factor))
        img_new_size = (1024, 1024)
        images_grid = Image.fromarray(images_grid)
        images_grid = images_grid.resize(img_new_size)
        masks_grid = Image.fromarray(masks_grid)
        masks_grid = masks_grid.resize(img_new_size)

        CONSOLE.print(f"Inpainting images in group {idx + 1}")
        reconstructed_grid_image = inpainting_pipeline(prompt="background", image=images_grid, mask_image=masks_grid).images[0]
        reconstructed_grid_image = reconstructed_grid_image.resize(img_old_size)
        reconstructed_grid_image = np.array(reconstructed_grid_image)
        reconstructed_images = get_images_from_grid(reconstructed_grid_image, img_old_size)

        for image, name in zip(reconstructed_images, image_names):
            cv2.imwrite(str(cache_path / f"images/{name}"), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if log_inpaint_images:
            for image, name in zip(reconstructed_images, image_names):
                cv2.imwrite(str(group_logs_path / f"reconstructed_{name}"), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Remove temporary masks
    shutil.rmtree(cache_path / "temp_masks")


def get_images_from_grid(grid_image: npt.NDArray, size: Tuple[int, int]) -> List[npt.NDArray]:

    height, width = int(size[0] / 2), int(size[1] / 2)
    images = []
    for i in range(2):
        for j in range(2):
            image = grid_image[i * width: (i + 1) * width, j * height: (j + 1) * height]
            images.append(image)

    return images


def get_image_grid(images: List[npt.NDArray]) -> npt.NDArray:

    width = images[0].shape[0]
    height = images[0].shape[1]
    if len(images[0].shape) > 2:
        images_grid = np.zeros((2 * width, 2 * height, 3), dtype=np.uint8)
    else:
        images_grid = np.zeros((2 * width, 2 * height), dtype=np.uint8)

    for i in range(2):
        for j in range(2):
            images_grid[i * width:(i+1) * width, j * height:(j + 1) * height] = images[i * 2 + j]

    return images_grid


def refine_mask(input_mask: npt.NDArray, rendered_image: npt.NDArray, sam_predictor: SamPredictor) -> npt.NDArray:

    # Generate features for rendered image
    sam_predictor.set_image(rendered_image)

    # Create a set of prompt points around objects visible on the masks
    input_mask = input_mask
    edges = cv2.Canny(input_mask, threshold1=100, threshold2=200)
    input_points = np.array(np.where(edges == 255)).transpose()[:, ::-1]
    indices = np.linspace(0, input_points.shape[0] - 1, 100, dtype=int)
    input_points = input_points[indices]
    input_labels = np.ones_like(input_points[:, 0])

    # Predict masks for the points and refine input mask
    masks = []
    for input_point, input_label in zip(input_points, input_labels):
        mask, _, _ = sam_predictor.predict(
            point_coords=np.array([input_point]),
            point_labels=np.array([input_label]),
            multimask_output=False,
        )
        masks.append(mask[0])

    masks = np.array(masks)

    sum_masks = np.clip(masks.sum(0), 0, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    sum_masks = cv2.morphologyEx((sum_masks * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)[None, ...] / 255

    refined_mask = np.array(np.logical_and(np.logical_xor(sum_masks[0], input_mask), input_mask)).astype(np.uint8) * 255
    refined_mask = cv2.dilate(refined_mask, kernel)

    return refined_mask


def render_preview_image(model: Model, cameras: Cameras, random_image: List[int]) -> npt.NDArray:

    image_idx = random_image[1]
    with torch.no_grad():
        outputs = model.get_outputs_for_camera(cameras[image_idx])

    rgb_preview = (outputs['rgb_rm_preview'].cpu().detach().numpy() * 255).astype(np.uint8)
    
    return rgb_preview

def generate_masks(camera_data: Dict[str, float], 
                   image_names: List[str],
                   bboxes_path: Path, 
                   output_path: Path,
                   dataparser_transform: torch.Tensor, 
                   dataparser_scale: torch.Tensor) -> None:
    cam_data_to_meta_idx = {}
    # Load bbox data
    with open(bboxes_path / "bbox.json", "r") as file:
        bboxes = json.load(file)

    # Create masks folder
    os.makedirs(output_path / "masks", exist_ok=True)
    os.makedirs(output_path / "temp_masks", exist_ok=True)

    # Extract camera parameters
    fl_x = camera_data["fl_x"]
    fl_y = camera_data["fl_y"]
    cx = camera_data["cx"]
    cy = camera_data["cy"]
    width = camera_data["w"] * 2
    height = camera_data["h"] * 2
    
    if "applied_transform" in camera_data:
        dataparser_transform = dataparser_transform.cpu().detach().numpy() @ np.linalg.inv(np.concatenate([np.array(camera_data["applied_transform"]), np.array([[0, 0, 0, 1]])], axis=0))

    else:
        dataparser_transform = dataparser_transform.cpu().detach().numpy()

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
    CONSOLE.print("Reading bbox data")
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

        position = np.array(bbox_data['position'])
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
    
    CONSOLE.print(f"Found {len(collision_groups)} groups of bounding boxes")

    # Iterate through frames
    masks_metadata = {}
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
        camera_params.extrinsic = extrinsic @ np.diag([-1, -1, -1, -1])

        # Set camera parameters to view control of the visualizer
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(camera_params, True)

        vis.poll_events()
        vis.update_renderer()

        # Render image
        img = vis.capture_screen_float_buffer(do_render=True)
        img = (255.0 * np.asarray(img)).astype(np.uint8)

        # Crop image according to the principal point
        width = camera_data["w"]
        height = camera_data["h"]

        img = img[round((height / 2) + (height / 2) - cy):round(((height / 2) + (height / 2) - cy) + height), round((width / 2) + (width / 2) - cx):round(((width / 2) + (width / 2) - cx) + width)]

        # Apply binary thresholding
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

        # Save images
        cv2.imwrite(f"{output_path}/temp_masks/{Path(frame['file_path']).name}", thr)

        kernel = np.ones((3, 3), np.uint8)
        dilated_image = cv2.dilate(thr, kernel, iterations=150)
        dilated_image -= thr

        cv2.imwrite(f"{output_path}/masks/{Path(frame['file_path']).name}", dilated_image)

        if Path(frame['file_path']).name not in image_names:
            continue
        cam_data_to_meta_idx[i] = len(masks_metadata.keys())
        masks_metadata[Path(frame['file_path']).name] = (np.array(np.unique(gray)[1:]) - 10).tolist()

    vis.destroy_window()

    print("MASKS METADATA")
    print(masks_metadata)

    # TODO VERY BAD PRACTICE FOR CAM DATA TO META IDX
    return collision_groups, masks_metadata, cam_data_to_meta_idx