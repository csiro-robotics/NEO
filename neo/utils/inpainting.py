import cv2
import os
import numpy as np
import numpy.typing as npt

from segment_anything import SamPredictor
from pathlib import Path
from typing import List, Tuple, Any
from rich.progress import track
from PIL import Image

from nerfstudio.models.base_model import Model
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.cameras.cameras import Cameras

from neo.utils.rendering import render_preview_image


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


def get_images_from_grid(grid_image: npt.NDArray, size: Tuple[int, int]) -> List[npt.NDArray]:

    height, width = int(size[0] / 2), int(size[1] / 2)
    images = []
    for i in range(2):
        for j in range(2):
            image = grid_image[i * width: (i + 1) * width, j * height: (j + 1) * height]
            images.append(image)

    return images


def get_image_grid(images: List[npt.NDArray]) -> npt.NDArray:

    batch_size = len(images) // 4
    width = images[0].shape[0]
    height = images[0].shape[1]
    if len(images[0].shape) > 2:
        image_grids = np.zeros((batch_size, 2 * width, 2 * height, 3), dtype=np.uint8)
    else:
        image_grids = np.zeros((batch_size, 2 * width, 2 * height), dtype=np.uint8)

    for bs in range(batch_size):
        for i in range(2):
            for j in range(2):
                image_grids[bs, i * width:(i+1) * width, j * height:(j + 1) * height] = images[i * 2 + j + (bs * 4)]

    return image_grids


def inpaint_images_in_grids(model: Model,
                            sam_predictor: SamPredictor,
                            inpainting_pipeline: Any,
                            selected_frames: List[Tuple[npt.NDArray, int]], 
                            cameras: Cameras,
                            cache_path: Path,
                            log_inpaint_images: bool = False, 
                            logs_path: Path = None,
                            rendering_output_name: str = 'rgb_rm_preview', 
                            strength: int = 0.99,
                            refine_masks: bool = True):

    # Inpaint selected images
    for idx, images_group in enumerate(selected_frames):
        if log_inpaint_images:
            group_logs_path = logs_path / f"{idx + 1}"
            os.makedirs(group_logs_path, exist_ok=True)

        images, masks = [], []
        image_names = []
        for random_image in track(images_group, description=f"Rendering images for group {idx + 1}", transient=False):
            image = render_preview_image(model, cameras, random_image[1], output_name=rendering_output_name)
            mask = cv2.imread(str(cache_path / f"temp_masks/{random_image[0]}"), cv2.IMREAD_GRAYSCALE)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.dilate(mask, kernel, iterations=2)
            downscale_factor = mask.shape[0] / image.shape[0]
            mask = cv2.resize(mask, (int(mask.shape[1] / downscale_factor), int(mask.shape[0] / downscale_factor)), cv2.INTER_MAX)

            if refine_masks:
                refined_mask = refine_mask(mask, image, sam_predictor)
            else:
                refined_mask = mask

            images.append(image)
            masks.append(refined_mask)
            image_names.append(random_image[0])

            mask = cv2.resize(mask, (int(mask.shape[1] * downscale_factor), int(mask.shape[0] * downscale_factor)), cv2.INTER_MAX)
            refined_mask = cv2.resize(refined_mask, (int(refined_mask.shape[1] * downscale_factor), int(refined_mask.shape[0] * downscale_factor)), cv2.INTER_MAX)
            cv2.imwrite(str(cache_path / f"masks/{random_image[0]}"), mask)

            if log_inpaint_images:
                cv2.imwrite(str(group_logs_path / random_image[0]), image)
                cv2.imwrite(str(group_logs_path / f"mask_{random_image[0]}"), mask)
                cv2.imwrite(str(group_logs_path / f"refined_mask_{random_image[0]}"), refined_mask)

        image_grids = get_image_grid(images)
        mask_grids = get_image_grid(masks)
        image_names = np.array(image_names).reshape(len(image_names) // 4, -1)

        for images_grid, masks_grid, image_name in zip(image_grids, mask_grids, image_names):
            img_old_size = (int(images_grid.shape[1] * downscale_factor), int(images_grid.shape[0] * downscale_factor))
            img_new_size = (1024, 1024)
            images_grid = Image.fromarray(images_grid)
            images_grid = images_grid.resize(img_new_size)
            masks_grid = Image.fromarray(masks_grid)
            masks_grid = masks_grid.resize(img_new_size)

            CONSOLE.print(f"Inpainting images in group {idx + 1}")
            reconstructed_grid_image = inpainting_pipeline(prompt="background", image=images_grid, mask_image=masks_grid, strength=strength).images[0]
            reconstructed_grid_image = reconstructed_grid_image.resize(img_old_size)
            reconstructed_grid_image = np.array(reconstructed_grid_image)
            reconstructed_images = get_images_from_grid(reconstructed_grid_image, img_old_size)

            for image, name in zip(reconstructed_images, image_name):
                cv2.imwrite(str(cache_path / f"images/{name}"), image)
            
            if log_inpaint_images:
                for image, name in zip(reconstructed_images, image_name):
                    cv2.imwrite(str(group_logs_path / f"reconstructed_{name}"), image)