import cv2
import os
import torch
import shutil

import numpy as np
import torch.nn as nn

from dataclasses import dataclass, field
from diffusers import AutoPipelineForInpainting
from segment_anything import SamPredictor
from pathlib import Path
from typing import Type, List, Tuple, Optional, Union
from rich.progress import track
from PIL import Image

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.models.base_model import Model
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.cameras.cameras import Cameras

from neo.utils.rendering import render_preview_image
from neo.utils.sam_patch import sam_model_registry


@dataclass
class InpainterConfig(InstantiateConfig):
    """Configuration for inpainting"""

    _target: Type = field(default_factory=lambda: Inpainter)
    """target class to instantiate"""
    sam_checkpoint: str = "sam_vit_h_4b8939.pth"
    """name of the SAM checkpoint"""
    sam_model_type: str = "vit_h"
    """SAM model type, vit_h by default"""
    inpainting_model: str = "runwayml/stable-diffusion-inpainting"
    """model used for inpainting"""
    inpainting_images_base_size: Tuple[int, int] = (512, 512)
    """base size of inpainting images"""
    log_inpaint_images: bool = True
    """log inpaint images"""
    render_batch_size: int = 4
    """batch size for rendering images"""
    inpaint_batch_size: int = 8
    """batch size for inpainting images"""
    inpainting_prompt: str = "highly detailed, 4K, hdr, sharp focus, image"
    """prompt for inpainting images"""


class Inpainter(nn.Module):

    def __init__(self, config: InpainterConfig, device: str = "cpu", output_path: Optional[Path] = None) -> None:
        super().__init__()
        self.config = config
        self.output_path = output_path
        self.device = device

        # Create log paths or replace existing logs
        if self.config.log_inpaint_images:
            assert self.output_path is not None, "Output path must be provided for logging inpaint images"
            self.logs_path = self.output_path / "logs"

            if os.path.exists(self.logs_path) and len(os.listdir(self.logs_path)) > 1:
                shutil.rmtree(self.logs_path)

            os.makedirs(self.logs_path, exist_ok=True)

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

    @staticmethod
    def refine_mask(input_masks: np.ndarray, rendered_images: torch.Tensor, sam_predictor: SamPredictor) -> np.ndarray:

        # Generate features for rendered image
        rendered_images = rendered_images.permute(0, 3, 1, 2)
        original_image_size = rendered_images.shape[-2:]
        resized_images = torch.nn.functional.interpolate(rendered_images, size=(1024, 1024), mode='bilinear', align_corners=False)
        resized_images = resized_images.to(torch.uint8)
        sam_predictor.set_torch_image(resized_images, original_image_size=original_image_size)

        # Create a set of prompt points around objects visible on the masks
        input_points_batch = []
        input_labels_batch = []

        for input_mask in input_masks:
            edges = cv2.Canny(input_mask, threshold1=100, threshold2=200)
            input_points = np.array(np.where(edges == 255)).transpose()[:, ::-1]
            indices = np.linspace(0, input_points.shape[0] - 1, 100, dtype=int)
            input_points = input_points[indices]
            input_labels = np.ones_like(input_points[:, 0])

            input_points_batch.append(input_points)
            input_labels_batch.append(input_labels)

        input_points = torch.tensor(np.concatenate(input_points_batch, axis=0), dtype=torch.float32).reshape(rendered_images.shape[0], 100, 1, -1).permute(1, 0, 2, 3)
        scale_x = torch.tensor(1024 / original_image_size[0], dtype=torch.float32)
        scale_y = torch.tensor(1024 / original_image_size[1], dtype=torch.float32)
        input_points[..., 0] *= scale_x
        input_points[..., 1] *= scale_y
        input_labels = torch.tensor(np.concatenate(input_labels_batch, axis=0), dtype=torch.float32).reshape(rendered_images.shape[0], 100, -1).permute(1, 0, 2)

        # Predict masks for the points and refine input mask
        masks = []
        with torch.no_grad():
            for input_point, input_label in zip(input_points, input_labels):
                mask, _, _ = sam_predictor.predict_torch(
                    point_coords=input_point.to(device=sam_predictor.device),
                    point_labels=input_label.to(device=sam_predictor.device),
                    multimask_output=False,
                )

                masks.append(mask)

        # Release GPU memory
        sam_predictor.reset_image()

        masks = torch.concat(masks, dim=1)
        sum_masks = np.clip(masks.sum(dim=1).cpu().detach().numpy(), 0, 1)

        refined_masks = []
        for sum_mask in sum_masks:
            sum_mask = cv2.resize(sum_mask, (input_mask.shape[1], input_mask.shape[0]), cv2.INTER_MAX)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            sum_mask = cv2.morphologyEx((sum_mask * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)[None, ...] / 255

            refined_mask = np.array(np.logical_and(np.logical_xor(sum_mask[0], input_mask), input_mask)).astype(np.uint8) * 255
            refined_mask = cv2.dilate(refined_mask, kernel)
            refined_masks.append(refined_mask)

        return refined_masks

    @staticmethod
    def get_images_from_grid(grid_image: np.ndarray, size: Tuple[int, int]) -> List[np.ndarray]:

        # Split the grid image into 4 images
        height, width = int(size[0] / 2), int(size[1] / 2)
        images = []
        for i in range(2):
            for j in range(2):
                image = grid_image[i * width: (i + 1) * width, j * height: (j + 1) * height]
                images.append(image)

        return images

    @staticmethod
    def get_image_grid(images: List[Union[np.ndarray, torch.Tensor]]) -> np.ndarray:

        # Check if the images are torch tensors and convert them to numpy arrays if necessary
        converted_images = []
        for image in images:
            if isinstance(image, torch.Tensor):
                converted_images.append(image.cpu().detach().numpy())
            else:
                converted_images.append(image)
        images = converted_images

        # Create a grid of images
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


    def inpaint_images_in_grids(self,
                                model: Model,
                                selected_frames: List[Tuple[np.ndarray, int]], 
                                cameras: Cameras,
                                output_path: Path,
                                log_name: str = "inpainting",
                                rendering_output_name: str = 'rgb_rm_preview', 
                                strength: int = 0.99,
                                # shuffle_frames: bool = True,
                                refine_masks: bool = False,
                                initial_inpaint: bool = False) -> None:
        
        # Create output directories
        renders_output_path = output_path / "renders"
        if initial_inpaint:
            os.makedirs(renders_output_path, exist_ok=True)

        # # Shuffle frames
        # if shuffle_frames:
        #     for group in selected_frames:
        #         np.random.shuffle(group)

        # Inpaint selected images
        for idx, images_group in enumerate(selected_frames):
            if self.config.log_inpaint_images:
                group_logs_path = self.logs_path / log_name / f"{idx + 1}"
                os.makedirs(group_logs_path, exist_ok=True)

            all_images, all_masks = [], []
            all_image_names = []
            render_bath_size = self.config.render_batch_size
            for i in track(range(int(len(images_group) / render_bath_size)), description=f"Rendering images for group {idx + 1}", transient=False):
                idx_list = []
                masks = []
                image_names = []
                hq_images = []
                for j in range(render_bath_size):
                    idx_list.append(images_group[j + (render_bath_size * i)][1])
                    mask = cv2.imread(str(output_path / f"temp_masks/{images_group[j + (render_bath_size * i)][0]}"), cv2.IMREAD_GRAYSCALE)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                    mask = cv2.dilate(mask, kernel, iterations=2)
                    downscale_factor = mask.shape[0] / cameras[j + (render_bath_size * i)].width
                    mask = cv2.resize(mask, (int(mask.shape[1] / downscale_factor), int(mask.shape[0] / downscale_factor)), cv2.INTER_MAX)
                    masks.append(mask)
                    image_names.append(images_group[j + (render_bath_size * i)][0])

                    if not initial_inpaint:
                        hq_images.append(cv2.imread(renders_output_path / images_group[j + (render_bath_size * i)][0]))

                images = render_preview_image(model, cameras, torch.tensor(idx_list), output_name=rendering_output_name)

                refined_masks = masks
                if refine_masks:
                    refined_masks = self.refine_mask(masks, images, self.sam_predictor)

                if not initial_inpaint:
                    hq_images_tensor = torch.tensor(hq_images, device=self.device)
                    masks_tensor = torch.tensor(masks, device=self.device)[..., None] / 255.0

                    images = images * masks_tensor + hq_images_tensor * (1 - masks_tensor)

                all_images.extend(images)
                all_masks.extend(refined_masks)
                all_image_names.extend(image_names)

                if initial_inpaint:
                    for image, mask, refined_mask, name in zip(images, masks, refined_masks, image_names):
                        cv2.imwrite(str(renders_output_path / name), image.cpu().detach().numpy())

                for image, mask, refined_mask, name in zip(images, masks, refined_masks, image_names):
                    mask = cv2.resize(mask, (int(mask.shape[1] * downscale_factor), int(mask.shape[0] * downscale_factor)), cv2.INTER_MAX)
                    refined_mask = cv2.resize(refined_mask, (int(refined_mask.shape[1] * downscale_factor), int(refined_mask.shape[0] * downscale_factor)), cv2.INTER_MAX)
                    cv2.imwrite(str(output_path / f"masks/{name}"), mask)

                    if self.config.log_inpaint_images:
                        cv2.imwrite(str(group_logs_path / name), image.cpu().detach().numpy())
                        cv2.imwrite(str(group_logs_path / f"mask_{name}"), mask)
                        cv2.imwrite(str(group_logs_path / f"refined_mask_{name}"), refined_mask)

            image_grids = self.get_image_grid(all_images)
            mask_grids = self.get_image_grid(all_masks)
            image_names = np.array(all_image_names).reshape(len(all_image_names) // 4, -1)
            img_old_size = (int(image_grids.shape[2] * downscale_factor), int(image_grids.shape[1] * downscale_factor))

            for i in range(int(len(image_grids) / self.config.inpaint_batch_size)):
                image_range = slice(i * self.config.inpaint_batch_size, (i + 1) * self.config.inpaint_batch_size)

                images_list = []
                masks_list = []
                for images_grid, masks_grid in zip(image_grids[image_range], mask_grids[image_range]):
                    images_grid = Image.fromarray(images_grid)
                    masks_grid = Image.fromarray(masks_grid)

                    if images_grid.size != self.config.inpainting_images_base_size:
                        images_grid = images_grid.resize(self.config.inpainting_images_base_size)
                        masks_grid = masks_grid.resize(self.config.inpainting_images_base_size)

                    images_list.append(images_grid)
                    masks_list.append(masks_grid)

                CONSOLE.print(f"Inpainting images in group {idx + 1}")
                reconstructed_grid_images = self.inpainting_pipeline(prompt=[self.config.inpainting_prompt] * len(images_list), image=images_list, mask_image=masks_list, strength=strength).images

                for reconstructed_grid_image, img_name in zip(reconstructed_grid_images, image_names[image_range]):
                    reconstructed_grid_image = reconstructed_grid_image.resize(img_old_size)
                    reconstructed_grid_image = np.array(reconstructed_grid_image)
                    reconstructed_images = self.get_images_from_grid(reconstructed_grid_image, img_old_size)

                    for image, name in zip(reconstructed_images, img_name):
                        cv2.imwrite(str(output_path / f"images/{name}"), image)
                    
                    if self.config.log_inpaint_images:
                        for image, name in zip(reconstructed_images, img_name):
                            cv2.imwrite(str(group_logs_path / f"reconstructed_{name}"), image)