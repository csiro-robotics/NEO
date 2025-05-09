"""
Sparse-Guided NeRF Model

This model is utilized by Neo for editing purposes. It learns region reconstruction from sparse inpainted dataset.
"""
import torch
import json
import cv2
import numpy as np

from collections import defaultdict
from dataclasses import dataclass, field
from rich.progress import track
from pathlib import Path
from typing import Optional, Type, List, Dict, Any

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames

from nerfstudio.model_components.losses import (
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)

from neo.utils.bbox import BoundingBox
from neo.utils.bbox_utils import BBoxData
from neo.utils.cameras import get_cameras
from neo.utils.inpainting import Inpainter


@dataclass
class SparseGuidedNeRFModelConfig(NerfactoModelConfig):
    """SparseGuidedNeRF Model Configuration."""

    _target: Type = field(default_factory=lambda: SparseGuidedNeRFModel)
    """target class to instantiate"""
    bbox_path: Optional[Path] = None
    """Path to the bounding box"""
    start_inpainting_strength: float = 0.8
    """Starting inpainting strength"""
    strength_step: float = 0.2
    """Inpainting strength step"""
    inpainting_start_iter: float = 1000
    """Inpainting start iteration"""
    inpainting_repeat_iter: float = 1000
    """Inpainting repeat iteration"""
    num_images_in_group: int = 32
    """Number of images in group"""
    proposal_warmup: int = 500
    """Scales n from 1 to proposal_update_every over this many steps"""


class SparseGuidedNeRFModel(NerfactoModel):
    """SparseGuidedNeRF Model."""

    config: SparseGuidedNeRFModelConfig

    def __init__(self, 
                 config: ModelConfig, 
                 scene_box: SceneBox, 
                 num_train_data: int, 
                 dataparser_config: Dict[str, Any],
                 **kwargs) -> None:
        super().__init__(config, scene_box, num_train_data, **kwargs)

        self.dataparser_config = dataparser_config
        self.reload_data_fn = kwargs['reload_data_fn']
        self.inpainter: Inpainter = kwargs['inpainter']

    def populate_modules(self):
        super().populate_modules()

        # Loading bbox data
        CONSOLE.print("Loading bounding boxes from: ")

        bbox_path = str(self.config.bbox_path)

        assert bbox_path is not None, "Please specify --pipeline.model.bbox-path"

        CONSOLE.print(f"- {bbox_path}")

        with open(bbox_path, 'r') as f:
            bboxes_file = json.load(f)

        self.bboxes: List[BoundingBox] = []
        for bbox in bboxes_file['bboxes']:
            data = bbox['data']
            bbox_data = BBoxData(bbox['idx'], 
                                 dimensions=data['dimensions'], 
                                 euler=data['euler'], 
                                 quaternion=data['quaternion'], 
                                 position=data['position'],
                                 is_reconfigured=data['is_reconfigured'],
                                 offset_position=data['offset_position'],
                                 offset_euler=data['offset_euler'])

            self.bboxes.append(BoundingBox(bbox_data))
        
        CONSOLE.print("Loaded bounding boxes succesfully!")
        CONSOLE.print()

    def get_outputs(self, ray_bundle: RayBundle):
        # apply the camera optimizer pose tweaks
        if self.training:
            with torch.no_grad():
                self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        masked_densities = None
        with torch.no_grad():
            masks = None
            for bbox in self.bboxes:
                mask = bbox.are_points_inside(ray_samples.frustums.get_positions(), offset_size=(0.01, 0.01, 0.0))

                if masks is None:
                    masks = mask
                else:
                    masks = torch.logical_or(masks, mask)

            masked_densities = torch.where(masks, 0, field_outputs[FieldHeadNames.DENSITY])

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
            "densities": field_outputs[FieldHeadNames.DENSITY],
            "masked_densities": masked_densities
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        return outputs
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        if self.step < 1000:
            loss_dict["density_loss"] = 0.1 * self.rgb_loss(outputs["densities"], outputs["masked_densities"])
            
        loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb)
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
        return loss_dict

    def inpaint_images(self, step):

        if step >= self.config.inpainting_start_iter and step % self.config.inpainting_repeat_iter == 0:
            # Create paths for generating dataset
            cache_path = self.config.bbox_path.parent / "cache"

            # Get cameras for rendering
            dataparser_config = self.dataparser_config
            dataparser_config.eval_mode = "filename"
            cameras, image_names = get_cameras(dataparser_config)

            # Extract dedicated cameras
            indices = []
            for idx, image_name in enumerate(image_names):
                if "dedicated" in image_name:
                    indices.append(idx)
            
            image_names = np.array(image_names)[indices]

            # Assign frames with names
            selected_frames = []
            num_img = self.config.num_images_in_group
            num_image_groups = len(image_names) // num_img
            for num_group in range(num_image_groups):
                frames = []
                for image_name, idx in zip(image_names[num_group * num_img:(num_group + 1) * num_img], indices[num_group * num_img:(num_group + 1) * num_img]):
                    frames.append((image_name, idx))
                selected_frames.append(frames)

            # Render images
            strength = round(self.config.start_inpainting_strength - (self.config.strength_step * ((step - self.config.inpainting_start_iter) // self.config.inpainting_repeat_iter)), 2)

            if strength > 0:
                self.eval()
                self.inpainter.inpaint_images_in_grids(model=self,
                                                       selected_frames=selected_frames,
                                                       cameras=cameras,
                                                       output_path=cache_path,
                                                       log_name=f"iteration_{step}_s_{strength}",
                                                       rendering_output_name='rgb',
                                                       strength=strength,
                                                       refine_masks=False)
                self.train()
            
                # Reload dataset
                self.reload_data_fn()

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                self.step = step
                train_frac = np.clip(step / N, 0, 1)
                self.step = step

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)


            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=100,
                    func=self.inpaint_images,
                )
            )

        return callbacks
    
    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle, prserve_device: bool = True) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        input_device = camera_ray_bundle.directions.device
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            # move the chunk inputs to the model device
            ray_bundle = ray_bundle.to(self.device)
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                if not isinstance(output, torch.Tensor):
                    # TODO: handle lists of tensors as well
                    continue
                # move the chunk outputs from the model device back to the device of the inputs.
                if prserve_device:
                    outputs_lists[output_name].append(output.to(input_device))
                else:
                    outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs
    
    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Assumes a ray-based model.

        Args:
            camera: generates raybundle
        """

        # TODO Not really stack rendering for now but there was a preoblem with RegionAwareSampling indices and 
        # objects were not removed properly in the final renderings. Speedup from stacking is unnoticable

        outputs = []
        for cam in camera:
            ray_bundle = cam.generate_rays(camera_indices=0, keep_shape=True, obb_box=obb_box)
            output = self.get_outputs_for_camera_ray_bundle(ray_bundle, prserve_device=False)
            outputs.append(output)

        if len(outputs) == 1:
            return outputs[0]
        
        for key in outputs[0].keys():
            outputs[0][key] = torch.stack([output[key] for output in outputs], dim=2)

        return outputs[0]