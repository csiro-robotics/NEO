"""
Sparse-Guided NeRF Model

This model is utilized by Neo for editing purposes. It learns region reconstruction from sparse inpainted dataset.
"""
import torch
import json
import yaml

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, List

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


@dataclass
class SparseGuidedNeRFModelConfig(NerfactoModelConfig):
    """SparseGuidedNeRF Model Configuration."""

    _target: Type = field(default_factory=lambda: SparseGuidedNeRFModel)

    bbox_path: Path = None
    """Path to the bounding box"""


class SparseGuidedNeRFModel(NerfactoModel):
    """SparseGuidedNeRF Model."""

    config: SparseGuidedNeRFModelConfig

    def populate_modules(self):
        super().populate_modules()

        # Loading bbox data
        CONSOLE.print("Loading bounding boxes from: ")

        bbox_path = self.config.bbox_path

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