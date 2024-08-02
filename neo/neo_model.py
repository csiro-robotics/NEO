"""
Neo Model.
"""
import json
import os
import numpy as np
import torch

from dataclasses import dataclass, field
from copy import deepcopy
from collections import defaultdict
from typing import Type, Optional, Tuple, List, Union, Dict
from scipy.spatial.transform import Rotation
from pathlib import Path

from neo.utils.bbox import BoundingBox

from neo.utils.bbox_utils import (
    BBoxData,
    BoundingBoxToolbox,
    ViserBoundingBox
)

from neo.utils.ray_samplers import (
    RegionAwareUniformLinDispPiecewiseSampler,
    RegionAwarePDFSampler,
    RegionAwareProposalNetworkSampler,
    ReconfigurationalProposalNetworkSampler
)

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.model_components.ray_samplers import UniformSampler
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.field_components.field_heads import FieldHeadNames

from nerfstudio.model_components.losses import (
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)


@dataclass
class NeoModelConfig(NerfactoModelConfig):
    """Neo Model Configuration."""

    _target: Type = field(default_factory=lambda: NeoModel)


class NeoModel(NerfactoModel):
    """Neo Model."""

    config: NeoModelConfig

    def populate_modules(self):
        super().populate_modules()

        # Dataparser transformations
        self.apply_changes_hook = self.kwargs['apply_changes_hook']

        # Load bbox data
        self.bbox = None
        self.bboxes = {"bboxes": []}
        self.bbox_vis: List[ViserBoundingBox] = []
        self.rec_bbox_vis: Dict[int, ViserBoundingBox] = {}
        self.bbox_dir = self.kwargs['bbox_dir'] / 'bbox.json'
        if not os.path.exists(self.bbox_dir):
            print(f"File bbox.json does not exist. Creating a file.")

            with open(self.bbox_dir, 'w') as f:
                json.dump(self.bboxes, f, indent=2)

        else:
            print(f"File bbox.json already exists. Loading the file.")

            with open(self.bbox_dir, 'r') as f:
                self.bboxes = json.load(f)

        if len(self.bboxes['bboxes']) > 0:
            for i, bbox in enumerate(self.bboxes['bboxes']):
                data = bbox['data']
                bbox_data = BBoxData(bbox['idx'], 
                                     dimensions=data['dimensions'], 
                                     euler=data['euler'], 
                                     quaternion=data['quaternion'], 
                                     position=data['position'],
                                     is_reconfigured=data['is_reconfigured'],
                                     offset_position=data['offset_position'],
                                     offset_euler=data['offset_euler'])

                self.bbox = ViserBoundingBox(data=bbox_data.viser_format(), cb_hook=self.display_wireframe)
                self.bbox_vis.append(self.bbox)

                # Register loaded bounding boxes
                setattr(self, f"temp_bbox_{i}", self.bbox)

                if data['is_reconfigured']:
                    rec_bbox = ViserBoundingBox(data=bbox_data.viser_format(), name_prefix="RecBox", wireframe=True, color=(0, 255, 0), offset=True)
                    self.rec_bbox_vis[bbox['idx']] = rec_bbox

                    # Register loaded bounding boxes
                    setattr(self, f"temp_rec_bbox_{i}", rec_bbox)
                
            self.bbox_tool = BoundingBoxToolbox("Boxes", 
                                                self.bbox_vis[-1].data, 
                                                move_hook=self.move_box, 
                                                offset_box_hook=self.offset_box_hook,
                                                resize_hook=self.resize_box, 
                                                reconfigure_hook=self.reconfigure_hook, 
                                                bbox_selector_hook=self.bbox_selector_hook, 
                                                add_box_hook=self.add_box_hook, 
                                                apply_changes_hook=self.apply_changes_hook)

        else:
            bbox_data = BBoxData(0, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            self.bbox_tool = BoundingBoxToolbox("Boxes", 
                                                bbox_data, 
                                                move_hook=self.move_box, 
                                                offset_box_hook=self.offset_box_hook,
                                                resize_hook=self.resize_box, 
                                                reconfigure_hook=self.reconfigure_hook, 
                                                bbox_selector_hook=self.bbox_selector_hook, 
                                                add_box_hook=self.add_box_hook, 
                                                apply_changes_hook=self.apply_changes_hook)

        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Change default samplers to Region-Aware samplers
        initial_sampler = RegionAwareUniformLinDispPiecewiseSampler(single_jitter=self.config.use_single_jitter)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)
        pdf_sampler = RegionAwarePDFSampler(include_original=False, single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = RegionAwareProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
            pdf_sampler=pdf_sampler,
        )

        self.reconfigurational_proposal_sampler = ReconfigurationalProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

    def add_box_hook(self, element):
        
        idx = len(self.bbox_vis)
        bbox_data = BBoxData(idx, [0.1, 0.1, 0.1], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

        self.bbox = ViserBoundingBox(data=bbox_data.viser_format(), cb_hook=self.display_wireframe, offset=True)
        self.bbox.install(self.bbox_tool.viser_server)

        self.bbox_vis.append(self.bbox)
        self.bbox_tool.bbox_selector_handle.value = idx

    def move_box(self, element):
        self.bbox.update(self.bbox_tool.get_bbox_data().viser_format())

    def offset_box_hook(self, element):

        # Initialize bbox
        if getattr(self, "bbox", False):
            self.bbox.install(self.bbox_tool.viser_server)
        else:
            self.bbox = None

        idx = self.bbox.data.idx
        bbox_data = self.bbox_tool.get_bbox_data().viser_format()
        self.bbox.update(bbox_data)

        if bbox_data.is_reconfigured:
            self.rec_bbox_vis[idx].update(bbox_data)

    def resize_box(self, element):
        self.bbox.box_handle.remove()
        self.bbox = ViserBoundingBox(data=self.bbox_tool.get_bbox_data().viser_format(), cb_hook=self.display_wireframe, wireframe=self.bbox.wireframe)
        self.bbox.install(self.bbox_tool.viser_server)
        self.bbox_vis[self.bbox_tool.bbox_selector_handle.value] = self.bbox

    def reconfigure_hook(self, element):
        idx = self.bbox_tool.bbox_selector_handle.value

        bbox_data = self.bbox_vis[idx].data
        if self.bbox_tool.reconfigure_handle.value:
            if not bbox_data.is_reconfigured:
                bbox_data.is_reconfigured = True
                bbox = ViserBoundingBox(data=bbox_data, name_prefix="RecBox", wireframe=True, color=(0, 255, 0))
                bbox.install(self.bbox_tool.viser_server)
                self.rec_bbox_vis[idx] = bbox

            self.bbox_tool.offset_position_handle.visible = True
            self.bbox_tool.offset_rotation_handle.visible = True

        else:
            if bbox_data.is_reconfigured:
                bbox_data.is_reconfigured = False
                self.rec_bbox_vis[idx].box_handle.remove()
                
            self.bbox_tool.offset_position_handle.visible = False
            self.bbox_tool.offset_rotation_handle.visible = False

    def display_wireframe(self, element):
        wireframe = not bool(self.bbox.wireframe)

        self.bbox.box_handle.remove()
        self.bbox = ViserBoundingBox(data=self.bbox.data, cb_hook=self.display_wireframe, wireframe=wireframe)
        self.bbox.install(self.bbox_tool.viser_server)
        self.bbox_vis[self.bbox_tool.bbox_selector_handle.value] = self.bbox

    def bbox_selector_hook(self, element):
        self.bbox = self.bbox_vis[self.bbox_tool.bbox_selector_handle.value]
        self.bbox_tool.update(self.bbox.data)

    def get_outputs(self, ray_bundle: RayBundle, ray_masks: Optional[torch.Tensor] = None):
        # Apply the camera optimizer pose tweaks
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)

        ray_samples: RaySamples

        if ray_masks == None:
            ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

        else:
            ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns, ray_masks=ray_masks)

        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

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
        }

        if not self.training and ray_masks is not None and len(self.bbox_vis) > 0:
            outputs["rgb_rm"] = rgb
            outputs["depth_rm"] = depth

            with torch.no_grad():
                bboxes: List[BoundingBox] = []
                for viser_bbox in self.bbox_vis:
                    bboxes.append(BoundingBox(viser_bbox.data.nerfstudio_format()))

                reconfigurational_samples, _, _ = self.reconfigurational_proposal_sampler(ray_bundle, density_fns=self.density_fns, replace_density_fns=self.density_fns, bboxes=bboxes, offset_size=(0.0, 0.0, 0.0))
                original_samples = deepcopy(reconfigurational_samples)

                densities = field_outputs[FieldHeadNames.DENSITY]
                rgbs = field_outputs[FieldHeadNames.RGB]

                rec_mask = None
                for bbox in bboxes:
                    if bbox.is_reconfigured:
                        mask = bbox.are_points_inside_reconfigured(original_samples.frustums.get_positions(), offset_size=(0.0, 0.0, 0.0))
                    else:
                        mask = torch.zeros_like(densities, dtype=torch.bool)

                    if rec_mask == None:
                        rec_mask = torch.any(mask, dim=1).squeeze(-1)
                    else:
                        rec_mask = torch.logical_or(torch.any(mask, dim=1).squeeze(-1), rec_mask)

                    reconfigurational_samples.frustums.origins = reconfigurational_samples.frustums.origins - bbox.offset_position
                    reconfigurational_samples.frustums.origins = ((reconfigurational_samples.frustums.origins - bbox.translation) @ bbox.offset_orientation.T) + bbox.translation
                    reconfigurational_samples.frustums.directions = reconfigurational_samples.frustums.directions @ bbox.offset_orientation.T
                    shifted_outputs = self.field.forward(reconfigurational_samples, compute_normals=self.config.predict_normals)

                    densities = torch.where(mask, shifted_outputs[FieldHeadNames.DENSITY], densities)
                    rgbs = torch.where(mask, shifted_outputs[FieldHeadNames.RGB], rgbs)

                weights = ray_samples.get_weights(densities)
                weights[rec_mask] = reconfigurational_samples.get_weights(densities)[rec_mask]
                rgb = self.renderer_rgb(rgb=rgbs, weights=weights)
                depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
                depth[rec_mask] = self.renderer_depth(weights=weights, ray_samples=reconfigurational_samples)[rec_mask]

                outputs["rgb"] = rgb
                outputs["depth"] = depth

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

        # Update json file
        if self.bbox is not None:

            self.bboxes["bboxes"] = []
            for bbox in self.bbox_vis:
                self.bboxes["bboxes"].append(bbox.data.get_json_data())

            with open(self.bbox_dir, 'w') as f:
                    json.dump(self.bboxes, f, indent=2)

        # Generate ray masks
        new_ray_masks = torch.zeros_like(ray_samples.frustums.starts)

        if self.bbox is not None:
            bound_dists = []
            for i, bbox in enumerate(self.bbox_vis):
                boundingbox = BoundingBox(bbox.data.nerfstudio_format())
                bound_dist = boundingbox.calculate_distances(ray_bundle.origins.reshape(-1, 3), ray_bundle.directions.reshape(-1, 3))
                bound_dists.append(bound_dist.unsqueeze(1))

            data = torch.concat(bound_dists, dim=1)
            data = torch.where(data == 0, 3, data)
            twos = torch.ones_like(data) * 2
            threes = torch.ones_like(data) * 3

            temp_1 = torch.concat([data[:, :, 0:1], twos[:, :, 0:1]], dim=2)
            temp_2 = torch.concat([data[:, :, 1:2], threes[:, :, 1:2]], dim=2)

            values = None
            for i in range(data.shape[1]):
                if values is None:
                    values = torch.concat([temp_1[:, i:i+1, :], temp_2[:, i:i+1, :]], dim=1)
                else:
                    values = torch.concat([values, temp_1[:, i:i+1, :], temp_2[:, i:i+1, :]], dim=1)
            
            indices = values[:, :, 0].sort(stable=True)[1]
            values = values[torch.arange(values.size(0)).unsqueeze(1), indices]
            values = values.reshape(values.shape[0], values.shape[1] * values.shape[2])

            # Indices of columns to keep
            indices_to_keep = [i for i in range(values.shape[1]) if ((i + 1) % 4) - 2 != 0]
            values = values[:, indices_to_keep]
            values = torch.where(values == 3, 0, values)
            values = torch.where(values == 2, 0.1, values)

            new_ray_masks[:, 1:values.shape[1] + 1, :] = values[..., None]

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        if ray_masks == None and not self.training:
            return outputs, new_ray_masks
        
        else:
            return outputs
        
    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
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
            outputs, ray_masks = self.forward(ray_bundle=ray_bundle)
            outputs_bounded = self.forward(ray_bundle=ray_bundle, ray_masks=ray_masks)

            for output_name, output in outputs.items():  # type: ignore
                if not isinstance(output, torch.Tensor):
                    # TODO: handle lists of tensors as well
                    continue
                # move the chunk outputs from the model device back to the device of the inputs.
                outputs_lists[output_name].append(output.to(input_device))

            for output_name, output in outputs_bounded.items():  # type: ignore
                if not isinstance(output, torch.Tensor):
                    # TODO: handle lists of tensors as well
                    continue
                # move the chunk outputs from the model device back to the device of the inputs.
                outputs_lists[f"{output_name}_preview"].append(output.to(input_device))

        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs
    
    def forward(self, ray_bundle: Union[RayBundle, Cameras], ray_masks: Optional[torch.Tensor] = None) -> Dict[str, Union[torch.Tensor, List]]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        if ray_masks == None:
            return self.get_outputs(ray_bundle)
        
        else:
            return self.get_outputs(ray_bundle, ray_masks)