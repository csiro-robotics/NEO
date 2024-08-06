"""
Replace NeRF for replacing objects from two NeRF models. Used by Neo pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Literal, Tuple, Type

import numpy as np
import json
import torch
from torch.nn import Parameter

from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    L1Loss,
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig

from neo.utils.bbox import BoundingBox
from neo.utils.bbox_utils import BBoxData
from neo.utils.ray_samplers import DualProposalNetworkSampler, ReconfigurationalProposalNetworkSampler


@dataclass
class ReplaceNeRFModelConfig(NerfactoModelConfig):
    """ReplaceNeRF Model Config"""

    _target: Type = field(default_factory=lambda: ReplaceNeRFModel)

    field_path: Path = ""
    """Path to the main NeRF"""
    replace_field_path: Path = ""
    """Path to the NeRF with reconstructed region"""
    bbox_path: Path = ""
    """Path to the bounding box"""


class ReplaceNeRFModel(NerfactoModel):
    """ReplaceNeRF model

    Args:
        config: ReplaceNeRF configuration to instantiate model
    """

    config: ReplaceNeRFModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        appearance_embedding_dim = self.config.appearance_embed_dim if self.config.use_appearance_embedding else 0

        # Fields
        self.field = NerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            average_init_density=self.config.average_init_density,
            implementation=self.config.implementation,
        )

        self.field_copy = NerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            average_init_density=self.config.average_init_density,
            implementation=self.config.implementation,
        )

        self.replace_field = NerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            average_init_density=self.config.average_init_density, # 1.0
            implementation=self.config.implementation,
        )

        # Load pretrained fields
        CONSOLE.print("Loading Fields from: ")

        field_path = self.config.field_path
        replace_field_path = self.config.replace_field_path

        CONSOLE.print(f"- {field_path}")
        CONSOLE.print(f"- {replace_field_path}")

        field_state = torch.load(field_path, map_location="cpu")
        field_state = {
            (key[len("_model.field.") :] if key.startswith("_model.field.") else key): value for key, value in field_state["pipeline"].items() if key.startswith("_model.field.")
        }

        field_copy_state = torch.load(field_path, map_location="cpu")
        field_copy_state = {
            (key[len("_model.field.") :] if key.startswith("_model.field.") else key): value for key, value in field_copy_state["pipeline"].items() if key.startswith("_model.field.")
        }

        replace_field_state = torch.load(replace_field_path, map_location="cpu")
        replace_field_state = {
            (key[len("_model.field.") :] if key.startswith("_model.field.") else key): value for key, value in replace_field_state["pipeline"].items() if key.startswith("_model.field.")
        }

        self.field.load_state_dict(field_state)
        self.field_copy.load_state_dict(field_copy_state)
        self.field_copy.eval()
        self.replace_field.load_state_dict(replace_field_state)
        self.replace_field.eval()

        CONSOLE.print("Fields loaded succesfully!")
        CONSOLE.print()

        # Loading bbox data
        CONSOLE.print("Loading bounding box from: ")

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
        
        CONSOLE.print("Fields bounding box succesfully!")
        CONSOLE.print()
        
        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        ######################################################################## Proposal Networks ########################################################################

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                average_init_density=self.config.average_init_density,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    average_init_density=self.config.average_init_density,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        self.density_fns_copy = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s) copy
        self.proposal_networks_copy = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                average_init_density=self.config.average_init_density,
                implementation=self.config.implementation,
            )
            self.proposal_networks_copy.append(network)
            self.density_fns_copy.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    average_init_density=self.config.average_init_density,
                    implementation=self.config.implementation,
                )
                self.proposal_networks_copy.append(network)
            self.density_fns_copy.extend([network.density_fn for network in self.proposal_networks_copy])

        self.replace_density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s) replace copy
        self.replace_proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                average_init_density=self.config.average_init_density, # 1.0
                implementation=self.config.implementation,
            )
            self.replace_proposal_networks.append(network)
            self.replace_density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    average_init_density=self.config.average_init_density, # 1.0
                    implementation=self.config.implementation,
                )
                self.replace_proposal_networks.append(network)
            self.replace_density_fns.extend([network.density_fn for network in self.replace_proposal_networks])

        ###################################################################################################################################################################

        CONSOLE.print("Load proposal network from:")
        CONSOLE.print(f"- {field_path}")
        CONSOLE.print(f"- {replace_field_path}")

        proposal_networks_state = torch.load(field_path, map_location="cpu")
        proposal_networks_state = {
            (key[len("_model.proposal_networks.") :] if key.startswith("_model.proposal_networks.") else key): value for key, value in proposal_networks_state["pipeline"].items() if key.startswith("_model.proposal_networks.")
        }
        
        self.proposal_networks_copy.load_state_dict(proposal_networks_state)
        self.proposal_networks_copy.eval()

        replace_proposal_networks_state = torch.load(replace_field_path, map_location="cpu")
        replace_proposal_networks_state = {
            (key[len("_model.proposal_networks.") :] if key.startswith("_model.proposal_networks.") else key): value for key, value in replace_proposal_networks_state["pipeline"].items() if key.startswith("_model.proposal_networks.")
        }
        
        self.replace_proposal_networks.load_state_dict(replace_proposal_networks_state)
        self.replace_proposal_networks.eval()

        CONSOLE.print("Proposal network loaded succesfully!")
        CONSOLE.print()

        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        self.dual_proposal_sampler = DualProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        self.reconfigurational_proposal_sampler = ReconfigurationalProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="median")
        self.renderer_expected_depth = DepthRenderer(method="expected")
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()
        self.depth_loss = MSELoss()
        self.color_loss = L1Loss()
        self.density_loss = L1Loss()
        self.step = 0
        # metrics
        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0
        self.frozen = False

    def get_outputs(self, ray_bundle: RayBundle):
        # apply the camera optimizer pose tweaks
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        with torch.no_grad():
            offset = (0.02, 0.02, 0.2)
            dual_samples_copy, _, _ = self.dual_proposal_sampler(ray_bundle, density_fns=self.density_fns_copy, replace_density_fns=self.replace_density_fns, bboxes=self.bboxes, offset_size=offset)

            field_copy_outputs = self.field_copy.forward(dual_samples_copy, compute_normals=self.config.predict_normals)
            replace_field_outputs = self.replace_field.forward(dual_samples_copy, compute_normals=self.config.predict_normals)

            masks = None
            for bbox in self.bboxes:
                mask = bbox.are_points_inside(dual_samples_copy.frustums.get_positions(), offset_size=offset)

                if masks is None:
                    masks = mask
                else:
                    masks = torch.logical_or(masks, mask)

            gt_densities = torch.where(masks, replace_field_outputs[FieldHeadNames.DENSITY], field_copy_outputs[FieldHeadNames.DENSITY])
            gt_rgbs = torch.where(masks, replace_field_outputs[FieldHeadNames.RGB], field_copy_outputs[FieldHeadNames.RGB])

            reconfigurational_samples, _, _ = self.reconfigurational_proposal_sampler(ray_bundle, density_fns=self.density_fns_copy, replace_density_fns=self.density_fns_copy, bboxes=self.bboxes, offset_size=(0.0, 0.0, 0.0))
            original_samples = deepcopy(reconfigurational_samples)

            rec_mask = None
            for bbox in self.bboxes:
                if bbox.is_reconfigured:
                    mask = bbox.are_points_inside_reconfigured(original_samples.frustums.get_positions(), offset_size=(0.0, 0.0, 0.0))
                else:
                    mask = torch.zeros_like(gt_densities, dtype=torch.bool)

                if rec_mask == None:
                    rec_mask = torch.any(mask, dim=1).squeeze(-1)
                else:
                    rec_mask = torch.logical_or(torch.any(mask, dim=1).squeeze(-1), rec_mask)
            
                reconfigurational_samples.frustums.origins = original_samples.frustums.origins - bbox.offset_position
                reconfigurational_samples.frustums.origins = ((reconfigurational_samples.frustums.origins - bbox.translation) @ bbox.offset_orientation.T) + bbox.translation
                reconfigurational_samples.frustums.directions = original_samples.frustums.directions @ bbox.offset_orientation.T
                shifted_outputs = self.field_copy.forward(reconfigurational_samples, compute_normals=self.config.predict_normals)
                
                gt_densities = torch.where(mask, shifted_outputs[FieldHeadNames.DENSITY], gt_densities)
                gt_rgbs = torch.where(mask, shifted_outputs[FieldHeadNames.RGB], gt_rgbs)

            gt_weights = dual_samples_copy.get_weights(gt_densities)
            gt_weights[rec_mask] = reconfigurational_samples.get_weights(gt_densities)[rec_mask]
            gt_rgb = self.renderer_rgb(rgb=gt_rgbs, weights=gt_weights)
            gt_depth = self.renderer_depth(weights=gt_weights, ray_samples=dual_samples_copy)
            gt_depth[rec_mask] = self.renderer_depth(weights=gt_weights, ray_samples=reconfigurational_samples)[rec_mask]

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
            "gt_rgb": gt_rgb,
            "accumulation": accumulation,
            "depth": depth,
            "gt_depth": gt_depth,
            "expected_depth": expected_depth,
            "gt_rgbs": gt_rgbs,
            "gt_densities": gt_densities,
            "rgbs": field_outputs[FieldHeadNames.RGB],
            "densities": field_outputs[FieldHeadNames.DENSITY]
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
    
    def get_metrics_dict(self, outputs):
        metrics_dict = {}
        gt_rgb = outputs["gt_rgb"]
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])

        # self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_loss_dict(self, outputs, step, metrics_dict=None):
        loss_dict = {}
        
        image = outputs["gt_rgb"]

        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )
        
        if step < 500:
            print(f"Sigma loss: {step}")
            
            if not self.frozen:
                self.frozen = True
                for name, param in self.field.named_parameters():
                    if 'mlp' in name:
                        param.requires_grad = False
                        print(f'Frozen {name}')

            loss_dict["rgbs_loss"] = self.color_loss(outputs["gt_rgbs"], outputs["rgbs"])
            loss_dict["densities_loss"] = self.density_loss(outputs["gt_densities"], outputs["densities"])

        else:
            print(f"Fotometric: {step}")
            
            if self.frozen:
                self.frozen = False
                for name, param in self.field.named_parameters():
                    if 'mlp' in name:
                        param.requires_grad = True
                        print(f'Unfrozen {name}')
            
            loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb)
            loss_dict["depth"] = self.depth_loss(outputs["gt_depth"], outputs["depth"])
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
                # Add loss from camera optimizer
                # self.camera_optimizer.get_loss_dict(loss_dict)
        return loss_dict
