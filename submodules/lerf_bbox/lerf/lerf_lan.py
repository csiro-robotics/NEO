from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, Optional
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation
from pathlib import Path

import numpy as np
import open_clip
import torch
import json
import os
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import OrientedBox, SceneBox
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.ray_samplers import PDFSampler
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.model_components.renderers import DepthRenderer
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils.colormaps import ColormapOptions, apply_colormap
from nerfstudio.viewer.viewer_elements import *
from torch.nn import Parameter

from lerf.encoders.image_encoder import BaseImageEncoder
from lerf.lerf_field import LERFField
from lerf.lerf_fieldheadnames import LERFFieldHeadNames
from lerf.lerf_renderers import CLIPRenderer, MeanRenderer
import copy

MAX_DENSITY=1000

@dataclass
class LERFLanModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: LERFLanModel)
    clip_loss_weight: float = 0.1
    n_scales: int = 30
    max_scale: float = 1.5
    """maximum scale used to compute relevancy with"""
    num_lerf_samples: int = 24
    hashgrid_layers: Tuple[int, ...] = (12, 12)
    hashgrid_resolutions: Tuple[Tuple[int, ...], ...] = ((16, 128), (128, 512))
    hashgrid_sizes: Tuple[int, ...] = (19, 19),
    output_folder: Optional[str] = ''


class LERFLanModel(NerfactoModel):
    config: LERFLanModelConfig

    def populate_modules(self):
        super().populate_modules()

        # if self.config.disable_scene_contraction:
        #     scene_contraction = None
        # else:
        #     scene_contraction = SceneContraction(order=float("inf"))

        # appearance_embedding_dim = self.config.appearance_embed_dim if self.config.use_appearance_embedding else 0

        # # Fields
        # self.field = NerfactoField(
        #     self.scene_box.aabb,
        #     hidden_dim=self.config.hidden_dim,
        #     num_levels=self.config.num_levels,
        #     max_res=self.config.max_res,
        #     base_res=self.config.base_res,
        #     features_per_level=self.config.features_per_level,
        #     log2_hashmap_size=self.config.log2_hashmap_size,
        #     hidden_dim_color=self.config.hidden_dim_color,
        #     hidden_dim_transient=self.config.hidden_dim_transient,
        #     spatial_distortion=scene_contraction,
        #     num_images=self.num_train_data - 31,
        #     use_pred_normals=self.config.predict_normals,
        #     use_average_appearance_embedding=self.config.use_average_appearance_embedding,
        #     appearance_embedding_dim=appearance_embedding_dim,
        #     average_init_density=self.config.average_init_density,
        #     implementation=self.config.implementation,
        # )

        if len(self.kwargs.get('output_folder')) > 0:
            self.config.output_folder = self.kwargs.get("output_folder")

        self.renderer_clip = CLIPRenderer()
        self.renderer_mean = MeanRenderer()

        self.image_encoder: BaseImageEncoder = self.kwargs["image_encoder"]
        self.lerf_field = LERFField(
            self.config.hashgrid_layers,
            self.config.hashgrid_sizes,
            self.config.hashgrid_resolutions,
            clip_n_dims=self.image_encoder.embedding_dim,
        )

    def get_max_across(self, ray_samples, weights, hashgrid_field, scales_shape, preset_scales=None):
        # TODO smoothen this out
        if preset_scales is not None:
            assert len(preset_scales) == len(self.image_encoder.positives)
            scales_list = torch.tensor(preset_scales)
        else:
            scales_list = torch.linspace(0.0, self.config.max_scale, self.config.n_scales)

        # probably not a good idea bc it's prob going to be a lot of memory
        n_phrases = len(self.image_encoder.positives)
        n_phrases_maxs = [None for _ in range(n_phrases)]
        n_phrases_sims = [None for _ in range(n_phrases)]
        for i, scale in enumerate(scales_list):
            scale = scale.item()
            with torch.no_grad():
                clip_output = self.lerf_field.get_output_from_hashgrid(
                    ray_samples,
                    hashgrid_field,
                    torch.full(scales_shape, scale, device=weights.device, dtype=hashgrid_field.dtype),
                )
            clip_output = self.renderer_clip(embeds=clip_output, weights=weights.detach())

            for j in range(n_phrases):
                if preset_scales is None or j == i:
                    probs = self.image_encoder.get_relevancy(clip_output, j)
                    pos_prob = probs[..., 0:1]
                    if n_phrases_maxs[j] is None or pos_prob.max() > n_phrases_sims[j].max():
                        n_phrases_maxs[j] = scale
                        n_phrases_sims[j] = pos_prob
        return torch.stack(n_phrases_sims), torch.Tensor(n_phrases_maxs)

    def get_relevancy_across(self, ray_samples, weights, hashgrid_field, scales_shape, preset_scales=None):
        # TODO smoothen this out
        if preset_scales is not None:
            assert len(preset_scales) == len(self.image_encoder.positives)
            scales_list = torch.tensor(preset_scales)
        else:
            scales_list = torch.linspace(0.0, self.config.max_scale, self.config.n_scales)

        # probably not a good idea bc it's prob going to be a lot of memory
        n_phrases = len(self.image_encoder.positives)
        n_phrases_sims = [[] for _ in range(n_phrases)]
        for i, scale in enumerate(scales_list):
            scale = scale.item()
            with torch.no_grad():
                clip_outputs = self.lerf_field.get_output_from_hashgrid(
                    ray_samples,
                    hashgrid_field,
                    torch.full(scales_shape, scale, device=weights.device, dtype=hashgrid_field.dtype),
                )

            for j in range(n_phrases):
                for k in range(clip_outputs.shape[1]):
                    clip_output = clip_outputs[:, k, :]
                    if preset_scales is None or j == i:
                        probs = self.image_encoder.get_relevancy(clip_output, j)
                        pos_prob = probs[..., 0:1]

                        n_phrases_sims[j].append(pos_prob)

        for j in range(n_phrases):
            n_phrases_sims[j] = torch.stack(n_phrases_sims[j]).permute(2, 1, 0)
                        
        return torch.concat(n_phrases_sims, dim=0)

    def get_outputs(self, ray_bundle: RayBundle, is_ortho: Optional[bool] = False):
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        ray_samples_list.append(ray_samples)

        nerfacto_field_outputs, outputs, weights = self._get_outputs_nerfacto(ray_samples)
        lerf_weights, best_ids = torch.topk(weights, self.config.num_lerf_samples, dim=-2, sorted=False)
        lerf_densities = torch.gather(outputs["densities"], -2, best_ids.expand(lerf_weights.shape))

        def gather_fn(tens):
            return torch.gather(tens, -2, best_ids.expand(*best_ids.shape[:-1], tens.shape[-1]))

        dataclass_fn = lambda dc: dc._apply_fn_to_fields(gather_fn, dataclass_fn)
        lerf_samples: RaySamples = ray_samples._apply_fn_to_fields(gather_fn, dataclass_fn)

        if self.training:
            with torch.no_grad():
                clip_scales = ray_bundle.metadata["clip_scales"]
                clip_scales = clip_scales[..., None]
                dist = (lerf_samples.frustums.get_positions() - ray_bundle.origins[:, None, :]).norm(
                    dim=-1, keepdim=True
                )
            clip_scales = clip_scales * ray_bundle.metadata["height"] * (dist / ray_bundle.metadata["fy"])
        else:
            clip_scales = torch.ones_like(lerf_samples.spacing_starts, device=self.device)

        override_scales = (
            None if "override_scales" not in ray_bundle.metadata else ray_bundle.metadata["override_scales"]
        )
        weights_list.append(weights)
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        lerf_field_outputs = self.lerf_field.get_outputs(lerf_samples, clip_scales)
        outputs["clip"] = self.renderer_clip(
            embeds=lerf_field_outputs[LERFFieldHeadNames.CLIP], weights=lerf_weights.detach()
        )
        outputs["dino"] = self.renderer_mean(
            embeds=lerf_field_outputs[LERFFieldHeadNames.DINO], weights=lerf_weights.detach()
        )

        if not self.training:
            # TODO Remove .cpu() if having GPU that can fit this variable
            outputs["raw_dino"] =  lerf_field_outputs[LERFFieldHeadNames.DINO].cpu()
            
            with torch.no_grad():
                if is_ortho and ray_bundle.metadata["override_scales"] is not None:
                    max_across = self.get_relevancy_across(
                        lerf_samples,
                        lerf_weights,
                        lerf_field_outputs[LERFFieldHeadNames.HASHGRID],
                        clip_scales.shape,
                        preset_scales=ray_bundle.metadata["override_scales"],
                    )

                    max_across = max_across.unsqueeze(-1)
                    positions = lerf_samples.frustums.get_positions().unsqueeze(0)
                    outputs["raw_relevancy"] = torch.cat([max_across, positions], dim=-1)  # N x B x 1
                    print("Outputs Relevancy Ortho:", outputs["raw_relevancy"].shape)
                    outputs["best_scales"] = ray_bundle.metadata["override_scales"]  # N
                    outputs["lerf_weights"] = lerf_weights
                    outputs["lerf_densities"] = lerf_densities

                    # print("Ortho:", max_across.shape)

                else:
                    max_across, best_scales = self.get_max_across(
                        lerf_samples,
                        lerf_weights,
                        lerf_field_outputs[LERFFieldHeadNames.HASHGRID],
                        clip_scales.shape,
                        preset_scales=override_scales,
                    )

                    outputs["raw_relevancy"] = max_across  # N x B x 1
                    print("Outputs Relevancy Regular:", outputs["raw_relevancy"].shape)
                    outputs["best_scales"] = best_scales.to(self.device)  # N

                    # print("Regular:", max_across.shape)

        return outputs

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        LERF overrides this from base_model since we need to compute the max_across relevancy in multiple batches,
        which are not independent since they need to use the same scale
        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        # TODO(justin) implement max across behavior
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)  # dict from name:list of outputs (1 per bundle)
        # DH Guess - Determine best scale to provide the highest relevancy values
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle)
            # take the best scale for each query across each ray bundle
            if i == 0:
                best_scales = outputs["best_scales"]
                best_relevancies = [m.max() for m in outputs["raw_relevancy"]]
            else:
                for phrase_i in range(outputs["best_scales"].shape[0]):
                    m = outputs["raw_relevancy"][phrase_i, ...].max()
                    if m > best_relevancies[phrase_i]:
                        best_scales[phrase_i] = outputs["best_scales"][phrase_i]
                        best_relevancies[phrase_i] = m

        # re-render the max_across outputs using the best scales across all batches
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            ray_bundle.metadata["override_scales"] = best_scales
            outputs = self.forward(ray_bundle=ray_bundle)
            # standard nerfstudio concatting
            for output_name, output in outputs.items():  # type: ignore
                if output_name == "best_scales":
                    continue
                if output_name == "raw_relevancy":
                    for r_id in range(output.shape[0]):
                        outputs_lists[f"relevancy_{r_id}"].append(output[r_id, ...])
                else:
                    outputs_lists[output_name].append(output)
        # DH guess - Get the orthogonal rays used for producing the bbox extraction relevancy samples
        if image_height > 200 and image_width > 200 and not self.training:
            ortho_ray_bundle = self.get_ray_bundle_for_ortho_camera()
            num_ortho_rays = len(ortho_ray_bundle)
            for i in range(0, num_ortho_rays, num_rays_per_chunk):
                start_idx = i
                end_idx = i + num_rays_per_chunk
                ray_bundle = ortho_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                ray_bundle.metadata["override_scales"] = best_scales
                outputs = self.forward(ray_bundle=ray_bundle, is_ortho=True)
                # standard nerfstudio concatting
                for output_name, output in outputs.items():  # type: ignore
                    if output_name == "best_scales":
                        continue
                    if output_name == "raw_relevancy":
                        for r_id in range(output.shape[0]):
                            outputs_lists[f"ortho_relevancy_{r_id}"].append(output[r_id, ...])
                    if output_name == "raw_dino":
                        outputs_lists[f"ortho_dino"].append(outputs["raw_dino"])
                    if output_name == "lerf_weights":
                        outputs_lists["ortho_weights"].append(outputs["lerf_weights"])
                    if output_name == "lerf_densities":
                        outputs_lists["ortho_densities"].append(outputs["lerf_densities"])
                    # else:
                    #     outputs_lists[output_name].append(output)
        else:
            outputs_lists[f"ortho_dino"].append(torch.zeros((480, 640, 1, 384), device=self.device))
            for r_id in range(output.shape[0]):
                outputs_lists[f"ortho_relevancy_{r_id}"].append(torch.zeros((480, 640, 1, 4), device=self.device))

        outputs = {}
        # Reshape content within outputs_lists (compile to image size)
        for output_name, outputs_list in outputs_lists.items():
            if not torch.is_tensor(outputs_list[0]):
                # TODO: handle lists of tensors as well
                continue
                
            if "ortho" in output_name:
                if "weights" in output_name:
                    ortho_weights = torch.cat(outputs_list).view(480, 640, -1)  # type: ignore
                elif "densities" in output_name:
                    ortho_densities = torch.cat(outputs_list).view(480, 640, -1)  # type: ignore
                elif "dino" in output_name:
                    outputs[output_name] = torch.cat(outputs_list).view(480, 640, -1, 384)  # type: ignore
                else:
                    outputs[output_name] = torch.cat(outputs_list).view(480, 640, -1, 4)  # type: ignore
            
            else:
                outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        
        # Visualisation for composited overlay
        for i in range(len(self.image_encoder.positives)):
            p_i = torch.clip(outputs[f"relevancy_{i}"] - 0.5, 0, 1)
            outputs[f"composited_{i}"] = apply_colormap(p_i / (p_i.max() + 1e-6), ColormapOptions("turbo"))
            mask = (outputs["relevancy_0"] < 0.5).squeeze()
            outputs[f"composited_{i}"][mask, :] = outputs["rgb"][mask, :]

        # print("################ OUTPUTS KEYS ####################")
        # print(outputs.keys())
        # # Save relevancy field
        # if image_height > 200 and image_width > 200 and not self.training:
        #     # np.save("relevancy_field_2.npy", outputs["ortho_relevancy_0"].detach().cpu().numpy())

        #     # Reduce dimensionality of relevancy field
        #     relevancy_field = outputs["ortho_relevancy_0"]
        #     dino_field = outputs["ortho_dino"]

        #     reduce_factor = 4

        #     reduction_rows = torch.arange(0, relevancy_field.shape[0] - 1, reduce_factor)
        #     reduced_field = relevancy_field[reduction_rows, :, :, :]
        #     reduced_dino = dino_field[reduction_rows, :, :, :]

        #     reduction_cols = torch.arange(0, relevancy_field.shape[1] - 1, reduce_factor)
        #     reduced_field = reduced_field[:, reduction_cols, :]
        #     reduced_dino = reduced_dino[:, reduction_cols, :]
        #     reduced_field = reduced_field.reshape(-1, 4).detach().cpu().numpy()
        #     reduced_dino = reduced_dino.reshape(-1, 384).detach().cpu().numpy()

        #     # Scale relevancy by weight (TODO density)
        #     below_zero_points = reduced_field[:,-1] < 0.04
        #     left_points = np.logical_or(reduced_field[:,1] < 0, reduced_field[:,1] > 0.65)
        #     back_points = np.logical_or(reduced_field[:,2] < 0, reduced_field[:,2] > 0.65)
        #     in_points = np.logical_not(np.logical_or(np.logical_or(below_zero_points, left_points), back_points))
        #     reduced_field = reduced_field[in_points]
        #     reduced_dino = reduced_dino[in_points]

        #     idxs = reduced_field[:, 0] > 0.6
        #     high_rel_field = reduced_field[idxs]
        #     reduced_dino = reduced_dino[idxs]

        #     if high_rel_field.shape[0] > 0:
        #         # Use DBSCAN to remove outliers
        #         db = DBSCAN(eps=0.05, min_samples=100).fit(high_rel_field[:,1:]) # Perhaps could need to remove first column if that is relevancy
        #         labels = db.labels_
        #         num_groups = len(np.unique(labels)) - 1
        #         print(f"NUM GROUPS: {num_groups}")
        #         bboxes = {"bboxes":[]}
        #         if num_groups > 0:
        #             # Go through and filter field by median relevancy
        #             # Keep only the highest relevancy
        #             dbscan_fields = []
        #             best_field = None
        #             best_dino = None
        #             max_median_relevancy = 0
        #             for i in range(num_groups):
        #                 print(f"GROUP ID: {i}")
        #                 mask = np.where(labels == i, True, False)
        #                 dbscan_field = high_rel_field[mask]
        #                 dbscan_dino = reduced_dino[mask]
        #                 median_rel = np.median(dbscan_field[:,0])
        #                 if median_rel > max_median_relevancy:
        #                     best_field = copy.deepcopy(dbscan_field)
        #                     best_dino = copy.deepcopy(dbscan_dino)
        #                     max_median_relevancy = median_rel
                        
        #                 group = dbscan_field[:,1:]
        #                 group = group.T
        #                 means = np.mean(group, axis=1)
        #                 group -= means[:, np.newaxis]
        #                 cov = np.cov(group)
        #                 _, evec = np.linalg.eig(cov)
        #                 evec, _ = np.linalg.qr(evec)

        #                 aligned_coords = np.matmul(evec.T, group)

        #                 min_x, min_y, min_z = np.min(aligned_coords.T, axis=0).astype(float)
        #                 max_x, max_y, max_z = np.max(aligned_coords.T, axis=0).astype(float)

        #                 dimensions = [np.round(np.abs(max_x - min_x), decimals=2), np.round(np.abs(max_y - min_y), decimals=2), np.round(np.abs(max_z - min_z), decimals=2)]
        #                 R = Rotation.from_matrix(evec)
        #                 euler = list(R.as_euler('xyz', False))
        #                 euler = [0.0, 0.0, -euler[2]]
        #                 quaternion = list(R.as_quat(False))
        #                 position = list(means.astype(float))
                    
        #             # Generate bounding boxes contiaining groups of points
        #             # bboxes = {"bboxes":[]}
        #             group = best_field[:,1:]
        #             dino_group = best_dino[:,1:]

        #             print("_______________________________________________________________________________________________________________________")
        #             print("Group")
        #             print(group.shape)
        #             print(dino_group.shape)
        #             median_dino = np.median(dino_group, axis=0)

        #             a_norm = np.linalg.norm(dino_group, axis=1)
        #             b_norm = np.linalg.norm(median_dino)[None, ...]

        #             cos_sim = (dino_group @ median_dino) / (a_norm * b_norm)

        #             print(cos_sim)
        #             print(cos_sim.shape)

        #             print("Max:", np.max(cos_sim))
        #             print("Mean:", np.mean(cos_sim))
        #             print("Median:", np.median(cos_sim))
        #             print("Min:", np.min(cos_sim))

        #             group = group[cos_sim > 0.5]

        #             print("_______________________________________________________________________________________________________________________")

        #             group = group.T
        #             means = np.mean(group, axis=1)
        #             group -= means[:, np.newaxis]
        #             cov = np.cov(group)
        #             _, evec = np.linalg.eig(cov)
        #             evec, _ = np.linalg.qr(evec)

        #             aligned_coords = np.matmul(evec.T, group)

        #             min_x, min_y, min_z = np.min(aligned_coords.T, axis=0).astype(float)
        #             max_x, max_y, max_z = np.max(aligned_coords.T, axis=0).astype(float)

        #             dimensions = [np.round(np.abs(max_x - min_x), decimals=2), np.round(np.abs(max_y - min_y), decimals=2), np.round(np.abs(max_z - min_z), decimals=2)]
        #             R = Rotation.from_matrix(evec)
        #             euler = list(R.as_euler('xyz', False))
        #             euler = [0.0, 0.0, -euler[2]]
        #             quaternion = list(R.as_quat(False))
        #             position = list(means.astype(float))

        #             print("Dim:", dimensions)
        #             print("Eul:", euler)
        #             print("Quat:", quaternion)
        #             print("Pos:", position)

        #             offset_position = [0.0, 0.0, 0.0]
        #             offset_euler = [0.0, 0.0, 0.0]

        #             # NOTE as we only save one, idx is 0 here. Should be tidied
        #             bbox_data = {"idx": 0, 
        #                         "data": {"dimensions": dimensions, 
        #                                 "euler": euler, 
        #                                 "quaternion": quaternion, 
        #                                 "position": position, 
        #                                 "is_reconfigured": False,
        #                                 "offset_position": offset_position, 
        #                                 "offset_euler": offset_euler
        #                                 }}
        #             bboxes["bboxes"].append(bbox_data)

        #         # Save bbox.json file
        #         #####################################
        #         original_output = Path(self.config.output_folder)
        #         path_components = list(original_output.parts)
        #         path_components[-2] = "neo"
        #         box_dir = Path(*path_components)
        #         box_dir.mkdir(parents=True, exist_ok=True)
        #         box_path = Path(*path_components) / "bbox.json"
        #         print(f"BBOX JSON FILE: {str(box_path)}")
        #         with open(str(box_path), 'w') as f:
        #             json.dump(bboxes, f, indent=2)

        return outputs

    def _get_outputs_nerfacto(self, ray_samples: RaySamples):
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "weights": weights,
            "densities": field_outputs[FieldHeadNames.DENSITY]
        }

        return field_outputs, outputs, weights

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        # loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        loss_dict = {}
        if self.training:
            unreduced_clip = self.config.clip_loss_weight * torch.nn.functional.huber_loss(
                outputs["clip"], batch["clip"], delta=1.25, reduction="none"
            )
            loss_dict["clip_loss"] = unreduced_clip.sum(dim=-1).nanmean()
            unreduced_dino = torch.nn.functional.mse_loss(outputs["dino"], batch["dino"], reduction="none")
            loss_dict["dino_loss"] = unreduced_dino.sum(dim=-1).nanmean()
        return loss_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["lerf"] = list(self.lerf_field.parameters())
        return param_groups
    
    def get_ray_bundle_for_ortho_camera(self) -> RayBundle:
        # Add custom camera for voxelgrid generation
        # camera_to_worlds = torch.tensor([[[1.0, 0.0, 0.0, 0.0],
        #                                  [0.0, 1.0, 0.0, 0.0],
        #                                  [0.0, 0.0, 1.0, 0.0]]], device=self.device)
        
        fx = torch.tensor([[500.0]], device=self.device)
        fy = torch.tensor([[500.0]], device=self.device)
        cx = torch.tensor([[320.0]], device=self.device)
        cy = torch.tensor([[240.0]], device=self.device)
        width = torch.tensor([[640]], device=self.device)
        height = torch.tensor([[480]], device=self.device)

        ortho_camera = Cameras(self.camera_to_worlds, fx, fy, cx, cy, width, height, camera_type=CameraType.ORTHOPHOTO)

        return ortho_camera.generate_rays(camera_indices=0, keep_shape=True)
    
    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Assumes a ray-based model.

        Args:
            camera: generates raybundle
        """
        self.camera_to_worlds = camera.camera_to_worlds
        return self.get_outputs_for_camera_ray_bundle(
            camera.generate_rays(camera_indices=0, keep_shape=True, obb_box=obb_box)
        )
    
    def forward(self, ray_bundle: Union[RayBundle, Cameras], is_ortho: Optional[bool] = False) -> Dict[str, Union[torch.Tensor, List]]:
        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        return self.get_outputs(ray_bundle, is_ortho=is_ortho)
    