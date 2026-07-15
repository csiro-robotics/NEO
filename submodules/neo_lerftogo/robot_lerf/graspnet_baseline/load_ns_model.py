from pathlib import Path

import matplotlib.pyplot as plt
import open3d as o3d
import trimesh as tr
import matplotlib
from typing import Dict, Tuple

import cv2
import viser
import viser.transforms as tf
import time

import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.exporter.exporter_utils import generate_point_cloud

from collections import deque
from scipy import ndimage
from scipy.spatial.transform import Rotation
import copy
import torchvision
import kornia.morphology as kmorph
import kornia.filters as kfilters
import os

class RealsenseCamera:
    # taken from graspnet demo parameters
    realsense = o3d.camera.PinholeCameraIntrinsic(
        1280, 720, 631.54864502, 631.20751953, 638.43517329, 366.49904066
    )

    @classmethod
    def get_camera(cls, c2w, center=None, image_shape=None, downscale=1) -> Cameras:
        if image_shape is None:
            height = cls.realsense.height
            width = cls.realsense.width
        else:
            height = image_shape[0]
            width = image_shape[1]

        if center is None:
            center_x = cls.realsense.intrinsic_matrix[0, 2]
            center_y = cls.realsense.intrinsic_matrix[1, 2]
        else:
            center_x = cls.realsense.intrinsic_matrix[0, 2] + (width-1)/2  - center[1] 
            center_y = cls.realsense.intrinsic_matrix[1, 2] + (height-1)/2 - center[0]

        camera = Cameras(
            camera_to_worlds = torch.Tensor(c2w).unsqueeze(0),
            fx = torch.Tensor([cls.realsense.intrinsic_matrix[0, 0]]),
            fy = torch.Tensor([cls.realsense.intrinsic_matrix[1, 1]]),
            cx = torch.Tensor([center_x]),
            cy = torch.Tensor([center_y]),
            width = torch.Tensor([width]).int(),
            height = torch.Tensor([height]).int(),
            )
        camera.rescale_output_resolution(downscale)

        return camera

class NerfstudioWrapper:
    def __init__(self, scene_path: str = None, pipeline: Pipeline = None):
        if scene_path is not None:
            _, pipeline, _, _ = eval_setup(Path(scene_path))
            pipeline.model.eval()
            self.pipeline = pipeline
        elif pipeline is not None:
            pipeline.model.eval()
            self.pipeline = pipeline
        else:
            raise ValueError("Must provide either scene_path or pipeline")
        self.camera_path: Cameras = pipeline.datamanager.train_dataset.cameras

        dp_outputs = pipeline.datamanager.train_dataparser_outputs
        applied_transform = np.eye(4)
        applied_transform[:3, :] = dp_outputs.dataparser_transform.numpy() #world to ns
        applied_transform = np.linalg.inv(applied_transform)
        applied_transform = applied_transform @ np.diag([1/dp_outputs.dataparser_scale]*3+[1]) #scale is post
        self.applied_transform = applied_transform

        self.num_cameras = len(self.camera_path.camera_to_worlds)

    # Note: applies to any real camera in world space
    def visercam_to_ns(self, c2w) -> np.ndarray:
        dp_outputs = self.pipeline.datamanager.train_dataparser_outputs
        foobar = np.concatenate([dp_outputs.dataparser_transform.numpy(), np.array([[0, 0, 0, 1]])], axis=0)
        c2w = foobar @ np.concatenate([c2w, np.array([[0, 0, 0, 1]])], axis=0)
        c2w = c2w[:3, :]
        c2w[:3, 2] *= -1
        c2w[:3, 1] *= -1
        return c2w

    def visercam_to_ns_world(self, c2w) -> np.ndarray:
        dp_outputs = self.pipeline.datamanager.train_dataparser_outputs
        foobar = np.concatenate([dp_outputs.dataparser_transform.numpy(), np.array([[0, 0, 0, 1]])], axis=0)
        c2w = foobar @ np.concatenate([c2w, np.array([[0, 0, 0, 1]])], axis=0)
        c2w = c2w[:3, :]
        return c2w

    # return in viser format
    def get_train_camera_c2w(self, train_cam_ind) -> np.ndarray:
        c2w = self.camera_path[train_cam_ind].camera_to_worlds.squeeze().numpy().copy()
        c2w[:3, 2] *= -1
        c2w[:3, 1] *= -1
        dp_outputs = self.pipeline.datamanager.train_dataparser_outputs
        foobar = np.concatenate([dp_outputs.dataparser_transform.numpy(), np.array([[0, 0, 0, 1]])], axis=0)
        c2w = np.linalg.inv(foobar) @ np.concatenate([c2w, np.array([[0, 0, 0, 1]])], axis=0)
        c2w = c2w[:3, :]
        return c2w

    def __call__(self, camera, render_lerf=False) -> Dict[str, np.ndarray]:
        if render_lerf:
            self.pipeline.model.step = 1000
        else:
            self.pipeline.model.step = 0
        camera_ray_bundle = camera.generate_rays(camera_indices=0).to(device)
        with torch.no_grad():
            outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        outputs['xyz'] = camera_ray_bundle.origins + camera_ray_bundle.directions * outputs['depth']
        for k, v in outputs.items():
            outputs[k] = v.squeeze().cpu().numpy()
        outputs['xyz'] = tr.transformations.transform_points(
            outputs['xyz'].reshape(-1, 3), 
            self.applied_transform
            ).reshape(outputs['xyz'].shape)
        return outputs

    # Helper function to rotate camera about a point along z axis
    def rotate_camera(self, curcam, rot, point):
        R = np.array([[1,  0,  0], 
                      [0,  0, -1], 
                      [0,  1,  0]])
        T = torch.tensor(R @ self.applied_transform[:3, :3]).float().to(device)
        curcam = copy.deepcopy(curcam)
        world_to_point = torch.cat((torch.cat((T, -T @ point.unsqueeze(1)), dim=1), torch.tensor([[0,0,0,1]]).to(device)), dim=0)
        rot = torch.tensor(rot)
        rotation_matrix_z = torch.tensor([[torch.cos(rot), -torch.sin(rot), 0, 0], [torch.sin(rot), torch.cos(rot), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).to(device)
        point_to_world = torch.inverse(world_to_point)
        homog_cam_to_world = torch.cat((curcam.camera_to_worlds.squeeze(), torch.tensor([[0,0,0,1]]).to(device)), dim=0)
        modcam_to_world = torch.matmul(point_to_world, torch.matmul(rotation_matrix_z, torch.matmul(world_to_point, homog_cam_to_world)))
        curcam.camera_to_worlds = modcam_to_world[:-1, :].unsqueeze(0)
        return curcam
    
    # Generate the lerf point cloud for the object
    def generate_lerf_pc(self, curcam, target_point, server):
        sweep = np.linspace(-np.pi,np.pi,11,dtype=np.float32, endpoint=False)
        dino_dim = 384 # dimension of dino feature vector
        points = []
        rgbs = []
        dinos = []
        clips = []
        for i in sweep:
            mod_curcam = self.rotate_camera(curcam, i, target_point)

            # Add camera for visualization
            c2w = mod_curcam.camera_to_worlds.cpu().numpy()[0]
            R = tf.SO3.from_matrix(c2w[:3, :3])
            R = R @ tf.SO3.from_x_radians(np.pi)
            server.scene.add_camera_frustum(
                name=f"/sweep_0_cameras/{i}",
                fov=float(2 * np.arctan((curcam.cx / curcam.fx[0]).cpu())),
                scale=0.25,
                aspect=float((curcam.cx[0] / curcam.cy[0]).cpu()),
                wxyz=R.wxyz,
                position=c2w[:3, 3],
            )

            with torch.no_grad():
                bundle = mod_curcam.generate_rays(camera_indices=0)
                outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(bundle)
            point = bundle.origins + bundle.directions * outputs["depth"]
            point = torch.reshape(point, (-1, 3))
            rgb = torch.reshape(outputs["rgb"], (-1, 3))
            dino = torch.reshape(outputs["dino"], (-1, dino_dim))
            points.append(point)
            rgbs.append(rgb)
            dinos.append(dino)
        points = torch.cat(points, dim=0)
        rgbs = torch.cat(rgbs, dim=0)
        dinos = torch.cat(dinos, dim=0).double().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.double().cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(rgbs.double().cpu().numpy())
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.01)
        if ind is not None:
            dinos = dinos[ind]
            
        return pcd, dinos, clips

    # Helper function to get DINO foreground mask and the object point in 2D
    def get_dino_first_comp_2d(self, server, curcam):
        default_view_cam = copy.deepcopy(curcam)
        
        # Hard-coded camera to world matrices for the zoomed in and overhead views of the experimental table setup
        R = np.array([[1,  0,  0,  0], 
                      [0,  0, -1,  0], 
                      [0,  1,  0,  0],
                      [0,  0,  0,  1]])
        
        camera_to_worlds_zoomed = np.array([[-1.0,  0.0,    0.0,    0.3], 
                                            [ 0.0,  0.866, -0.5,    0.6],
                                            [ 0.0, -0.5,   -0.866,  0.5],
                                            [ 0.0,  0.0,    0.0,    1.0]])
        
        camera_to_worlds_zoomed = (R @ camera_to_worlds_zoomed)[:3, :]
        camera_to_worlds_zoomed = torch.tensor(self.visercam_to_ns(camera_to_worlds_zoomed), device=device).float().unsqueeze(0)
        
        camera_to_worlds_overhead = np.array([[-1.0,  0.0,  0.0,  0.3], 
                                              [ 0.0,  1.0,  0.0,  0.28],
                                              [ 0.0,  0.0, -1.0,  0.6],
                                              [ 0.0,  0.0,  0.0,  1.0]])
        
        camera_to_worlds_overhead = (R @ camera_to_worlds_overhead)[:3, :]
        camera_to_worlds_overhead = torch.tensor(self.visercam_to_ns(camera_to_worlds_overhead), device=device).float().unsqueeze(0)

        # Find the first principal component of the DINO feature vectors
        default_view_cam.camera_to_worlds = camera_to_worlds_zoomed

        # Add camera for visualization
        temp_cam_0 = copy.deepcopy(default_view_cam)
        c2w = temp_cam_0.camera_to_worlds.cpu().numpy()[0]
        R = tf.SO3.from_matrix(c2w[:3, :3])
        R = R @ tf.SO3.from_x_radians(np.pi)
        server.scene.add_camera_frustum(
            name=f"/cameras/zoomed",
            fov=float(2 * np.arctan((temp_cam_0.cx / temp_cam_0.fx[0]).cpu())),
            scale=0.25,
            aspect=float((temp_cam_0.cx[0] / temp_cam_0.cy[0]).cpu()),
            wxyz=R.wxyz,
            position=c2w[:3, 3],
        )

        bundle = default_view_cam.generate_rays(camera_indices=0)
        outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(bundle)
        dinos = outputs["dino"]
        dinos = dinos.view(-1, dinos.shape[-1])
        _, _, v = torch.pca_lowrank(dinos, niter=5)
        dino_first_comp_2d = v[..., :1]

        # Save the first principal component as an image for debugging
        mul_dinos = dinos @ dino_first_comp_2d
        mul_dinos -= torch.min(mul_dinos)
        mul_dinos /= torch.max(mul_dinos)
        # cv2.imwrite("dino_first_comp_2d_zoomed.png", np.uint8((mul_dinos).reshape(180, 320, -1).detach().cpu().numpy() * 255))
        
        # Use the first principal component to create a foreground mask
        default_view_cam.camera_to_worlds = camera_to_worlds_overhead

        # Add camera for visualization
        temp_cam_1 = copy.deepcopy(default_view_cam)
        c2w = temp_cam_1.camera_to_worlds.cpu().numpy()[0]
        R = tf.SO3.from_matrix(c2w[:3, :3])
        R = R @ tf.SO3.from_x_radians(np.pi)
        server.scene.add_camera_frustum(
            name=f"/cameras/overhead",
            fov=float(2 * np.arctan((temp_cam_1.cx / temp_cam_1.fx[0]).cpu())),
            scale=0.25,
            aspect=float((temp_cam_1.cx[0] / temp_cam_1.cy[0]).cpu()),
            wxyz=R.wxyz,
            position=c2w[:3, 3],
        )

        bundle = default_view_cam.generate_rays(camera_indices=0)
        outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(bundle)
        dinos = outputs["dino"]
        dinos = dinos.view(-1, dinos.shape[-1])

        # Save the first principal component as an image for debugging
        mul_dinos = dinos @ dino_first_comp_2d
        mul_dinos -= torch.min(mul_dinos)
        mul_dinos /= torch.max(mul_dinos)
        # cv2.imwrite("dino_first_comp_2d_overhead.png", np.uint8((mul_dinos).reshape(180, 320, -1).detach().cpu().numpy() * 255))

        THRESHOLD = 0.8 # Threshold to create the DINO mask depending if the objts are in the foreground or background
        mask = torch.where(torch.matmul(dinos, dino_first_comp_2d) < THRESHOLD, 1, 0)
        mask_test=torch.reshape(mask, (curcam.height.item(), curcam.width.item())).cpu().numpy()
        labels, _=ndimage.label(mask_test)
        component_sizes = np.bincount(labels.ravel())
        largets_component = np.argmax(component_sizes[1:]) + 1
        largets_component_area = component_sizes[largets_component]
        if largets_component_area > 0.5 * mask.shape[0] * mask.shape[1]:
            dino_first_comp_2d = -1 * dino_first_comp_2d
            mask = torch.where(torch.matmul(dinos, dino_first_comp_2d) < THRESHOLD, 1, 0)
        mask = cv2.morphologyEx(np.uint8(mask.reshape(180, 320, -1).detach().cpu().numpy() * 255), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # TODO: Hard-coded mask for the overhead view
        mask[:, :60] = 0.0
        mask[:, -60:] = 0.0
        mask = torch.tensor(mask / 255).to(device).view(-1, 1)
        relevancy_0 = outputs["relevancy_0"]

        # Save the mask and relevancy as an image for debugging
        # cv2.imwrite("dino_mask_overhead.png", np.uint8(mask.reshape(180, 320, -1).detach().cpu().numpy() * 255))
        # cv2.imwrite("relevancy_0_overhead.png", np.uint8(relevancy_0.detach().cpu().numpy() * 255))

        relevancy_0 = relevancy_0.view(-1, relevancy_0.shape[-1])
        masked_relevancy = relevancy_0 * mask
        masked_relevancy = masked_relevancy.view(curcam.height, curcam.width, 1).permute(2, 0, 1).unsqueeze(0)
        blur = kfilters.BoxBlur((3, 3))
        masked_relevancy = blur(masked_relevancy)[0].permute(1, 2, 0)

        print(masked_relevancy.shape)
        # Save the masked relevancy as an image for debugging
        # cv2.imwrite("masked_relevancy_overhead.png", np.uint8(masked_relevancy.squeeze().detach().cpu().numpy() * 255))

        # Flatten relevancy and all points
        flat_relevancy = masked_relevancy.squeeze().flatten()
        all_points = bundle.origins + bundle.directions * outputs["depth"]
        all_points = torch.reshape(all_points, (-1, 3))

        # Mask out points under the table (z < 0)
        valid_mask = all_points[:, 2] >= 0
        if valid_mask.sum() == 0:
            print("Warning: No points above table found! Returning most relevant point regardless of z.")
            target_idx = torch.topk(flat_relevancy, 1, largest=True).indices.item()
        else:
            filtered_relevancy = flat_relevancy[valid_mask]
            filtered_points = all_points[valid_mask]
            # Get index in filtered array, then map back to original
            rel_idx = torch.topk(filtered_relevancy, 1, largest=True).indices.item()
            # Ensure all indexing tensors are on cpu
            valid_mask_cpu = valid_mask.cpu() if valid_mask.is_cuda else valid_mask
            target_idx = torch.arange(all_points.shape[0])[valid_mask_cpu][rel_idx].item()

        print("Target index", (target_idx % curcam.width).item(), (target_idx // curcam.width).item())
        target_points = all_points[target_idx]
        print("Target points", target_points)
        return dino_first_comp_2d, target_points
    
    # Helper function to project a point cloud to image space
    def project_to_image(self, curcam, point_cloud, round_px=True):
        K = curcam.get_intrinsics_matrices().to(device)[0]
        points_proj = torch.matmul(K, point_cloud)
        if len(points_proj.shape) == 1:
            points_proj = points_proj[:, None]
        point_depths = points_proj[2, :]
        point_z = point_depths.repeat(3, 1)
        points_proj = torch.divide(points_proj, point_z)
        if round_px:
            points_proj = torch.round(points_proj)
        points_proj = points_proj[:2, :].int()

        valid_ind = torch.where((points_proj[0, :] >= 0) & (points_proj[1, :] >= 0) & (points_proj[0, :] < curcam.image_width.item()) & (points_proj[1, :] < curcam.image_height.item()), 1, 0).to(device)
        valid_ind = torch.argwhere(valid_ind.squeeze())
        depth_data = torch.ones([curcam.image_height, curcam.image_width], device=device) * -1
        depth_data[points_proj[1, valid_ind], points_proj[0, valid_ind]] = point_depths[valid_ind]
        reverse = valid_ind.flip(0)
        depth_data[points_proj[1, reverse], points_proj[0, reverse]] = torch.minimum(point_depths[reverse], depth_data[points_proj[1, reverse], points_proj[0, reverse]])
        return depth_data

    # Flood fill a point cloud given seed points
    def flood_fill_3d(self, pcd, pcd_tree, dino_vectors, seed_indices, tolerance):
        q = deque(seed_indices)
        seed_value = dino_vectors[seed_indices].mean(axis=0)
        np_pts = np.asarray(pcd.points)
        mask = np.zeros(np_pts.shape[0], dtype=bool)
        count = 0
        while q:
            # print(len(q))
            pt_indx = q.popleft()
            if mask[pt_indx] == False:
                mask[pt_indx] = True
                [_, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[pt_indx], 0.03)
                np_idx = np.asarray(idx)
                idx_tolerance = np.mean((dino_vectors[np_idx]- seed_value) ** 2, axis=-1)
                idx_of_id = np.argwhere(idx_tolerance < tolerance).squeeze()
                temp_list = np_idx[idx_of_id]
                added = temp_list.tolist() if isinstance(temp_list, np.ndarray) else [temp_list]
                q.extend(added)
                count += 1
        return mask
    
    # Main function to get generate the lerf point cloud for the object part
    def get_lerf_pointcloud(self, server: viser.ViserServer, curcam, render_lerf=True) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        if render_lerf:
            self.pipeline.model.step = 1000
        else:
            self.pipeline.model.step = 0
        curcam = curcam.to(device)
        dino_first_comp_2d, target_points = self.get_dino_first_comp_2d(server, curcam)
        
        dino_first_comp_2d = dino_first_comp_2d.double().cpu().detach().numpy()
        if curcam is None:
            return

        print("Target points", target_points)

        # Add frame to visualize target points
        server.scene.add_frame("Target Points", position=target_points.detach().cpu().numpy(), axes_length=0.1)

        X_OFFSET, Y_OFFSET = 0.7, 0.0
        c2w = np.array([[ 0.0,  0.5,       -0.8660254,  0.0],
                        [ 1.0,  0.0,        0.0,        0.0],
                        [ 0.0, -0.8660254, -0.5,        0.0],
                        [ 0.0,  0.0,        0.0,        1.0]])
        
        R = np.array([[1,  0,  0,  0], 
                      [0,  0, -1,  0], 
                      [0,  1,  0,  0],
                      [0,  0,  0,  1]])
        
        c2w = (R @ c2w)[:3, :]
        c2w = torch.tensor(self.visercam_to_ns(c2w), device=device).float().unsqueeze(0)
        c2w[:, :3, 3] = target_points + torch.tensor(self.applied_transform[:3, :3].T @ np.array([X_OFFSET, -0.2, Y_OFFSET]), device=device)
        curcam.camera_to_worlds = c2w

        c2w = curcam.camera_to_worlds.cpu().numpy()[0]
        R = tf.SO3.from_matrix(c2w[:3, :3])
        R = R @ tf.SO3.from_x_radians(np.pi)
        server.scene.add_camera_frustum(
            name=f"/cameras/hardcoded",
            fov=float(2 * np.arctan((curcam.cx / curcam.fx[0]).cpu())),
            scale=0.25,
            aspect=float((curcam.cx[0] / curcam.cy[0]).cpu()),
            wxyz=R.wxyz,
            position=c2w[:3, 3],
        )
        
        print("Camera for pcd", curcam.camera_to_worlds)
        
        pcd, oryg_dinos, place_pt = self.generate_lerf_pc(curcam, target_points, server)

        # Hard-coded table bounding box
        # table_center = np.array((0.27, 0.28, 0.11))
        # inflated_bbox = o3d.geometry.OrientedBoundingBox(table_center,np.eye(3),np.array((0.75, 0.78, 0.4)))
        # pcd = pcd.crop(inflated_bbox)

        unchanged_pcd = copy.deepcopy(pcd)

        colors = copy.deepcopy(np.asarray(unchanged_pcd.points)[:, 2].squeeze())
        colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-6)
        colors = matplotlib.colormaps['jet'](colors)[:, :3]

        server.scene.add_point_cloud(
                name=f"/unchanged_pcd",
                points=np.asarray(unchanged_pcd.points),
                colors=colors,
                point_size=0.007
            )
        
        # pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        # [_, idx, _] = pcd_tree.search_knn_vector_3d(target_points.double().cpu().detach().numpy(), 1)
        # seed_indices = np.array(idx)
        # dinos = np.matmul(oryg_dinos, dino_first_comp_2d)
        # print("floodfill start")
        # mask = self.flood_fill_3d(pcd, pcd_tree, dinos, seed_indices, tolerance=7.5)
        # print("floodfill end")
        # pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[mask, :])
        # pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask, :])

        # Instead of flood fill, keep only points at least 1cm above z=0
        points_np = np.asarray(pcd.points)
        mask = points_np[:, 2] > 0.01
        pcd.points = o3d.utility.Vector3dVector(points_np[mask, :])
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask, :])

        # Hard-coded table bounding box
        table_center = np.array((0.27, 0.28, 0.11))
        inflated_bbox = o3d.geometry.OrientedBoundingBox(table_center,np.eye(3),np.array((0.75, 0.78, 0.4)))
        pcd = pcd.crop(inflated_bbox)

        dino_pcd = copy.deepcopy(pcd)
        points, rgbs, relevancies, points_0, relevancies_0 = [], [], [], [], []
        
        sweep= np.linspace(-np.pi,np.pi,11,dtype=np.float32, endpoint=False)
        for i in sweep:
            mod_curcam = self.rotate_camera(curcam, i, target_points)

            # Add camera for visualization
            c2w = mod_curcam.camera_to_worlds.cpu().numpy()[0]
            R = tf.SO3.from_matrix(c2w[:3, :3])
            R = R @ tf.SO3.from_x_radians(np.pi)
            server.scene.add_camera_frustum(
                name=f"/sweep_1_cameras/{i}",
                fov=float(2 * np.arctan((curcam.cx / curcam.fx[0]).cpu())),
                scale=0.25,
                aspect=float((curcam.cx[0] / curcam.cy[0]).cpu()),
                wxyz=R.wxyz,
                position=c2w[:3, 3],
            )

            homog_cam_to_world = torch.cat((mod_curcam.camera_to_worlds.squeeze(), torch.tensor([[0,0,0,1]]).to(device)), dim=0)
            homog_world_to_cam = torch.inverse(homog_cam_to_world)
            pcd_points = np.asarray(pcd.points)
            homog_pcd_points = np.ones((pcd_points.shape[0],4))
            homog_pcd_points[:, :3] = pcd_points
            rotation_matrix_x = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], device=device).float() #180 deg rot about cam x axis
            point_cloud_in_camera_frame = torch.matmul(rotation_matrix_x, torch.matmul(homog_world_to_cam, torch.tensor(homog_pcd_points.T, device=device).float()))
            dino_2d_data = self.project_to_image(mod_curcam, point_cloud_in_camera_frame[:3, :])
            dino_2d_mask = torch.where(dino_2d_data > 0, 1, 0)
            image_point = dino_2d_mask.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).float()
            kernel = torch.ones(3, 3).to(device)
            dino_2d_mask = kmorph.dilation(image_point, kernel)
            dino_2d_mask = dino_2d_mask.squeeze()[0, :, :]
            with torch.no_grad():
                bundle = mod_curcam.generate_rays(camera_indices=0)
                s = (*bundle.shape,1)
                bundle.nears = torch.full(s , 0.05, device=device)
                bundle.fars = torch.full(s, 10, device=device)
                outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(bundle)
            # testing if this works:
            lerf_mask = torch.where((outputs['relevancy_1'] > 0) & (dino_2d_mask.unsqueeze(2) > 0) & (torch.abs(outputs['depth'].squeeze() - dino_2d_data).unsqueeze(2) < 0.3), 1, 0)
            # ACTUAL:
            # lerf_mask = torch.where((outputs['relevancy_1'] > 0) & (torch.abs(outputs['depth'].squeeze() - dino_2d_data).unsqueeze(2) < 0.08), 1, 0)
            oryg_points = bundle.origins + bundle.directions * outputs["depth"]
            oryg_points = torch.reshape(oryg_points, (-1, 3))
            rgb = torch.reshape(outputs["rgb"], (-1, 3))
            relevancy = torch.reshape(outputs["relevancy_1"], (-1, 1))
            lerf_mask = torch.reshape(lerf_mask, (-1, 1)).squeeze()
            point = oryg_points[torch.nonzero(lerf_mask).squeeze()]
            rgb = rgb[torch.nonzero(lerf_mask).squeeze()]
            relevancy = relevancy[torch.nonzero(lerf_mask).squeeze()]
            if len(point.shape) > 1:
                points.append(point)
                rgbs.append(rgb)
                relevancies.append(relevancy)

            lerf_mask = torch.where((outputs['relevancy_0'] > 0) & (dino_2d_mask.unsqueeze(2) > 0) & (torch.abs(outputs['depth'].squeeze() - dino_2d_data).unsqueeze(2) < 0.3), 1, 0)
            lerf_mask = torch.reshape(lerf_mask, (-1, 1)).squeeze()
            relevancy_0 = torch.reshape(outputs["relevancy_0"], (-1, 1))
            relevancy_0 = relevancy_0[torch.nonzero(lerf_mask).squeeze()]
            point_0 = oryg_points[torch.nonzero(lerf_mask).squeeze()]
            if len(point_0.shape) > 1:
                points_0.append(point_0)
                relevancies_0.append(relevancy_0)

        points = torch.cat(points, dim=0)
        points_0 = torch.cat(points_0, dim=0)
        rgbs = torch.cat(rgbs, dim=0)
        relevancies = torch.cat(relevancies, dim=0)
        relevancies_0 = torch.cat(relevancies_0, dim=0)
        if len(relevancies) == 0 or len(relevancies_0) == 0:
            print("No points found with positive lerf relevancy")
            return None, None, None
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.double().cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(rgbs.double().cpu().numpy())
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.00001)
        relevancies = relevancies - relevancies.min()
        relevancies = relevancies / relevancies.max()
        relevancies = relevancies[ind]
        oryg_dinos = oryg_dinos[ind]

        pcd_0 = o3d.geometry.PointCloud()
        pcd_0.points = o3d.utility.Vector3dVector(points_0.double().cpu().numpy())
        pcd_0, ind = pcd_0.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.00001)
        relevancies_0 = relevancies_0 - relevancies_0.min()
        relevancies_0 = relevancies_0 / relevancies_0.max()
        relevancies_0 = relevancies_0[ind]

        if len(place_pt) > 0:
            place_pt = torch.tensor([place_pt.pop() in range(place_pt.count())]).to(device)
        else:
            place_pt = torch.tensor(place_pt).to(device)

        return pcd, relevancies.cpu().numpy(), dino_pcd, target_points, place_pt, pcd_0, relevancies_0.cpu().numpy(), oryg_dinos

    def create_pointcloud(self) -> Tuple[tr.PointCloud, np.ndarray]:
        self.pipeline.model.step = 0
        orig_num_rays_per_batch = self.pipeline.datamanager.train_pixel_sampler.num_rays_per_batch
        self.pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = 100000

        global_pointcloud: o3d.geometry.PointCloud = generate_point_cloud(
            self.pipeline, 
            remove_outliers=True, 
            std_ratio=0.1,
            depth_output_name='depth',
            num_points = 1000000
            )
        self.pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = orig_num_rays_per_batch

        print("Applied transform:", self.applied_transform)

        global_pointcloud.points = o3d.utility.Vector3dVector(tr.transformations.transform_points(np.asarray(global_pointcloud.points), self.applied_transform)) #nerfstudio pc to world/viser pc
        global_pointcloud = global_pointcloud.voxel_down_sample(0.001)

        rotation = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        global_pointcloud.rotate(rotation, center=np.array([0, 0, 0]))

        #find the table by fitting a plane
        # _,plane_ids = global_pointcloud.segment_plane(.003,3,1000)
        # plane_pts = global_pointcloud.select_by_index(plane_ids)
        # #downsample the points to reduce density
        # plane_pts = plane_pts.voxel_down_sample(0.001)
        # #remove outlier points from the plane to restrict it to the table points
        # plane_pts,_=plane_pts.remove_radius_outlier(150,.01,print_progress=True)
        # #get the oriented bounding box of this restricted plane (should be the table)
        # bbox_o3d = plane_pts.get_oriented_bounding_box()
        # #stretch the 3rd dimension, which corresponds to raising the bounding box
        # stretched_extent = np.array((bbox_o3d.extent[0],bbox_o3d.extent[1],1))
        # inflated_bbox = o3d.geometry.OrientedBoundingBox(bbox_o3d.center,bbox_o3d.R,stretched_extent)

        # it's probably more robust to just hardcode the table
        table_center = np.array((0.27, 0.28, 0.11))
        inflated_bbox = o3d.geometry.OrientedBoundingBox(table_center,np.eye(3),np.array((0.75, 0.78, 0.4)))
        # inv_t = self.applied_transform
        # inflated_bbox = inflated_bbox.rotate(inv_t[:3,:3])
        # inflated_bbox = inflated_bbox.translate(inv_t[:3,3])
        world_pointcloud_o3d = global_pointcloud.crop(inflated_bbox)
        world_pointcloud = tr.PointCloud(
            vertices=np.asarray(world_pointcloud_o3d.points),
            colors=np.asarray(world_pointcloud_o3d.colors)
        )
        global_pointcloud = tr.PointCloud(
            vertices=np.asarray(global_pointcloud.points),
            colors=np.asarray(global_pointcloud.colors)
        )
        return world_pointcloud,global_pointcloud,table_center
