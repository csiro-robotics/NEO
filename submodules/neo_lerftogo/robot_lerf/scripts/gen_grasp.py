import copy
import time
import numpy as np
from sklearn.cluster import DBSCAN
import trimesh as tr
import tyro

import os
import os.path as osp
import re
import yaml
from pathlib import Path
import open3d as o3d
import matplotlib
import json
from typing import List, Dict, Tuple
from scipy.spatial.transform import Rotation

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import viser
import viser.transforms as tf
from viser import _messages
from autolab_core import RigidTransform

from graspnetAPI import GraspGroup, Grasp
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.engine.trainer import TrainerConfig

from robot_lerf.graspnet_baseline.load_ns_model import NerfstudioWrapper, RealsenseCamera
from robot_lerf.graspnet_baseline.graspnet_module import GraspNetModule
from robot_lerf.capture_utils import _generate_hemi
from robot_lerf.grasp_planner_cmk import UR5GraspPlanner


def get_relevancy_pointcloud(ns_wrapper: NerfstudioWrapper, **kwargs):
    """Get relevancy pointcloud, used to get semantic score

    Args:
        ns_wrapper (NerfstudioWrapper): nerf scene

    Returns:
        o3d.utility.Vector3DVector: points in pointcloud (xyz)
        np.ndarray: relevancy score
    """
    lerf_xyz, lerf_relevancy = [], []

    # Camera is looking up if no rotation
    center_pos_matrix = np.array([[-1.0,  0.0,         0.0,         0.3], 
                                  [ 0.0,  0.70710678, -0.70710678,  0.8],
                                  [ 0.0, -0.70710678, -0.70710678,  0.3],
                                  [ 0.0,  0.0,         0.0,         1.0]])
    
    R = np.array([[1,  0,  0,  0], 
                  [0,  0, -1,  0], 
                  [0,  1,  0,  0],
                  [0,  0,  0,  1]])
    
    center_pos_matrix = (R @ center_pos_matrix)[:3, :]
    
    c2w = ns_wrapper.visercam_to_ns(center_pos_matrix)
    rscam = RealsenseCamera.get_camera(c2w, downscale=1/4)
    lerf_pcd, lerf_relevancy, dino_pcd, pick_object_pt, place_pt, pcd_0, relevancies_0, oryg_dinos = ns_wrapper.get_lerf_pointcloud(kwargs['server'], rscam)
    pick_object_pt = tr.transformations.transform_points(pick_object_pt.unsqueeze(0).cpu().numpy(), ns_wrapper.applied_transform).squeeze()
    if place_pt is not None:
        if len(place_pt) > 0:
            place_pt = tr.transformations.transform_points(place_pt.unsqueeze(0).cpu().numpy(), ns_wrapper.applied_transform).squeeze()
        else:
            place_pt = None

    rotation = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    lerf_pcd_temp = tr.transformations.transform_points(np.asarray(lerf_pcd.points), ns_wrapper.applied_transform)
    pcd_0_temp = tr.transformations.transform_points(np.asarray(pcd_0.points), ns_wrapper.applied_transform)
    dino_pcd_temp = tr.transformations.transform_points(np.asarray(dino_pcd.points), ns_wrapper.applied_transform)
    lerf_pcd_temp = (rotation @ lerf_pcd_temp.T).T
    pcd_0_temp = (rotation @ pcd_0_temp.T).T
    dino_pcd_temp = (rotation @ dino_pcd_temp.T).T
    
    lerf_pcd.points = o3d.utility.Vector3dVector(lerf_pcd_temp) #nerfstudio pc to world/viser pc
    pcd_0.points = o3d.utility.Vector3dVector(pcd_0_temp)
    dino_pcd.points = o3d.utility.Vector3dVector(dino_pcd_temp)
    lerf_points_o3d = lerf_pcd.points
    pcd_0_points_o3d = pcd_0.points
    dino_points_o3d = dino_pcd.points

    return lerf_points_o3d, lerf_relevancy, dino_points_o3d, pick_object_pt, place_pt, pcd_0_points_o3d, relevancies_0, oryg_dinos, rscam

def get_grasps(
    graspnet: GraspNetModule,
    world_pointcloud: tr.PointCloud,
    hemisphere: List[RigidTransform],
    graspnet_batch_size: int = 15,
    ) -> GraspGroup:
    """Get grasps from graspnet, as images taken from the hemisphere
    
    Args: 
        graspnet (GraspNetModule): graspnet module
        world_pointcloud (tr.PointCloud): world pointcloud
        hemisphere (List[RigidTransform]): list of camera poses
    
    Returns:
        GraspGroup: grasps
    """
    torch.cuda.empty_cache()
    gg_all = None
    for i in range(0, len(hemisphere), graspnet_batch_size):
        start = time.time()
        ind_range = range(i, min(i+graspnet_batch_size, len(hemisphere)))
        rgbd_cropped_list = []
        for j in ind_range:
            c2w = hemisphere[j].matrix[:3,:]
            rgbd_cropped = world_pointcloud.copy()
            rgbd_cropped.vertices = tr.transformations.transform_points(
                rgbd_cropped.vertices,
                np.linalg.inv(np.concatenate([c2w, [[0, 0, 0, 1]]], axis=0))
            )
            rgbd_cropped_list.append(rgbd_cropped)
        print("Transform time: ", time.time() - start)

        gg_list = graspnet(rgbd_cropped_list)
        for g_ind, gg in enumerate(gg_list):
            c2w = hemisphere[i + g_ind].matrix[:3,:]
            gg.transform(np.concatenate([c2w, [[0, 0, 0, 1]]], axis=0))
        print(f"Grasp pred time: {time.time() - start:.2f}s")
        start = time.time()

        gg_all_curr = gg_list[0]
        for gg in gg_list[1:]:
            gg_all_curr.add(gg)
        gg = gg_all_curr

        print("Num of grasps: ", len(gg))

        # If the grasps are too close to the ground, then lift them a bit.
        # This is hardcoded though, so it might not work for all scenes
        gg_translations = gg.translations
        gg_translations[gg_translations[:, 2] < 0.0] += np.tile(np.array([0, 0, 0.01]), ((gg_translations[:, 2] < 0.0).sum(), 1))
        gg.translations = gg_translations
        # gg[gg.translations[:, 2] < -0.16].translations += np.tile(np.array([0, 0, 0.04]), ((gg.translations[:, 2] < -0.16).sum(), 1))
        gg = gg[(gg.translations[:, 0] > 0.22)] #& (gg.translations[:, 2] < 0.05)]

        gg = gg[np.abs(gg.rotation_matrices[:, :, 1][:, 2]) < 0.5]

        # gg = gg[gg.scores > 0.6]
        if len(gg) == 0:
            continue

        gg = gg.nms(translation_thresh=0.01, rotation_thresh=30.0/180.0*np.pi)

        print("Num of grasps after nms filtering: ", len(gg))

        # select grasps that are not too close to the table
        # Currently, this function does general grasp filtering (using collision detection, grasp includes non-table components, ...)
        gg = graspnet.local_collision_detection(gg)

        print(f"Collision detection time: {time.time() - start:.2f}s")
        print(f"Post proc time: {time.time() - start:.2f}s")
        if gg_all is None:
            gg_all = gg
        else:
            gg_all.add(gg)

    if gg_all is None:
        return GraspGroup()
    
    gg_all = gg_all.nms(translation_thresh=0.01, rotation_thresh=30.0/180.0*np.pi)
    gg_all.sort_by_score()
    # Diagnostics: print width statistics to verify effect of GRASP_MAX_WIDTH changes
    try:
        w = gg_all.widths
        if len(w) > 0:
            print(f"Grasp widths: min={w.min():.4f} m, mean={w.mean():.4f} m, max={w.max():.4f} m")
    except Exception:
        pass
    torch.cuda.empty_cache()

    return gg_all

def instantiate_scene_from_model(
    server: viser.ViserServer,
    graspnet_ckpt: str,
    floor_height: float,
    scene_name: str,
    config_path: str = None,
    pipeline: Pipeline = None,
) -> Tuple[NerfstudioWrapper, tr.PointCloud, tr.PointCloud, np.ndarray, GraspNetModule, GraspGroup, np.ndarray]:
    """Instantiate a scene from a NerfStudio model, as well as the associated GraspNetModule.

    Args:
        server (viser.ViserServer): viser server, used for visualization
        config_path (str): Nerfstudio model config path
        graspnet_ckpt (str): GraspNet checkpoint path
        floor_height (float): floor height

    Returns:
        Tuple[NerfstudioWrapper, tr.PointCloud, tr.PointCloud, np.ndarray, GraspNetModule]:
         - world_pointcloud: world pointcloud
         - global_pointcloud: nerf pointcloud (not cropped)
         - table_center: table center
         - graspnet: graspnet module
         - grasps: GraspGroup
         - overall_scores: overall scores

    """
    if config_path is not None:
        ns_wrapper = NerfstudioWrapper(scene_path=config_path)
    elif pipeline is not None:
        ns_wrapper = NerfstudioWrapper(pipeline=pipeline)
    else:
        raise ValueError("Must provide either scene_path or pipeline")
    
    world_pointcloud, global_pointcloud, table_center = ns_wrapper.create_pointcloud()

    graspnet = GraspNetModule()
    graspnet.init_net(graspnet_ckpt, global_pointcloud, cylinder_radius=0.04, floor_height=floor_height)

    server.scene.add_point_cloud(
        name=f"/world_pointcloud",
        points=np.asarray(world_pointcloud.vertices),
        colors=np.asarray(world_pointcloud.visual.vertex_colors[:, :3]),
        point_size=0.002,
    )

    hemi_radius = 2.0
    hemi_theta_N = 30
    hemi_phi_N = 15
    hemi_th_range = 180
    hemi_phi_down = 0
    hemi_phi_up = 90

    # Load config file
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)

    force_recompute = os.environ.get("FORCE_RECOMPUTE_GRASPS", "0") == "1"
    if osp.exists(config.get_base_dir() / "grasps.npy") and not force_recompute:
        grasps = GraspGroup(np.load(config.get_base_dir() / "grasps.npy"))
    else:
        grasp_hemisphere = _generate_hemi(
            hemi_radius,hemi_theta_N,hemi_phi_N,
            (np.deg2rad(-hemi_th_range),np.deg2rad(hemi_th_range)),
            (np.deg2rad(hemi_phi_down),np.deg2rad(hemi_phi_up)),
            center_pos=table_center,look_pos=table_center
            )
        
        grasps = get_grasps(graspnet, world_pointcloud, grasp_hemisphere)
        grasps.save_npy(config.get_base_dir() / "grasps.npy")

    # # just add all the grasps lol
    # for i, grasp in enumerate(grasps):
    #     add_grasps(server, grasp, i, 0.5)

    return (
        ns_wrapper,
        world_pointcloud,
        global_pointcloud,
        table_center,
        grasps,
        np.array(grasps.scores)
    )

def add_grasps(
    server: viser.ViserServer,
    grasp: Grasp,
    ind: int,
    score: float,
) -> Tuple[viser.SceneNodeHandle, viser.SceneNodeHandle, viser.SceneNodeHandle, viser.SceneNodeHandle]:
    """Curry function for adding grasps to the scene.

    Args:
        server (viser.ViserServer): _description_
        grasp (Grasp): _description_
        ind (int): _description_
        score (float): The color is based on the score -- put it from 0 to 1

    Returns:
        Tuple[viser.SceneNodeHandle, viser.SceneNodeHandle, viser.SceneNodeHandle]:
         - frame_handle: [Grasp frame (graspnetAPI)] to [world]
         - grasp_handle: mesh
         - ur5_handle: [UR5 frame (EE)] to [Grasp frame (graspnetAPI)]
    """
    colormap = matplotlib.colormaps['RdYlGn']
    robot_frame_R = RigidTransform(
        rotation=RigidTransform.y_axis_rotation(np.pi/2) @ RigidTransform.z_axis_rotation(np.pi/2)
    )

    default_grasp = Grasp()
    default_grasp.depth = grasp.depth
    default_grasp.width = grasp.width
    default_grasp.height = grasp.height
    default_grasp = default_grasp.to_open3d_geometry()

    frame_handle = server.scene.add_frame(
        name=f'/predicted_grasps/grasps_{ind}',
        wxyz=tf.SO3.from_matrix(grasp.rotation_matrix).wxyz,
        position=grasp.translation,
        show_axes=False
    )
    grasp_handle = server.scene.add_mesh_simple(
        name=f'/predicted_grasps/grasps_{ind}/mesh',
        vertices=np.asarray(default_grasp.vertices),
        faces=np.asarray(default_grasp.triangles),
        color=colormap(score)[:3],
    )
    ur5_handle = server.scene.add_frame(
        name=f'/predicted_grasps/grasps_{ind}/ur5',
        wxyz=robot_frame_R.quaternion,
        # position=np.array([0.03, 0, 0]),
        position=np.array([grasp.depth-0.015, 0, 0]),
        axes_length=0.05,
        axes_radius=0.002,
        show_axes=True,
        visible=False
    )
    return frame_handle, grasp_handle, ur5_handle


def get_bbox_from_grasp(gg: Grasp) -> o3d.geometry.OrientedBoundingBox:
    center = gg.translation
    R = gg.rotation_matrix
    H= np.eye(4)
    H[:3,:3] = R
    H[:3,3] = center
    extent=np.array((gg.depth,gg.width,gg.height))
    box = o3d.geometry.OrientedBoundingBox(center,H[:3,:3],extent)
    return box


class GenGrasp():

    def __init__(self):
        self.gt_offset_pos = [0.0, 0.0, 0.0]
        self.gt_offset_euler = [0.0, 0.0, 0.0]
        self.reconfigure = False
        pass
                 
    def populate(self, config_path: str = None, graspnet_ckpt: str = 'robot_lerf/graspnet_baseline/logs/log_kn/checkpoint.tar'):
        server = viser.ViserServer()
        self.server = server

        # TODO: Add a camera position and look at
        # server._queue(_messages.SetCameraPositionMessage((0.87, 0.19, 0.24)))
        # server._queue(_messages.SetCameraLookAtMessage((0.42, 0.066, -0.185)))
        
        curr_client = None
        @server.on_client_connect
        def set_init_cam(client: viser.ClientHandle) -> None:
            nonlocal curr_client
            curr_client = client
            import viser.transforms as vtf

        absolute_path = Path(__file__).parents[2]
        self.graspnet_ckpt = str(absolute_path / graspnet_ckpt)

        # Create all necessary global variables
        self.grasps, self.grasps_dict, self.lerf_scores, self.geom_scores, self.overall_scores, self.pick_object_pt, self.place_point, self.grasp_point, self.fin_grasp = None, {}, [], [], [], None, None, None, None
        self.lerf_points_o3d, self.lerf_relevancy = None, None
        if config_path is not None:
            self.ns_wrapper, self.world_pointcloud, self.global_pointcloud, self.table_center, self.grasps, self.overall_scores = instantiate_scene_from_model(
                server,
                self.graspnet_ckpt,
                config_path=config_path,
                scene_name=config_path.split('/')[-4],
                floor_height=UR5GraspPlanner.FLOOR_HEIGHT
            )
        else:
            self.ns_wrapper, self.world_pointcloud, self.global_pointcloud, self.table_center, self.grasps, self.overall_scores = None, None, None, None, None, None

        """
        All the functions that require the LERF load/train/save -- includes:
        - Load/save directory, assumes
            - output/NAME for LERF capture dataset
            - outputs/NAME for LERF training
            - the latest written config.yml for LERF loading
        
        Note that `lerf_dataset_path` affects the LERF capture capability.
        """
        with server.gui.add_folder("LERF load/train"):
            self.lerf_dataset_path = server.gui.add_text(
                label="Dataset path",
                initial_value="",
            )
            self.lerf_train_button = server.gui.add_button(
                label="Train/Load LERF",
            )
            self.lerf_reset_button = server.gui.add_button(
                label="Reset LERF",
            )
            if config_path is not None:
                self.lerf_dataset_path.disabled = True
                self.lerf_train_button.disabled = True

        self.lerf_train_button.on_click(lambda _: self.lerf_train_button_callback())
        self.lerf_reset_button.on_click(lambda _: self.lerf_reset_button_callback())

        """
        All the functions that affect grasp generation and scores -- includes:
        - Grasp generation (for geometric scores)
        - LERF query (for semantic scores)
        - Hemisphere parameters (for generating grasps)
        - Update overall scores
        - Reset grasps

        This also includes the capability to select the best set of grasps,
        given the semantic scores and/or the geometric scores.
        """
        # Actually, prob generate the grasps when the scene is loaded.
        with server.gui.add_folder("Grasp generation"):
            with server.gui.add_folder("LERF"):
                self.gen_grasp_text = server.gui.add_text(
                    label="LERF query",
                    initial_value="",
                )
                self.gen_grasp_button = server.gui.add_button(
                    label="Calculate LERF query",
                )
                self.primative = server.gui.add_dropdown(
                    label="Primative",
                    options = ["grasp",
                            "pick & place",
                            "twist",
                            "pour"],
                    initial_value = "grasp"
                )
            with server.gui.add_folder("Grasp scores"):
                self.update_overall_scores_button = server.gui.add_button(
                    label=f"Update overall scores",
                )
                self.update_overall_scores_threshold = server.gui.add_slider(
                    label=f"Quantile threshold",
                    min=0.0,
                    max=1.0,
                    initial_value=0.95,
                    step=0.01,
                )
                self.update_overall_scores_slider = server.gui.add_slider(
                    label=f"LERF score weight",
                    min=0.0,
                    max=1.0,
                    initial_value=0.95,
                    step=0.01,
                )

        self.gen_grasp_button.on_click(lambda _: self.gen_grasp_button_callback())
        self.update_overall_scores_button.on_click(lambda _: self.update_overall_scores_button_callback())

    def spin(self):
        while True:
            time.sleep(0.1)

    def lerf_reset_button_callback(self):
        if self.ns_wrapper is None:
            return
        # this doesn't actually seem to work, need to figure out why...
        del self.ns_wrapper.pipeline
        self.world_pointcloud, self.global_pointcloud, self.table_center, self.grasps, self.overall_scores = None, None, None, None, None
        torch.cuda.empty_cache()


    def lerf_train_button_callback(self, neo_data_root: str = 'data', lerf_dataset_path: str = None):

        if lerf_dataset_path is not None:
            self.lerf_dataset_path.value = lerf_dataset_path

        data = Path(f'{neo_data_root}/processed/{self.lerf_dataset_path.value}')
        if not osp.exists(data):
            print(f"data file \"{data}\" doesn't exist; can't load/train; return")
            return
        
        self.lerf_train_button.disabled = True
        import nerfstudio.configs.method_configs as method_configs

        num_steps = 3000
        config: TrainerConfig = method_configs.all_methods['lerf-lite']
        config.pipeline.datamanager.data = data
        config.max_num_iterations = num_steps+1
        config.steps_per_save = num_steps
        config.viewer.quit_on_train_completion = True
        config.output_dir = str(data.parents[1]).replace(neo_data_root+'/processed', 'outputs')
        config.experiment_name = f"{data.parent.name}/before"

        if osp.exists(config.get_base_dir().parent):
            print("PATH EXISTS")
            timestamp_pattern = r'\d{4}-\d{2}-\d{2}_\d{6}'
            timestamps = [d for d in os.listdir(config.get_base_dir().parent) if re.match(timestamp_pattern, d)]
            if timestamps:
                self.timestamp = max(timestamps)
            else:
                self.timestamp = time.strftime('%Y-%m-%d_%H%M%S')
        
        else:
            raise FileNotFoundError(f"Directory {config.get_base_dir().parent} does not exist.")

        # CHECK FOR presence of reconfiguration file
        reconfig_file = Path(f"{str(data.parents[1])}/{data.parent.name}/object_reconfiguration.json")
        print(f"RECONFIG FILE: {reconfig_file}")
        if reconfig_file.exists():
            print("Reconfig file exists!! Applying offset position and offset euler")
            with reconfig_file.open() as f:
                recon_dict = json.load(f)
                self.gt_offset_pos = recon_dict["translation"]
                self.gt_offset_euler = recon_dict["rotation_euler_deg"]
                self.reconfigure = True

        config.timestamp = self.timestamp

        # Neo dataset settings
        config.pipeline.datamanager.dataparser.orientation_method = "none"
        config.pipeline.datamanager.dataparser.center_method = "none"
        config.pipeline.datamanager.dataparser.auto_scale_poses = False
        config.pipeline.datamanager.dataparser.train_split_fraction = 0.99

        if (
            osp.exists(config.get_base_dir()) and
            osp.exists(config.get_base_dir() / "nerfstudio_models")
        ):
            print("we are going to load a model")
            config_path = config.get_base_dir() / "config.yml"
            pipeline = None

        else:
            print("we are going to train a model")
            os.makedirs(config.get_base_dir(), exist_ok=True)
            config.save_config()

            trainer = config.setup(local_rank=0, world_size=1)
            trainer.setup()

            start = time.time()
            trainer.train()
            print(f"Training took {time.time() - start} seconds")
            pipeline = trainer.pipeline
            config_path = None
            pipeline.datamanager.config.patch_size = 1
            pipeline.datamanager.train_pixel_sampler = pipeline.datamanager._get_pixel_sampler(
                pipeline.datamanager.train_dataset, 
                pipeline.datamanager.config.train_num_rays_per_batch
                )

        self.ns_wrapper, self.world_pointcloud, self.global_pointcloud, self.table_center, self.grasps, self.overall_scores = instantiate_scene_from_model(
            self.server,
            self.graspnet_ckpt,
            floor_height=UR5GraspPlanner.FLOOR_HEIGHT,
            config_path=config_path,
            pipeline=pipeline,
            scene_name=self.lerf_dataset_path.value
        )
        self.lerf_train_button.disabled = False

    def gen_grasp_button_callback(self, neo_data_root: str = 'data', gen_grasp_text: str = None, model_types: List[str] = ["neo"]):
        """
        Updates LERF scores by updating the LERF query
        """
        if gen_grasp_text is not None:
            self.gen_grasp_text.value = f"{gen_grasp_text};{gen_grasp_text}"

        self.gen_grasp_text.disabled = True
        self.gen_grasp_button.disabled = True

        lerf_weight = self.update_overall_scores_slider.value
        lerf_word = self.gen_grasp_text.value.split(";")
        if self.primative.value == "pick & place" or self.primative.value == "pour":
            if len(lerf_word) != 3:
                print("Please enter a valid LERF query! Expects three words")
                self.gen_grasp_text.disabled = False
                self.gen_grasp_button.disabled = False
                return
        else:
            if len(lerf_word) != 2:
                print("Please enter a valid LERF query! Expects two words")
                self.gen_grasp_text.disabled = False
                self.gen_grasp_button.disabled = False
                return

        # Get the LERF activation pointcloud for the given query
        self.ns_wrapper.pipeline.image_encoder.set_positives(lerf_word)
        lerf_points_o3d, oryg_lerf_relevancy, dino_pcd, pick_object_pt, place_point, pcd_0_points_o3d, relevancies_0, dinos, camera = get_relevancy_pointcloud(self.ns_wrapper, table_center=self.table_center, server=self.server)

        c2w = camera.camera_to_worlds.cpu().numpy()[0]
        R = tf.SO3.from_matrix(c2w[:3, :3])
        R = R @ tf.SO3.from_x_radians(np.pi)
        self.server.scene.add_camera_frustum(
            name=f"/cameras/camera_center",
            fov=float(2 * np.arctan((camera.cx / camera.fx[0]).cpu())),
            scale=0.25,
            aspect=float((camera.cx[0] / camera.cy[0]).cpu()),
            wxyz=R.wxyz,
            position=c2w[:3, 3],
        )
        
        # Visualize the relevancy pointcloud 
        colors = copy.deepcopy(oryg_lerf_relevancy.squeeze())
        colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-6)
        colors = matplotlib.colormaps['jet'](colors)[:, :3]

        colors_0 = relevancies_0.squeeze()
        colors_0 = (colors_0 - colors_0.min()) / (colors_0.max() - colors_0.min() + 1e-6)
        colors_0 = matplotlib.colormaps['jet'](colors_0)[:, :3]

        lerf_points_o3d = np.asarray(lerf_points_o3d)
        pcd_0_points_o3d = np.asarray(pcd_0_points_o3d)

        if lerf_weight > 0:
            self.server.scene.add_point_cloud(
                name=f"/rel_1_pcd",
                points=lerf_points_o3d,
                colors=colors,
                point_size=0.007,
                visible=False
            )
            self.server.scene.add_point_cloud(
                name=f"/rel_0_pcd",
                points=pcd_0_points_o3d,
                colors=colors_0,
                point_size=0.007,
                visible=False
            )

        # Filter the points based on the LERF relevancy
        lerf_relevancy = copy.deepcopy(oryg_lerf_relevancy)
        rel_mask = (np.where(lerf_relevancy >= 0.5, 1, 0).squeeze() == 1)
        lerf_points_o3d_reduced = lerf_points_o3d[rel_mask]
        lerf_relevancy = lerf_relevancy[rel_mask]
        colors = colors[rel_mask]
        dinos = dinos[rel_mask]

        # Filter the points based on the DBSCAN clustering
        db = DBSCAN(eps=0.02, min_samples=100).fit(lerf_points_o3d_reduced)
        labels = db.labels_
        # num_groups = len(np.unique(labels)) - 1
        num_groups = len(np.unique(labels))
        print(f"NUM GROUPS: {num_groups}")
        bboxes = {"bboxes":[]}
        if num_groups > 0:
            # Go through and filter field by median relevancy
            # Keep only the highest relevancy
            best_field = None
            best_dino = None
            best_colors = None
            max_median_relevancy = 0
            for i in range(num_groups):
                print(f"GROUP ID: {i}")
                mask = np.where(labels == i, True, False)
                dbscan_field = lerf_points_o3d_reduced[mask]
                dbscan_dino = dinos[mask]
                dbscan_relevancy = lerf_relevancy[mask]
                dbscan_colors = colors[mask]
                if np.sum(dbscan_relevancy) == 0:
                    continue
                median_rel = np.median(dbscan_relevancy)
                if median_rel > max_median_relevancy:
                    best_field = copy.deepcopy(dbscan_field)
                    best_dino = copy.deepcopy(dbscan_dino)
                    best_colors = copy.deepcopy(dbscan_colors)
                    max_median_relevancy = median_rel

            print("_______________________________________________________________________________________________________________________")
            print("Group")
            median_dino = np.median(best_dino, axis=0)

            a_norm = np.linalg.norm(best_dino, axis=1)
            b_norm = np.linalg.norm(median_dino)[None, ...]

            cos_sim = (best_dino @ median_dino) / (a_norm * b_norm)

            print("Max:", np.max(cos_sim))
            print("Mean:", np.mean(cos_sim))
            print("Median:", np.median(cos_sim))
            print("Min:", np.min(cos_sim))

            best_field = best_field[cos_sim > 0.5]
            best_colors = best_colors[cos_sim > 0.5]

            print("_______________________________________________________________________________________________________________________")

            # Use Open3D's statistical outlier removal
            best_field_o3d = o3d.geometry.PointCloud()
            best_field_o3d.points = o3d.utility.Vector3dVector(best_field)
            best_field_o3d.colors = o3d.utility.Vector3dVector(best_colors)

            _, ind = best_field_o3d.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            best_field = best_field[ind]
            best_colors = best_colors[ind]

            group = copy.deepcopy(best_field.T)
            means = np.mean(group, axis=1)
            group -= means[:, np.newaxis]
            cov = np.cov(group)
            _, evec = np.linalg.eig(cov)
            evec, _ = np.linalg.qr(evec)

            # Align the points to the XY plane
            theta = np.arctan2(evec[1, 0], evec[0, 0])
            evec = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])

            aligned_coords = evec.T @ group

            min_x, min_y, min_z = np.min(aligned_coords.T, axis=0).astype(float)
            max_x, max_y, max_z = np.max(aligned_coords.T, axis=0).astype(float)

            print("Min:", min_x, min_y, min_z)
            print("Max:", max_x, max_y, max_z)

            rectCoords = lambda x1, y1, z1, x2, y2, z2: np.array([[x1, x1, x2, x2, x1, x1, x2, x2],
                                                                    [y1, y2, y2, y1, y1, y2, y2, y1],
                                                                    [z1, z1, z1, z1, z2, z2, z2, z2]])

            realigned_coords = evec @ aligned_coords
            realigned_coords += means[:, np.newaxis]

            rrc = evec @ rectCoords(min_x, min_y, min_z, max_x, max_y, max_z)
            rrc += means[:, np.newaxis]

            OFFSET = 0.01
            means = np.mean(rrc, axis=1)
            dimensions = [round(abs(max_x - min_x) + OFFSET, 2), round(abs(max_y - min_y) + OFFSET, 2), round(abs(max_z - min_z) + (OFFSET / 2), 2)]
            R = Rotation.from_matrix(evec)
            euler = list(R.as_euler('xyz', False))
            R_2 = Rotation.from_euler('xyz', [0.0, 0.0, euler[2]], degrees=False)
            euler = [0.0, 0.0, -euler[2]]
            quaternion = list(R_2.as_quat(False, scalar_first=True))
            position = list(means.astype(float))
            position[2] += OFFSET / 2

            # # Clip Z dimension to 0.0
            # if position[2] - (dimensions[2] / 2) < 0.0:
            #     print(f"Clipping Z dimension: {position[2] - (dimensions[2] / 2)}")
            #     position[2] += abs(position[2] - (dimensions[2] / 2)) / 2
            #     dimensions[2] -= abs(position[2] - (dimensions[2] / 2)) / 2

            # Extend bottom face of the bounding box to always touch z=-0.005 (i.e., -0.5 cm)
            dimensions[0] += 0.02
            dimensions[1] += 0.02
            target_z = 0.002
            bottom_z = position[2] - (dimensions[2] / 2)
            if bottom_z != target_z:
                shift = target_z - bottom_z
                position[2] += shift / 2
                dimensions[2] += round(abs(shift), 2)
                print(f"Adjusted bbox so bottom face touches z={target_z}. New position[2]: {position[2]}, new dimensions[2]: {dimensions[2]}")

            print("Dim:", dimensions)
            print("Eul:", euler)
            print("Quat:", quaternion)
            print("Pos:", position)

            # NOTE as we only save one, idx is 0 here. Should be tidied
            bbox_data = {"idx": 0, 
                        "data": {"dimensions": dimensions, 
                                "euler": euler, 
                                "quaternion": quaternion, 
                                "position": position, 
                                "is_reconfigured": self.reconfigure,
                                "offset_position": self.gt_offset_pos, 
                                "offset_euler": self.gt_offset_euler
                                }}
            bboxes["bboxes"].append(bbox_data)

        # Save bbox.json file
        data = Path(f'{neo_data_root}/processed/{self.lerf_dataset_path.value}')
        if not osp.exists(data):
            print(f"data file \"{data}\" doesn't exist; can't load/train; return")
            return

        for model_type in model_types:
            print("----------------------------------------------------------------------------------------------------------------------------")
            box_dir = Path(f"{str(data.parents[1]).replace(neo_data_root+'/processed', 'outputs')}/{data.parent.name}/before/{model_type}/{self.timestamp}")
            print(f"OUTPUT FOLDER: {box_dir}")
            print("----------------------------------------------------------------------------------------------------------------------------")

            box_dir.mkdir(parents=True, exist_ok=True)
            box_path = box_dir / "bbox.json"
            print(f"BBOX JSON FILE: {str(box_path)}")
            with open(str(box_path), 'w') as f:
                json.dump(bboxes, f, indent=2)

        self.server.scene.add_point_cloud(
                name=f"/rel_1_pcd_reduced",
                points=best_field,
                colors=best_colors,
                point_size=0.007
            )
        
        self.server.scene.add_box(
            name=f"/lerf/bbox",
            dimensions=np.array(dimensions),
            wxyz=np.array(quaternion),
            position=np.array(position),
            color=[255, 0, 0]
        )
        
        self.lerf_scores = []
        self.geom_scores = []

        for i, grasp in enumerate(self.grasps):
            box = get_bbox_from_grasp(grasp)
            # get the indices of the lerf_xyz pointcloud which lie inside the box and avg
            pts = box.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(lerf_points_o3d))
            pts1 = box.get_point_indices_within_bounding_box(dino_pcd)
            if len(pts) == 0:
                score = 0.0
            else:
                # score = lerf_relevancy[pts].mean()
                score = np.median(oryg_lerf_relevancy[pts].squeeze()).item()
            score1 = grasp.score
            if len(pts1) == 0:
                score1 = 0.0
            
            self.lerf_scores.append(score)
            self.geom_scores.append(score1)

        #All visualization stuff
        self.lerf_scores = np.array(self.lerf_scores)
        self.lerf_scores /= oryg_lerf_relevancy.max()

        self.gen_grasp_text.disabled = False
        self.gen_grasp_button.disabled = False

    def update_overall_scores_button_callback(self):
        """
        Update the overall scores by updating the LERF score weight
        """
        lerf_weight = self.update_overall_scores_slider.value
        geom_weight = 1.0 - lerf_weight
        # Update the scores...
        if self.lerf_scores is None or len(self.lerf_scores) == 0:
            scores = self.geom_scores
        elif lerf_weight == 1.0:
            scores = self.lerf_scores
        else:
            scores = (lerf_weight) * np.array(self.lerf_scores) + (geom_weight) * np.array(self.geom_scores)
        if len(scores) == 0:
            print("Need to run LERF query first!")
            return

        scores_threshold = np.quantile(scores, self.update_overall_scores_threshold.value)
        grasps_selected = [grasp for (ind, grasp) in enumerate(self.grasps) if scores[ind] > scores_threshold]
        inds_selected = [ind for (ind, grasp) in enumerate(self.grasps) if scores[ind] > scores_threshold]

        grasps_selected = GraspGroup(np.stack([grasp.grasp_array for grasp in grasps_selected]))
        # grasps_selected = grasps_selected.nms(translation_thresh=0.02, rotation_thresh=30.0/180.0*np.pi)
        grasps_selected = grasps_selected.sort_by_score()

        for ind, grasp_list in self.grasps_dict.items():
            for grasp in grasp_list:
                grasp.remove()
        self.grasps_dict = {}

        scores -= scores[inds_selected].min()
        scores /= scores[inds_selected].max() + 1e-9

        for ind, grasp in zip(inds_selected, grasps_selected):
            self.grasps_dict[ind] = add_grasps(self.server, grasp, ind, scores[ind])

        self.overall_scores = scores


def main():
    gen_grasp = GenGrasp()
    gen_grasp.populate()
    gen_grasp.spin()
    
if __name__ == "__main__":
    tyro.cli(main)