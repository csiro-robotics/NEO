import pyrealsense2 as rs
import numpy as np
import cv2
import os
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
import subprocess
import json
from copy import deepcopy
from scipy.optimize import minimize

def read_realsense_bag(bag_file_path, output_path, rotate_images, images=[], skip_frames=6):
    # Configure the pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable playback of the bag file
    config.enable_device_from_file(bag_file_path, repeat_playback=False)

    # Start the pipeline
    pipeline.start(config)

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path / 'rgb', exist_ok=True)
    os.makedirs(output_path / 'depth', exist_ok=True)

    i = 0
    j = 1
    try:
        while True:
            # Wait for a frame from the pipeline
            try:
                frames = pipeline.wait_for_frames()
            except:
                break

            # Get color and depth frames
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                break

            # Convert images to numpy arrays
            color_image = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)
            depth_image = np.asanyarray(depth_frame.get_data())

            if i % skip_frames == 0:
                print(i)

                num = str(j)
                while len(num) < 4:
                    num = '0' + num

                if rotate_images:
                    color_image = cv2.rotate(color_image, cv2.ROTATE_180)
                    depth_image = cv2.rotate(depth_image, cv2.ROTATE_180)

                # Display the images
                cv2.imwrite(str(output_path / f'rgb/{num}.png'), color_image)
                cv2.imwrite(str(output_path / f'depth/{num}.png'), depth_image)
                images.append(color_image)
                j += 1

            i += 1
    finally:
        # Stop the pipeline
        pipeline.stop()


def get_marker_pose(corners, camera_matrix, dist_coeffs):
    """ Compute the pose of a marker given its corners. """
    # Define the length of the marker's side (in meters or other units)
    marker_length = 0.1  # Adjust based on your marker size
    # Create an object point array (3D points in the marker's coordinate system)
    obj_points = np.array([
        [-marker_length / 2, marker_length / 2, 0],
        [marker_length / 2, marker_length / 2, 0],
        [marker_length / 2, -marker_length / 2, 0],
        [-marker_length / 2, -marker_length / 2, 0]
    ], dtype=np.float32)

    # Solve the PnP problem to get rotation and translation vectors
    _, rvec, tvec = cv2.solvePnP(obj_points, corners, camera_matrix, dist_coeffs, False, cv2.SOLVEPNP_IPPE_SQUARE)
    return rvec, tvec

def get_arucos(images, transforms_dict):
    # Define ArUco dictionary and parameters
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    arucos = []
    camera_matrix = np.array([[transforms_dict["fl_x"], 0, transforms_dict["cx"]], 
                        [0, transforms_dict["fl_y"], transforms_dict["cy"]], 
                        [0, 0, 1]])
    dist_coeffs = np.zeros((4, 1))
    for image, frame in zip(images, transforms_dict["frames"]):
        corners, ids, _ = detector.detectMarkers(image)
        for i in range(len(corners)):
            # Compute poses
            rvec, tvec = get_marker_pose(corners[i], camera_matrix, dist_coeffs)

            R, _ = cv2.Rodrigues(rvec)
            
            trans = np.eye(4)
            trans[:3, :3] = R
            trans[:3, 3:] = tvec

            # Convert rotation vectors to matrices
            transform_matrix = np.array(frame["transform_matrix"])

            transform = np.array([[1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])

            marker_pose = (transform_matrix, transform @ trans, ids[i][0])
            arucos.append(marker_pose)
    return arucos

def aruco_objective(scale, arucos):
    aruco_dict = {}
    for i in range(100):
        aruco_dict[i] = []

    for trans_mat, trans, idx in arucos:
        temp = deepcopy(trans_mat)
        temp[:3, 3:] = temp[:3, 3:] / scale[0]
        aruco_dict[idx].append(temp @ trans)

    marker_poses_0 = np.array(aruco_dict[0])
    marker_poses_1 = np.array(aruco_dict[1])
    marker_poses_2 = np.array(aruco_dict[2])
    marker_poses_3 = np.array(aruco_dict[3])

    sum_0 = np.abs(np.sum(np.diff(marker_poses_0[:, :3, 3], axis=0)))
    sum_1 = np.abs(np.sum(np.diff(marker_poses_1[:, :3, 3], axis=0)))
    sum_2 = np.abs(np.sum(np.diff(marker_poses_2[:, :3, 3], axis=0)))
    sum_3 = np.abs(np.sum(np.diff(marker_poses_3[:, :3, 3], axis=0)))
    
    return sum_0 + sum_1 + sum_2 + sum_3

def apply_scale(arucos, best_scale,transforms_dict):
# Apply best scale and create aruco dictionsary
    aruco_dict_before = {}
    for i in range(100):
        aruco_dict_before[i] = []

    camera_path_before = []
    i = 0

    world_to_aruco_R = None
    world_to_aruco_t = None

    # Setup pose of 0 aruco marker as world center
    for trans_mat, trans, idx in arucos:
        temp = deepcopy(trans_mat)
        temp[:3, 3:] = temp[:3, 3:] / best_scale
        marker_pose = temp @ trans
        if idx == 0:
            world_to_aruco_R = deepcopy(marker_pose[:3, :3])
            marker_pose[:3, 3:] = world_to_aruco_R.T @ marker_pose[:3, 3:]
            world_to_aruco_t = deepcopy(marker_pose[:3, 3:])
            break

    for trans_mat, trans, idx in arucos:
        temp = deepcopy(trans_mat)
        temp[:3, 3:] = temp[:3, 3:] / best_scale
        marker_pose = temp @ trans
        marker_pose[:3, 3:] = world_to_aruco_R.T @ marker_pose[:3, 3:]
        marker_pose[:3, 3:] = marker_pose[:3, 3:] - world_to_aruco_t
        aruco_dict_before[idx].append(marker_pose)

        if idx == 0:
            temp[:3, 3:] = (world_to_aruco_R.T @ temp[:3, 3:]) - world_to_aruco_t
            camera_path_before.append(temp)

        i += 1

    # Convert transform.json file to meters
    for frame in transforms_dict["frames"]:
        transform_matrix = np.array(frame["transform_matrix"])
        transform_matrix[:3, :3] = world_to_aruco_R.T @ transform_matrix[:3, :3]
        transform_matrix[:3, 3:] = transform_matrix[:3, 3:] / best_scale
        transform_matrix[:3, 3:] = (world_to_aruco_R.T @ transform_matrix[:3, 3:]) - world_to_aruco_t
        frame["transform_matrix"] = transform_matrix.tolist()
    

parser = ArgumentParser()
parser.add_argument("--scene_name", "-n", type=str, 
                    help="name of scene. Used in naming all related files")
parser.add_argument("--root_bag", default="data/raw_bagfiles",
                    help="absolute path of root location of bag files")
parser.add_argument("--skip_frames", default=6, type=int, 
                    help="number of frames to be skipped from original video")
parser.add_argument("--action", choices=["remove", "translate", 
                                       "rotate", "translate_and_rotate"])
parser.add_argument("--scene_type", choices=["single", "multiple_one", 
                                             "multiple_sequential", 
                                             "multiple_simultaneous"])
parser.add_argument("--sequence_num", type=int)
parser.add_argument("--root_output", default="data/neo_dataset",
                    help="absolute path of output for dataset")
args=parser.parse_args()


action_type = "remove" if args.action == 'remove' else "reconfigure"
# HACK!!! we ran into issues with COLMAP within the docker. 
# Should find out best practice to avoid this
# subprocess.run(f"chmod -R 777 {args.root_output}", shell=True)
# subprocess.run(f"chmod -R 777 {args.root_bag}", shell=True)
scene_name = args.scene_name if args.scene_type != "multiple_simultaneous" \
    else f"{args.scene_name}-{args.sequence_num:04}"
if args.scene_type == "single":
    output_path = Path(args.root_output) / "real" / args.scene_type / scene_name
else:
    # TODO need to implement this and think about folder structure further
    raise NotImplementedError("Currently only working for single object scenarios")
# Need both before and after to be present in order to perform final alignment
frames ={"before": [], "after": []}
transform_dicts = {}
arucos = {}
best_scale = {}
for stage in ["before", "after"]:
    bag_file_path = os.path.join(args.root_bag, action_type, 
                                f"{scene_name}_{stage}.bag")
    
    print(f"OUTPUT PATH: {output_path}")

    # Extract raw frames from realsense camera
    read_realsense_bag(bag_file_path, output_path /stage, True, frames[stage], args.skip_frames)
    print(f"NUMBER OF FRAMES: {len(frames[stage])}")    
    # Use COLMAP to get initial positions
    subprocess.run(f"ns-process-data images --data {output_path / stage / 'rgb'} --output-dir {output_path / stage}", shell=True)
    subprocess.run(f"rm -rf {output_path / stage /'rgb'}", shell=True)
    subprocess.run(f"cp {output_path / stage / 'transforms.json'} {output_path / stage / 'raw_transforms.json'}", shell=True)

print("LOADING TRANSFORMATION FILES")
# Load transformation
with open(output_path / "before" / 'transforms.json', 'r') as file:
    transform_dicts["before"] = json.load(file)

with open(output_path / "after" / 'transforms.json', 'r') as file:
    transform_dicts["after"] = json.load(file)

# Define camera parameters for world before
print("DETECTING ARUCOS")
# Detect Aruco markers
arucos["before"] = get_arucos(frames["before"], transform_dicts["before"])
arucos["after"] = get_arucos(frames["after"], transform_dicts["after"])

# Optimize for best scale
print("OPTIMIZING SCALE")
scale_min, scale_max = 0, 20
rand_start = scale_min + np.random.rand(1) * (scale_max - scale_min)
best_scale["before"] = minimize(aruco_objective, rand_start, arucos["before"], method='nelder-mead')['x']
rand_start = scale_min + np.random.rand(1) * (scale_max - scale_min)
best_scale["after"] = minimize(aruco_objective, rand_start, arucos["after"], method='nelder-mead')['x']

print("APPLYING SCALE")
# Apply that scale to the transforms
apply_scale(arucos["before"], best_scale["before"], transform_dicts["before"])
apply_scale(arucos["after"], best_scale["after"], transform_dicts["after"])

# Save the updated transforms file
with open(output_path / "before" / 'transforms.json', 'w') as file:
    json.dump(transform_dicts["before"], file, indent=4)
with open(output_path / "after" / 'transforms.json', 'w') as file:
    json.dump(transform_dicts["after"], file, indent=4)

