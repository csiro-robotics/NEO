import cv2
import json
import numpy as np
import tyro

from copy import deepcopy
from scipy.optimize import minimize

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE

from scripts.experiment_config import NeoDataConfig

MARKER_LENGTH = 0.1065

def get_marker_pose(corners, camera_matrix, dist_coeffs, marker_length=0.1):
    """ Compute the pose of a marker given its corners. """
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

def get_arucos(images, transforms_dict, marker_length=0.1):
    # Define ArUco dictionary and parameters
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    arucos = []
    camera_matrix = np.array([[transforms_dict["fl_x"], 0, transforms_dict["cx"]], 
                              [0, transforms_dict["fl_y"], transforms_dict["cy"]], 
                              [0, 0, 1]])
    dist_coeffs = np.array([transforms_dict["k1"], transforms_dict["k2"], transforms_dict["p1"], transforms_dict["p2"]])
    for image, frame in zip(images, transforms_dict["frames"]):
        corners, ids, _ = detector.detectMarkers(image)
        for i in range(len(corners)):
            # Compute poses
            rvec, tvec = get_marker_pose(corners[i], camera_matrix, dist_coeffs, marker_length=marker_length)

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

def aruco_objective(scale, arucos = None, rand_0 = None, rand_1 = None, rand_2 = None, rand_3 = None):

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

    sum_0 = 0
    for i in range(len(aruco_dict[0]) - 1):
        sum_0 += np.linalg.norm(marker_poses_0[rand_0[i + 1], :3, 3] - marker_poses_0[rand_0[i], :3, 3])

    sum_1 = 0
    for i in range(len(aruco_dict[1]) - 1):
        sum_1 += np.linalg.norm(marker_poses_1[rand_1[i + 1], :3, 3] - marker_poses_1[rand_1[i], :3, 3])

    sum_2 = 0
    for i in range(len(aruco_dict[2]) - 1):
        sum_2 += np.linalg.norm(marker_poses_2[rand_2[i + 1], :3, 3] - marker_poses_2[rand_2[i], :3, 3])

    sum_3 = 0
    for i in range(len(aruco_dict[3]) - 1):
        sum_3 += np.linalg.norm(marker_poses_3[rand_3[i + 1], :3, 3] - marker_poses_3[rand_3[i], :3, 3])
    
    return sum_0 + sum_1 + sum_2 + sum_3

def apply_scale(arucos, best_scale, transforms_dict):
    # Apply best scale and create aruco dictionsary
    aruco_dict_before = {}
    for i in range(100):
        aruco_dict_before[i] = []

    # Calculate mean of aruco positions
    aruco_t = []
    for aruco_id in range(4):
        aruco_ts = []
        for trans_mat, trans, idx in arucos:
            temp = deepcopy(trans_mat)
            temp[:3, 3:] = temp[:3, 3:] / best_scale
            marker_pose = temp @ trans
            if idx == aruco_id:
                aruco_ts.append(deepcopy(marker_pose[:3, 3:]))

        aruco_ts = np.array(aruco_ts)
        aruco_ts = reject_outliers_iqr(aruco_ts[:, :, 0])
        aruco_t.append(np.mean(aruco_ts, axis=0))

    # Calculate world axes according to nieghbouring arucos 1 and 2
    points = np.array(aruco_t)

    aruco_1 = points[1] - points[0]
    aruco_2 = points[2] - points[0]

    gamma = np.arctan2(-aruco_1[1], aruco_1[0])

    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                [np.sin(gamma), np.cos(gamma), 0],
                [0, 0, 1]])

    aruco_1 = Rz @ aruco_1
    aruco_2 = Rz @ aruco_2

    beta = np.arctan2(aruco_1[2], aruco_1[0])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                [0, 1, 0],
                [-np.sin(beta), 0, np.cos(beta)]])

    aruco_1 = Ry @ aruco_1
    aruco_2 = Ry @ aruco_2

    alpha = np.arctan2(-aruco_2[2], aruco_2[1])
    Rx = np.array([[1, 0, 0],
                [0, np.cos(alpha), -np.sin(alpha)],
                [0, np.sin(alpha), np.cos(alpha)]])

    aruco_1 = Rx @ aruco_1
    aruco_2 = Rx @ aruco_2

    world_to_aruco_R = (Rx @ Ry @ Rz).T

    # Calculate mean of aruco positions
    world_to_aruco_ts = []
    for trans_mat, trans, idx in arucos:
        temp = deepcopy(trans_mat)
        temp[:3, 3:] = temp[:3, 3:] / best_scale
        marker_pose = temp @ trans
        if idx == 0:
            marker_pose[:3, 3:] = world_to_aruco_R.T @ marker_pose[:3, 3:]
            world_to_aruco_ts.append(deepcopy(marker_pose[:3, 3:]))

    world_to_aruco_ts = np.array(world_to_aruco_ts)
    world_to_aruco_ts = reject_outliers_iqr(world_to_aruco_ts[:, :, 0])
    world_to_aruco_t = np.mean(world_to_aruco_ts, axis=0)[..., None]

    for trans_mat, trans, idx in arucos:
        temp = deepcopy(trans_mat)
        temp[:3, 3:] = temp[:3, 3:] / best_scale
        marker_pose = temp @ trans

        marker_pose[:3, 3:] = world_to_aruco_R.T @ marker_pose[:3, 3:]
        marker_pose[:3, 3:] = marker_pose[:3, 3:] - world_to_aruco_t

        aruco_dict_before[idx].append(marker_pose)

    # Convert transform.json file to meters
    for frame in transforms_dict["frames"]:
        transform_matrix = np.array(frame["transform_matrix"])
        transform_matrix[:3, :3] = world_to_aruco_R.T @ transform_matrix[:3, :3]
        transform_matrix[:3, 3:] = transform_matrix[:3, 3:] / best_scale
        transform_matrix[:3, 3:] = (world_to_aruco_R.T @ transform_matrix[:3, 3:]) - world_to_aruco_t
        frame["transform_matrix"] = transform_matrix.tolist()

def reject_outliers_iqr(points, multiplier=1.5):
    """
    Rejects outliers based on the IQR (Interquartile Range) method using NumPy.
    
    Parameters:
    - points: numpy array of shape (n_points, 3) representing 3D points
    - multiplier: the factor to multiply the IQR by to define the outlier range (default is 1.5)
    
    Returns:
    - filtered_points: numpy array with outliers removed
    """
    # Compute the 1st (Q1) and 3rd (Q3) quartiles for each dimension (x, y, z)
    Q1 = np.percentile(points, 25, axis=0)
    Q3 = np.percentile(points, 75, axis=0)
    
    # Compute the Interquartile Range (IQR)
    IQR = Q3 - Q1
    
    # Calculate lower and upper bounds for each dimension
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # Find the points that lie within the bounds for each dimension
    mask = np.all((points >= lower_bound) & (points <= upper_bound), axis=1)
    
    # Return the filtered points (outliers removed)
    return points[mask, :]

def match_arucos(config: NeoDataConfig, skip_after: bool = False) -> None:
    """Main function."""
    
    CONSOLE.print("[bold green]Matching Arucos[/bold green]")
    output_path = config.get_base_dir()

    # Data structures
    transform_dicts = {}
    arucos = {}
    best_scale = {}

    for stage in ["before", "after"]:

        if stage == "after" and skip_after:
            continue

        # Load transformation
        CONSOLE.print("Loading transformation files")
        with open(output_path / stage / 'raw_transforms.json', 'r') as file:
            transform_dicts[stage] = json.load(file)

        # TODO for some reason reading from file is better? TO BE CONFIRMED
        frames = {stage: []}
        for frame in transform_dicts[stage]["frames"]:
            frames[stage].append(cv2.imread(str(output_path / stage / frame["file_path"])))

        # Detect Aruco markers
        CONSOLE.print("Detecting arucos")
        arucos[stage] = get_arucos(frames[stage], transform_dicts[stage], marker_length=MARKER_LENGTH)

        # Optimize for best scale
        CONSOLE.print("Optimizing scale")
        scale_min, scale_max = 0, 20
        n_neighbours = 50

        aruco_dict = {}
        for i in range(100):
            aruco_dict[i] = []

        for _, _, idx in arucos[stage]:
            aruco_dict[idx].append(idx)

        rand_0 = np.random.randint(0, len(aruco_dict[0]), size=(len(aruco_dict[0]), n_neighbours))
        rand_1 = np.random.randint(0, len(aruco_dict[1]), size=(len(aruco_dict[1]), n_neighbours))
        rand_2 = np.random.randint(0, len(aruco_dict[2]), size=(len(aruco_dict[2]), n_neighbours))
        rand_3 = np.random.randint(0, len(aruco_dict[3]), size=(len(aruco_dict[3]), n_neighbours))

        rand_start = scale_min + np.random.rand(1) * (scale_max - scale_min)
        best_scale[stage] = minimize(aruco_objective, rand_start, method='nelder-mead', options={'maxiter': 1000}, args=(arucos[stage], rand_0, rand_1, rand_2, rand_3))['x']

        CONSOLE.print(f"Best scales {stage}: {best_scale[stage]}")

        # Apply that scale to the transforms
        CONSOLE.print("Applying scale")
        apply_scale(arucos[stage], best_scale[stage], transform_dicts[stage])

        # Save the updated transforms file
        with open(output_path / stage / 'transforms.json', 'w') as file:
            json.dump(transform_dicts[stage], file, indent=4)

        CONSOLE.print("[bold green]Arucos matched succesfully![/bold green]")

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")

    match_arucos(
        tyro.cli(
            NeoDataConfig,
            description=convert_markup_to_ansi(__doc__),
        )
    )

if __name__ == "__main__":
    entrypoint()