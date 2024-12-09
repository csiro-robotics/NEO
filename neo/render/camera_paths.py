import torch
import numpy as np
import numpy.typing as npt

from typing import Any, List, Dict
from scipy.spatial.transform import Rotation

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.viewer_legacy.server.utils import three_js_perspective_camera_focal_length


def get_path_from_json(camera_path: Dict[str, Any]) -> Cameras:
    """Takes a camera path dictionary and returns a trajectory as a Camera instance.

    Args:
        camera_path: A dictionary of the camera path information coming from the viewer.

    Returns:
        A Cameras instance with the camera path.
    """

    image_height = camera_path["render_height"]
    image_width = camera_path["render_width"]

    if "camera_type" not in camera_path:
        camera_type = CameraType.PERSPECTIVE
    elif camera_path["camera_type"] == "fisheye":
        camera_type = CameraType.FISHEYE
    elif camera_path["camera_type"] == "equirectangular":
        camera_type = CameraType.EQUIRECTANGULAR
    elif camera_path["camera_type"].lower() == "omnidirectional":
        camera_type = CameraType.OMNIDIRECTIONALSTEREO_L
    elif camera_path["camera_type"].lower() == "vr180":
        camera_type = CameraType.VR180_L
    else:
        camera_type = CameraType.PERSPECTIVE

    c2ws = []
    fxs = []
    fys = []
    for camera in camera_path["camera_path"]:
        # pose
        c2w = torch.tensor(camera["camera_to_world"]).view(4, 4)[:3]
        c2ws.append(c2w)
        if camera_type in [
            CameraType.EQUIRECTANGULAR,
            CameraType.OMNIDIRECTIONALSTEREO_L,
            CameraType.OMNIDIRECTIONALSTEREO_R,
            CameraType.VR180_L,
            CameraType.VR180_R,
        ]:
            fxs.append(image_width / 2)
            fys.append(image_height)
        else:
            # field of view
            fov = camera["fov"]
            focal_length = three_js_perspective_camera_focal_length(fov, image_height)

            if "fl_x" in camera_path:
                fxs.append(camera_path["fl_x"])
            else:
                fxs.append(focal_length)

            if "fl_y" in camera_path:
                fys.append(camera_path["fl_y"])
            else:
                fys.append(focal_length)

    # Iff ALL cameras in the path have a "time" value, construct Cameras with times
    if all("render_time" in camera for camera in camera_path["camera_path"]):
        times = torch.tensor([camera["render_time"] for camera in camera_path["camera_path"]])
    else:
        times = None

    if "cx" not in camera_path:
        cx = image_width / 2
    
    else:
        cx = camera_path["cx"]

    if "cy" not in camera_path:
        cy = image_height / 2
    
    else:
        cy = camera_path["cy"]

    camera_to_worlds = torch.stack(c2ws, dim=0)
    fx = torch.tensor(fxs)
    fy = torch.tensor(fys)
    cx = torch.tensor(cx)
    cy = torch.tensor(cy)
    return Cameras(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=image_width,
        height=image_height,
        camera_to_worlds=camera_to_worlds,
        camera_type=camera_type,
        times=times,
    )


def generate_hemisphere_transforms(radius: float, num_layers: int, origin: np.ndarray, num_per_layer: int = 4, dtype: type = np.array, shape: tuple = (4, 4)) -> List[npt.NDArray]:
    transformation_matrices = []
    
    thetas = np.linspace(0.0, np.pi / 2, num_layers + 2)
    phi_offsets = np.linspace(0.0, np.pi / 2, num_layers, endpoint=False)
    for i, theta in enumerate(thetas[1:-1]):
        for j in range(num_per_layer):
            
            phi = (((2 * np.pi) / num_per_layer) * j) + phi_offsets[i]

            x = (np.cos(theta) * np.sin(-phi) * radius) + origin[0]
            y = (np.cos(theta) * np.cos(-phi) * radius) + origin[1]
            z = (np.sin(theta) * radius) + origin[2]

            rot_1 = Rotation.from_euler('XYZ', [-np.pi / 2, -phi, 0.0], degrees=False)
            rot_2 = Rotation.from_euler('XYZ', [theta, 0.0, 0.0], degrees=False)
            rot_3 = Rotation.from_euler('XYZ', [0.0, 0.0, np.pi], degrees=False)
            R = (rot_1 * rot_2 * rot_3).as_matrix()

            trans_mat = np.eye(4)
            trans_mat[:3, 3:] = np.array([[x, y, z]]).T
            trans_mat[:3, :3] = R

            if dtype == torch.Tensor:
                trans_mat = torch.tensor(trans_mat).float()

            if shape == (3, 4):
                trans_mat = trans_mat[:3, :]

            transformation_matrices.append(trans_mat)
    
    return transformation_matrices