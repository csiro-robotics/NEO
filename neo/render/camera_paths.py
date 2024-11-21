from typing import Any, Dict

import torch

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
