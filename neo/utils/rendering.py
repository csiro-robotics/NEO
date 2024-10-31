import torch
import numpy as np
import numpy.typing as npt

from typing import List
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.base_model import Model


def render_preview_image(model: Model, cameras: Cameras, image_idx: int, output_name: str = 'rgb_rm_preview') -> npt.NDArray:

    with torch.no_grad():
        outputs = model.get_outputs_for_camera(cameras[image_idx])

    rgb_preview = (outputs[output_name].cpu().detach().numpy() * 255).astype(np.uint8)
    
    return rgb_preview