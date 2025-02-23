import cv2
import torch
import numpy as np
import numpy.typing as npt

from typing import List
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.base_model import Model


def render_preview_image(model: Model, cameras: Cameras, image_idxs: torch.Tensor, output_name: str = 'rgb_rm_preview') -> npt.NDArray:

    with torch.no_grad():
        outputs = model.get_outputs_for_camera(cameras[image_idxs])

    img_shape = outputs['rgb'].shape
    rendered_images = (outputs[output_name] * 255).reshape(img_shape[0], img_shape[1], -1, 3).permute(2, 0, 1, 3)
    rendered_images = rendered_images[:, :, :, [2, 1, 0]]

    return rendered_images