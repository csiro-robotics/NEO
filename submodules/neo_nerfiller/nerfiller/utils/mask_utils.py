"""
Code for generating masks to inpaint.
"""

import torch

from nerfiller.utils.mesh_utils import (
    dilate,
    get_mesh_from_perspective_images
)
from nerfiller.utils.typing import *
from nerfstudio.cameras.camera_utils import viewmatrix


def random_mask_custom(im_shape, ratio=1, mask_full_image=False, max_np=10, min_np=2):
    num_points = int(torch.rand(1) * (max_np - min_np)) + min_np
    size = torch.randint(low=0, high=im_shape[0] // 4, size=(num_points,))
    y = torch.randint(low=0, high=im_shape[0], size=(num_points,))
    x = torch.randint(low=0, high=im_shape[0], size=(num_points,))
    mask = torch.zeros(im_shape)
    for i in range(num_points):
        m = torch.zeros(im_shape)
        m[y[i], x[i]] = 1.0
        for i in range(size[i]):
            m = dilate(m[None, None])[0, 0]
        mask = (mask + m).clamp(0, 1)
    return mask


def downscale_mask(mask, size=None, scale_factor=None, dilate_iters=0, dilate_kernel_size=3):
    """
    Downscale the mask in a conservative way. 1s are where to inpaint, 0 where to not inpaint.
    Inpaints extra pixels to prevent leakage under the mask.
    """
    assert size or scale_factor
    if size:
        assert scale_factor is None
    if scale_factor:
        assert size is None
    for _ in range(dilate_iters):
        mask = dilate(mask, kernel_size=dilate_kernel_size)
    mask = torch.nn.functional.interpolate(mask, size=size, scale_factor=scale_factor, mode="bilinear")
    mask = (mask != 0.0).float()  # expands the mask slightly for no leakage of pixels
    return mask
