import torch
import math

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal
from jaxtyping import Float, Int, Shaped
from torch import Tensor

import nerfstudio.utils.poses as pose_utils
from nerfstudio.cameras.cameras import CameraType, Cameras
from nerfstudio.cameras import camera_utils

from nerfstudio.cameras.rays import RayBundle


@dataclass(init=False)
class VoxelCameras(Cameras):

    def __init__(self, 
                 camera_to_worlds: Tensor, 
                 fx: Tensor | float, 
                 fy: Tensor | float, 
                 cx: Tensor | float, 
                 cy: Tensor | float, 
                 width: Tensor | int | None = None, 
                 height: Tensor | int | None = None, 
                 distortion_params: Tensor | None = None, 
                 camera_type: Tensor | int | List[CameraType] | CameraType = CameraType.PERSPECTIVE, 
                 times: Tensor | None = None, 
                 metadata: Dict | None = None) -> None:

        super().__init__(camera_to_worlds, fx, fy, cx, cy, width, height, distortion_params, camera_type, times, metadata)

    def _generate_rays_from_coords(
        self,
        camera_indices: Int[Tensor, "*num_rays num_cameras_batch_dims"],
        coords: Float[Tensor, "*num_rays 2"],
        camera_opt_to_camera: Optional[Float[Tensor, "*num_rays 3 4"]] = None,
        distortion_params_delta: Optional[Float[Tensor, "*num_rays 6"]] = None,
        disable_distortion: bool = False,
    ) -> RayBundle:
        """Generates rays for the given camera indices and coords where self isn't jagged

        This is a fairly complex function, so let's break this down slowly.

        Shapes involved:
            - num_rays: This is your output raybundle shape. It dictates the number and shape of the rays generated
            - num_cameras_batch_dims: This is the number of dimensions of our camera

        Args:
            camera_indices: Camera indices of the flattened cameras object to generate rays for.
                The shape of this is such that indexing into camera_indices["num_rays":...] will return the
                index into each batch dimension of the camera in order to get the correct camera specified by
                "num_rays".

                Example:
                    >>> cameras = Cameras(...)
                    >>> cameras.shape
                        (2, 3, 4)

                    >>> camera_indices = torch.tensor([0, 0, 0]) # We need an axis of length 3 since cameras.ndim == 3
                    >>> camera_indices.shape
                        (3,)
                    >>> coords = torch.tensor([1,1])
                    >>> coords.shape
                        (2,)
                    >>> out_rays = cameras.generate_rays(camera_indices=camera_indices, coords = coords)
                        # This will generate a RayBundle with a single ray for the
                        # camera at cameras[0,0,0] at image coordinates (1,1), so out_rays.shape == ()
                    >>> out_rays.shape
                        ()

                    >>> camera_indices = torch.tensor([[0,0,0]])
                    >>> camera_indices.shape
                        (1, 3)
                    >>> coords = torch.tensor([[1,1]])
                    >>> coords.shape
                        (1, 2)
                    >>> out_rays = cameras.generate_rays(camera_indices=camera_indices, coords = coords)
                        # This will generate a RayBundle with a single ray for the
                        # camera at cameras[0,0,0] at point (1,1), so out_rays.shape == (1,)
                        # since we added an extra dimension in front of camera_indices
                    >>> out_rays.shape
                        (1,)

                If you want more examples, check tests/cameras/test_cameras and the function check_generate_rays_shape

                The bottom line is that for camera_indices: (num_rays:..., num_cameras_batch_dims), num_rays is the
                output shape and if you index into the output RayBundle with some indices [i:...], if you index into
                camera_indices with camera_indices[i:...] as well, you will get a 1D tensor containing the batch
                indices into the original cameras object corresponding to that ray (ie: you will get the camera
                from our batched cameras corresponding to the ray at RayBundle[i:...]).

            coords: Coordinates of the pixels to generate rays for. If None, the full image will be rendered, meaning
                height and width get prepended to the num_rays dimensions. Indexing into coords with [i:...] will
                get you the image coordinates [x, y] of that specific ray located at output RayBundle[i:...].

            camera_opt_to_camera: Optional transform for the camera to world matrices.
                In terms of shape, it follows the same rules as coords, but indexing into it with [i:...] gets you
                the 2D camera to world transform matrix for the camera optimization at RayBundle[i:...].

            distortion_params_delta: Optional delta for the distortion parameters.
                In terms of shape, it follows the same rules as coords, but indexing into it with [i:...] gets you
                the 1D tensor with the 6 distortion parameters for the camera optimization at RayBundle[i:...].

            disable_distortion: If True, disables distortion.

        Returns:
            Rays for the given camera indices and coords. RayBundle.shape == num_rays
        """
        # Make sure we're on the right devices
        camera_indices = camera_indices.to(self.device)
        coords = coords.to(self.device)

        # Checking to make sure everything is of the right shape and type
        num_rays_shape = camera_indices.shape[:-1]
        assert camera_indices.shape == num_rays_shape + (self.ndim,)
        assert coords.shape == num_rays_shape + (2,)
        assert coords.shape[-1] == 2
        assert camera_opt_to_camera is None or camera_opt_to_camera.shape == num_rays_shape + (3, 4)
        assert distortion_params_delta is None or distortion_params_delta.shape == num_rays_shape + (6,)

        # Here, we've broken our indices down along the num_cameras_batch_dims dimension allowing us to index by all
        # of our output rays at each dimension of our cameras object
        true_indices = [camera_indices[..., i] for i in range(camera_indices.shape[-1])]

        # Get all our focal lengths, principal points and make sure they are the right shapes
        y = coords[..., 0]  # (num_rays,) get rid of the last dimension
        x = coords[..., 1]  # (num_rays,) get rid of the last dimension
        fx, fy = self.fx[true_indices].squeeze(-1), self.fy[true_indices].squeeze(-1)  # (num_rays,)
        cx, cy = self.cx[true_indices].squeeze(-1), self.cy[true_indices].squeeze(-1)  # (num_rays,)
        assert (
            y.shape == num_rays_shape
            and x.shape == num_rays_shape
            and fx.shape == num_rays_shape
            and fy.shape == num_rays_shape
            and cx.shape == num_rays_shape
            and cy.shape == num_rays_shape
        ), (
            str(num_rays_shape)
            + str(y.shape)
            + str(x.shape)
            + str(fx.shape)
            + str(fy.shape)
            + str(cx.shape)
            + str(cy.shape)
        )

        # Get our image coordinates and image coordinates offset by 1 (offsets used for dx, dy calculations)
        # Also make sure the shapes are correct
        coord = torch.stack([(x - cx) / fx, (y - cy) / fy], -1)  # (num_rays, 2)
        coord_x_offset = torch.stack([(x - cx + 1) / fx, (y - cy) / fy], -1)  # (num_rays, 2)
        coord_y_offset = torch.stack([(x - cx) / fx, (y - cy + 1) / fy], -1)  # (num_rays, 2)
        assert (
            coord.shape == num_rays_shape + (2,)
            and coord_x_offset.shape == num_rays_shape + (2,)
            and coord_y_offset.shape == num_rays_shape + (2,)
        )

        # Stack image coordinates and image coordinates offset by 1, check shapes too
        coord_stack = torch.stack([coord, coord_x_offset, coord_y_offset], dim=0)  # (3, num_rays, 2)
        assert coord_stack.shape == (3,) + num_rays_shape + (2,)

        # Undistorts our images according to our distortion parameters
        distortion_params = None
        if not disable_distortion:
            if self.distortion_params is not None:
                distortion_params = self.distortion_params[true_indices]
                if distortion_params_delta is not None:
                    distortion_params = distortion_params + distortion_params_delta
            elif distortion_params_delta is not None:
                distortion_params = distortion_params_delta

            # Do not apply distortion for equirectangular images
            if distortion_params is not None:
                mask = (self.camera_type[true_indices] != CameraType.EQUIRECTANGULAR.value).squeeze(-1)  # (num_rays)
                coord_mask = torch.stack([mask, mask, mask], dim=0)
                if mask.any() and (distortion_params != 0).any():
                    coord_stack[coord_mask, :] = camera_utils.radial_and_tangential_undistort(
                        coord_stack[coord_mask, :].reshape(3, -1, 2),
                        distortion_params[mask, :],
                    ).reshape(-1, 2)

        # Switch from OpenCV to OpenGL
        coord_stack[..., 1] *= -1

        # Make sure after we have undistorted our images, the shapes are still correct
        assert coord_stack.shape == (3,) + num_rays_shape + (2,)

        # Gets our directions for all our rays in camera coordinates and checks shapes at the end
        # Here, directions_stack is of shape (3, num_rays, 3)
        # directions_stack[0] is the direction for ray in camera coordinates
        # directions_stack[1] is the direction for ray in camera coordinates offset by 1 in x
        # directions_stack[2] is the direction for ray in camera coordinates offset by 1 in y
        cam_types = torch.unique(self.camera_type, sorted=False)
        directions_stack = torch.empty((3,) + num_rays_shape + (3,), device=self.device)

        c2w = self.camera_to_worlds[true_indices]
        assert c2w.shape == num_rays_shape + (3, 4)

        for cam in cam_types:
            if CameraType.PERSPECTIVE.value in cam_types:
                mask = (self.camera_type[true_indices] == CameraType.PERSPECTIVE.value).squeeze(-1)  # (num_rays)
                mask = torch.stack([mask, mask, mask], dim=0)
                directions_stack[..., 0][mask] = torch.masked_select(coord_stack[..., 0], mask).float()
                directions_stack[..., 1][mask] = torch.masked_select(coord_stack[..., 1], mask).float()
                directions_stack[..., 2][mask] = -1.0

            elif CameraType.FISHEYE.value in cam_types:
                mask = (self.camera_type[true_indices] == CameraType.FISHEYE.value).squeeze(-1)  # (num_rays)
                mask = torch.stack([mask, mask, mask], dim=0)

                theta = torch.sqrt(torch.sum(coord_stack**2, dim=-1))
                theta = torch.clip(theta, 0.0, math.pi)

                sin_theta = torch.sin(theta)

                directions_stack[..., 0][mask] = torch.masked_select(
                    coord_stack[..., 0] * sin_theta / theta, mask
                ).float()
                directions_stack[..., 1][mask] = torch.masked_select(
                    coord_stack[..., 1] * sin_theta / theta, mask
                ).float()
                directions_stack[..., 2][mask] = -torch.masked_select(torch.cos(theta), mask).float()

            elif CameraType.EQUIRECTANGULAR.value in cam_types:
                mask = (self.camera_type[true_indices] == CameraType.EQUIRECTANGULAR.value).squeeze(-1)  # (num_rays)
                mask = torch.stack([mask, mask, mask], dim=0)

                # For equirect, fx = fy = height = width/2
                # Then coord[..., 0] goes from -1 to 1 and coord[..., 1] goes from -1/2 to 1/2
                theta = -torch.pi * coord_stack[..., 0]  # minus sign for right-handed
                phi = torch.pi * (0.5 - coord_stack[..., 1])
                # use spherical in local camera coordinates (+y up, x=0 and z<0 is theta=0)
                directions_stack[..., 0][mask] = torch.masked_select(-torch.sin(theta) * torch.sin(phi), mask).float()
                directions_stack[..., 1][mask] = torch.masked_select(torch.cos(phi), mask).float()
                directions_stack[..., 2][mask] = torch.masked_select(-torch.cos(theta) * torch.sin(phi), mask).float()

            elif CameraType.ORTHOPHOTO.value in cam_types:
                # here the focal length determine the imaging area, the smaller fx, the bigger imaging area.
                mask = (self.camera_type[true_indices] == CameraType.ORTHOPHOTO.value).squeeze(-1)
                dir_mask = torch.stack([mask, mask, mask], dim=0)
                # in orthophoto cam, all rays have same direction, dir = R @ [0, 0, 1], R will be applied following.
                directions_stack[dir_mask] = torch.tensor(
                    [0.0, 0.0, -1.0], dtype=directions_stack.dtype, device=directions_stack.device
                )
                # in orthophoto cam, ray origins are grids, then transform grids with c2w, c2w @ P.
                grids = coord[mask]
                grids[..., 1] *= -1.0  # convert to left-hand system.
                grids = torch.cat([grids, torch.zeros_like(grids[..., -1:]), torch.ones_like(grids[..., -1:])], dim=-1)
                grids = torch.matmul(c2w[mask], grids[..., None]).squeeze(-1)
                c2w[..., :3, 3][mask] = grids

            elif CameraType.FISHEYE624.value in cam_types:
                mask = (self.camera_type[true_indices] == CameraType.FISHEYE624.value).squeeze(-1)  # (num_rays)
                coord_mask = torch.stack([mask, mask, mask], dim=0)

                # fisheye624 requires pixel coordinates to unproject, so we need to recomput the offsets in pixel coords.
                pcoord = torch.stack([x, y], -1)  # (num_rays, 2)
                pcoord_x_offset = torch.stack([x + 1, y], -1)  # (num_rays, 2)
                pcoord_y_offset = torch.stack([x, y + 1], -1)  # (num_rays, 2)

                # Stack image coordinates and image coordinates offset by 1, check shapes too
                pcoord_stack = torch.stack([pcoord, pcoord_x_offset, pcoord_y_offset], dim=0)  # (3, num_rays, 2)

                assert distortion_params is not None
                masked_coords = pcoord_stack[coord_mask, :]
                # The fisheye unprojection does not rely on planar/pinhole unprojection, thus the method needs
                # to access the focal length and principle points directly.
                camera_params = torch.cat(
                    [
                        fx[mask].unsqueeze(1),
                        fy[mask].unsqueeze(1),
                        cx[mask].unsqueeze(1),
                        cy[mask].unsqueeze(1),
                        distortion_params[mask, :],
                    ],
                    dim=1,
                )
                directions_stack[coord_mask] = camera_utils.fisheye624_unproject(masked_coords, camera_params)

            else:
                raise ValueError(f"Camera type {cam} not supported.")

        assert directions_stack.shape == (3,) + num_rays_shape + (3,)

        if camera_opt_to_camera is not None:
            c2w = pose_utils.multiply(c2w, camera_opt_to_camera)
        rotation = c2w[..., :3, :3]  # (..., 3, 3)
        assert rotation.shape == num_rays_shape + (3, 3)

        directions_stack = torch.sum(
            directions_stack[..., None, :] * rotation, dim=-1
        )  # (..., 1, 3) * (..., 3, 3) -> (..., 3)
        directions_stack, directions_norm = camera_utils.normalize_with_norm(directions_stack, -1)
        assert directions_stack.shape == (3,) + num_rays_shape + (3,)

        origins = c2w[..., :3, 3]  # (..., 3)
        assert origins.shape == num_rays_shape + (3,)

        directions = directions_stack[0]
        assert directions.shape == num_rays_shape + (3,)

        # norms of the vector going between adjacent coords, giving us dx and dy per output ray
        dx = torch.sqrt(torch.sum((directions - directions_stack[1]) ** 2, dim=-1))  # ("num_rays":...,)
        dy = torch.sqrt(torch.sum((directions - directions_stack[2]) ** 2, dim=-1))  # ("num_rays":...,)
        assert dx.shape == num_rays_shape and dy.shape == num_rays_shape

        pixel_area = (dx * dy)[..., None]  # ("num_rays":..., 1)
        assert pixel_area.shape == num_rays_shape + (1,)

        times = self.times[camera_indices, 0] if self.times is not None else None

        metadata = (
            self._apply_fn_to_dict(self.metadata, lambda x: x[true_indices]) if self.metadata is not None else None
        )
        if metadata is not None:
            metadata["directions_norm"] = directions_norm[0].detach()
        else:
            metadata = {"directions_norm": directions_norm[0].detach()}

        return RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=pixel_area,
            camera_indices=camera_indices,
            times=times,
            metadata=metadata,
        )