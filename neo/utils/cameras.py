import torch
import numpy as np

from PIL import Image
from typing import Dict, Any
from pathlib import Path

from nerfstudio.utils.io import load_from_json
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.utils.rich_utils import CONSOLE

from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_all,
    get_train_eval_split_filename,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
)

MAX_AUTO_RESOLUTION = 1600


def get_cameras(dataparser_config: Dict[str, Any], split: str = "train") -> Cameras:

    if dataparser_config.data.suffix == ".json":
        camera_data = load_from_json(dataparser_config.data)
        data_dir = dataparser_config.data.parent
    else:
        camera_data = load_from_json(dataparser_config.data / "transforms.json")
        data_dir = dataparser_config.data

    image_filenames = []
    mask_filenames = []
    depth_filenames = []
    poses = []

    fx_fixed = "fl_x" in camera_data
    fy_fixed = "fl_y" in camera_data
    cx_fixed = "cx" in camera_data
    cy_fixed = "cy" in camera_data
    height_fixed = "h" in camera_data
    width_fixed = "w" in camera_data
    distort_fixed = False
    for distort_key in ["k1", "k2", "k3", "p1", "p2", "distortion_params"]:
        if distort_key in camera_data:
            distort_fixed = True
            break
    fisheye_crop_radius = camera_data.get("fisheye_crop_radius", None)
    fx = []
    fy = []
    cx = []
    cy = []
    height = []
    width = []
    distort = []

    # sort the frames by fname
    fnames = []
    for frame in camera_data["frames"]:
        filepath = Path(frame["file_path"])
        fname = _get_fname(dataparser_config, filepath, data_dir)
        fnames.append(fname)
    inds = np.argsort(fnames)
    frames = [camera_data["frames"][ind] for ind in inds]

    for frame in frames:
        filepath = Path(frame["file_path"])
        fname = _get_fname(dataparser_config, filepath, data_dir)

        if not fx_fixed:
            assert "fl_x" in frame, "fx not specified in frame"
            fx.append(float(frame["fl_x"]))
        if not fy_fixed:
            assert "fl_y" in frame, "fy not specified in frame"
            fy.append(float(frame["fl_y"]))
        if not cx_fixed:
            assert "cx" in frame, "cx not specified in frame"
            cx.append(float(frame["cx"]))
        if not cy_fixed:
            assert "cy" in frame, "cy not specified in frame"
            cy.append(float(frame["cy"]))
        if not height_fixed:
            assert "h" in frame, "height not specified in frame"
            height.append(int(frame["h"]))
        if not width_fixed:
            assert "w" in frame, "width not specified in frame"
            width.append(int(frame["w"]))
        if not distort_fixed:
            distort.append(
                torch.tensor(frame["distortion_params"], dtype=torch.float32)
                if "distortion_params" in frame
                else camera_utils.get_distortion_params(
                    k1=float(frame["k1"]) if "k1" in frame else 0.0,
                    k2=float(frame["k2"]) if "k2" in frame else 0.0,
                    k3=float(frame["k3"]) if "k3" in frame else 0.0,
                    k4=float(frame["k4"]) if "k4" in frame else 0.0,
                    p1=float(frame["p1"]) if "p1" in frame else 0.0,
                    p2=float(frame["p2"]) if "p2" in frame else 0.0,
                )
            )

        image_filenames.append(fname)
        poses.append(np.array(frame["transform_matrix"]))
        if "mask_path" in frame:
            mask_filepath = Path(frame["mask_path"])
            mask_fname = _get_fname(
                dataparser_config,
                mask_filepath,
                data_dir,
                downsample_folder_prefix="masks_",
            )
            mask_filenames.append(mask_fname)

        if "depth_file_path" in frame:
            depth_filepath = Path(frame["depth_file_path"])
            depth_fname = _get_fname(dataparser_config, depth_filepath, data_dir, downsample_folder_prefix="depths_")
            depth_filenames.append(depth_fname)

    assert len(mask_filenames) == 0 or (len(mask_filenames) == len(image_filenames)), """
    Different number of image and mask filenames.
    You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
    """
    assert len(depth_filenames) == 0 or (len(depth_filenames) == len(image_filenames)), """
    Different number of image and depth filenames.
    You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
    """

    has_split_files_spec = any(f"{split}_filenames" in camera_data for split in ("train", "val", "test"))
    if f"{split}_filenames" in camera_data:
        # Validate split first
        split_filenames = set(_get_fname(dataparser_config, Path(x), data_dir) for x in camera_data[f"{split}_filenames"])
        unmatched_filenames = split_filenames.difference(image_filenames)
        if unmatched_filenames:
            raise RuntimeError(f"Some filenames for split {split} were not found: {unmatched_filenames}.")

        indices = [i for i, path in enumerate(image_filenames) if path in split_filenames]
        CONSOLE.log(f"[yellow] Dataset is overriding {split}_indices to {indices}")
        indices = np.array(indices, dtype=np.int32)
    elif has_split_files_spec:
        raise RuntimeError(f"The dataset's list of filenames for split {split} is missing.")
    else:
        # find train and eval indices based on the eval_mode specified
        if dataparser_config.eval_mode == "fraction":
            i_train, i_eval = get_train_eval_split_fraction(image_filenames, dataparser_config.train_split_fraction)
        elif dataparser_config.eval_mode == "filename":
            i_train, i_eval = get_train_eval_split_filename(image_filenames)
        elif dataparser_config.eval_mode == "interval":
            i_train, i_eval = get_train_eval_split_interval(image_filenames, dataparser_config.eval_interval)
        elif dataparser_config.eval_mode == "all":
            CONSOLE.log(
                "[yellow] Be careful with '--eval-mode=all'. If using camera optimization, the cameras may diverge in the current implementation, giving unpredictable results."
            )
            i_train, i_eval = get_train_eval_split_all(image_filenames)
        else:
            raise ValueError(f"Unknown eval mode {dataparser_config.eval_mode}")

        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

    if "orientation_override" in camera_data:
        orientation_method = camera_data["orientation_override"]
        CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
    else:
        orientation_method = dataparser_config.orientation_method

    poses = torch.from_numpy(np.array(poses).astype(np.float32))
    poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
        poses,
        method=orientation_method,
        center_method=dataparser_config.center_method,
    )

    # Scale poses
    scale_factor = 1.0
    if dataparser_config.auto_scale_poses:
        scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
    scale_factor *= dataparser_config.scale_factor

    poses[:, :3, 3] *= scale_factor

    # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
    image_filenames = [image_filenames[i] for i in indices]
    mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
    depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []

    idx_tensor = torch.tensor(indices, dtype=torch.long)
    poses = poses[idx_tensor]

    # in x,y,z order
    # assumes that the scene is centered at the origin
    aabb_scale = dataparser_config.scene_scale
    scene_box = SceneBox(
        aabb=torch.tensor(
            [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
        )
    )

    if "camera_model" in camera_data:
        camera_type = CAMERA_MODEL_TO_TYPE[camera_data["camera_model"]]
    else:
        camera_type = CameraType.PERSPECTIVE

    fx = float(camera_data["fl_x"]) if fx_fixed else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
    fy = float(camera_data["fl_y"]) if fy_fixed else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
    cx = float(camera_data["cx"]) if cx_fixed else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
    cy = float(camera_data["cy"]) if cy_fixed else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
    height = int(camera_data["h"]) if height_fixed else torch.tensor(height, dtype=torch.int32)[idx_tensor]
    width = int(camera_data["w"]) if width_fixed else torch.tensor(width, dtype=torch.int32)[idx_tensor]
    if distort_fixed:
        distortion_params = (
            torch.tensor(camera_data["distortion_params"], dtype=torch.float32)
            if "distortion_params" in camera_data
            else camera_utils.get_distortion_params(
                k1=float(camera_data["k1"]) if "k1" in camera_data else 0.0,
                k2=float(camera_data["k2"]) if "k2" in camera_data else 0.0,
                k3=float(camera_data["k3"]) if "k3" in camera_data else 0.0,
                k4=float(camera_data["k4"]) if "k4" in camera_data else 0.0,
                p1=float(camera_data["p1"]) if "p1" in camera_data else 0.0,
                p2=float(camera_data["p2"]) if "p2" in camera_data else 0.0,
            )
        )
    else:
        distortion_params = torch.stack(distort, dim=0)[idx_tensor]

    # Only add fisheye crop radius parameter if the images are actually fisheye, to allow the same config to be used
    # for both fisheye and non-fisheye datasets.
    metadata = {}
    if (camera_type in [CameraType.FISHEYE, CameraType.FISHEYE624]) and (fisheye_crop_radius is not None):
        metadata["fisheye_crop_radius"] = fisheye_crop_radius

    cameras = Cameras(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        distortion_params=distortion_params,
        height=height,
        width=width,
        camera_to_worlds=poses[:, :3, :4],
        camera_type=camera_type,
        metadata=metadata,
    )

    assert dataparser_config.downscale_factor is not None
    cameras.rescale_output_resolution(scaling_factor=1.0 / dataparser_config.downscale_factor)

    return cameras


def _get_fname(dataparser_config: Dict[str, Any], filepath: Path, data_dir: Path, downsample_folder_prefix="images_") -> Path:
    """Get the filename of the image file.
    downsample_folder_prefix can be used to point to auxiliary image data, e.g. masks

    filepath: the base file name of the transformations.
    data_dir: the directory of the data that contains the transform file
    downsample_folder_prefix: prefix of the newly generated downsampled images
    """

    if dataparser_config.downscale_factor is None:
        if dataparser_config.downscale_factor is None:
            test_img = Image.open(data_dir / filepath)
            h, w = test_img.size
            max_res = max(h, w)
            df = 0
            while True:
                if (max_res / 2 ** (df)) <= MAX_AUTO_RESOLUTION:
                    break
                if not (data_dir / f"{downsample_folder_prefix}{2**(df+1)}" / filepath.name).exists():
                    break
                df += 1

            dataparser_config.downscale_factor = 2**df
            CONSOLE.log(f"Auto image downscale factor of {dataparser_config.downscale_factor}")
        else:
            dataparser_config.downscale_factor = dataparser_config.downscale_factor

    if dataparser_config.downscale_factor > 1:
        return data_dir / f"{downsample_folder_prefix}{dataparser_config.downscale_factor}" / filepath.name
    return data_dir / filepath