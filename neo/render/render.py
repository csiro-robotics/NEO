from __future__ import annotations

import json
import shutil
import sys
import os
import cv2
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import mediapy as media
import numpy as np
import torch
import tyro
import viser.transforms as tf
from rich import box, style
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from typing_extensions import Annotated

from nerfstudio.cameras.cameras import Cameras, CameraType, RayBundle
from nerfstudio.model_components import renderers
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.scripts.render import insert_spherical_metadata_into_file, get_crop_from_json
from nerfstudio.scripts.render import RenderCameraPath, RenderInterpolated, SpiralRender, DatasetRender, CropData
from nerfstudio.utils import colormaps, install_checks
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
from nerfstudio.utils.scripts import run_command

from neo.render.camera_paths import get_path_from_json


def _render_trajectory_video(
    pipeline: Pipeline,
    cameras: Cameras,
    image_names: List[str],
    output_filename: Path,
    rendered_output_names: List[str],
    crop_data: Optional[CropData] = None,
    rendered_resolution_scaling_factor: float = 1.0,
    seconds: float = 5.0,
    output_format: Literal["images", "video"] = "video",
    image_format: Literal["jpeg", "png"] = "jpeg",
    jpeg_quality: int = 100,
    depth_near_plane: Optional[float] = None,
    depth_far_plane: Optional[float] = None,
    colormap_options: colormaps.ColormapOptions = colormaps.ColormapOptions(),
    render_nearest_camera=False,
    check_occlusions: bool = False,
) -> None:
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        output_filename: Name of the output file.
        rendered_output_names: List of outputs to visualise.
        crop_data: Crop data to apply to the rendered images.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        seconds: Length of output video.
        output_format: How to save output data.
        depth_near_plane: Closest depth to consider when using the colormap for depth. If None, use min value.
        depth_far_plane: Furthest depth to consider when using the colormap for depth. If None, use max value.
        colormap_options: Options for colormap.
        render_nearest_camera: Whether to render the nearest training camera to the rendered camera.
        check_occlusions: If true, checks line-of-sight occlusions when computing camera distance and rejects cameras not visible to each other
    """
    CONSOLE.print("[bold green]Creating trajectory " + output_format)
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    cameras = cameras.to(pipeline.device)
    fps = len(cameras) / seconds

    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(
            text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
            show_speed=True,
        ),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=False, compact=False),
        TimeElapsedColumn(),
    )
    output_image_dir = output_filename.parent / output_filename.stem

    # Create directroies to save images
    for rendered_output_name in rendered_output_names:
        os.makedirs(output_image_dir / rendered_output_name, exist_ok=True)

    if output_format == "images":
        output_image_dir.mkdir(parents=True, exist_ok=True)
    if output_format == "video":
        # make the folder if it doesn't exist
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        # NOTE:
        # we could use ffmpeg_args "-movflags faststart" for progressive download,
        # which would force moov atom into known position before mdat,
        # but then we would have to move all of mdat to insert metadata atom
        # (unless we reserve enough space to overwrite with our uuid tag,
        # but we don't know how big the video file will be, so it's not certain!)

    with ExitStack() as stack:
        writer = None

        if render_nearest_camera:
            assert pipeline.datamanager.train_dataset is not None
            train_dataset = pipeline.datamanager.train_dataset
            train_cameras = train_dataset.cameras.to(pipeline.device)
        else:
            train_dataset = None
            train_cameras = None

        with progress:
            for camera_idx in progress.track(range(cameras.size), description=""):
                obb_box = None
                if crop_data is not None:
                    obb_box = crop_data.obb

                max_dist, max_idx = -1, -1
                true_max_dist, true_max_idx = -1, -1

                if render_nearest_camera:
                    assert pipeline.datamanager.train_dataset is not None
                    assert train_dataset is not None
                    assert train_cameras is not None
                    cam_pos = cameras[camera_idx].camera_to_worlds[:, 3].cpu()
                    cam_quat = tf.SO3.from_matrix(cameras[camera_idx].camera_to_worlds[:3, :3].numpy(force=True)).wxyz

                    for i in range(len(train_cameras)):
                        train_cam_pos = train_cameras[i].camera_to_worlds[:, 3].cpu()
                        # Make sure the line of sight from rendered cam to training cam is not blocked by any object
                        bundle = RayBundle(
                            origins=cam_pos.view(1, 3),
                            directions=((cam_pos - train_cam_pos) / (cam_pos - train_cam_pos).norm()).view(1, 3),
                            pixel_area=torch.tensor(1).view(1, 1),
                            nears=torch.tensor(0.05).view(1, 1),
                            fars=torch.tensor(100).view(1, 1),
                            camera_indices=torch.tensor(0).view(1, 1),
                            metadata={},
                        ).to(pipeline.device)
                        outputs = pipeline.model.get_outputs(bundle)

                        q = tf.SO3.from_matrix(train_cameras[i].camera_to_worlds[:3, :3].numpy(force=True)).wxyz
                        # calculate distance between two quaternions
                        rot_dist = 1 - np.dot(q, cam_quat) ** 2
                        pos_dist = torch.norm(train_cam_pos - cam_pos)
                        dist = 0.3 * rot_dist + 0.7 * pos_dist

                        if true_max_dist == -1 or dist < true_max_dist:
                            true_max_dist = dist
                            true_max_idx = i

                        if outputs["depth"][0] < torch.norm(cam_pos - train_cam_pos).item():
                            continue

                        if check_occlusions and (max_dist == -1 or dist < max_dist):
                            max_dist = dist
                            max_idx = i

                    if max_idx == -1:
                        max_idx = true_max_idx

                if crop_data is not None:
                    with renderers.background_color_override_context(
                        crop_data.background_color.to(pipeline.device)
                    ), torch.no_grad():
                        outputs = pipeline.model.get_outputs_for_camera(
                            cameras[camera_idx : camera_idx + 1], obb_box=obb_box
                        )
                else:
                    with torch.no_grad():
                        outputs = pipeline.model.get_outputs_for_camera(
                            cameras[camera_idx : camera_idx + 1], obb_box=obb_box
                        )
                        if rendered_output_names is not None and "rgba" in rendered_output_names:
                            rgba = pipeline.model.get_rgba_image(outputs=outputs, output_name="rgb")
                            outputs["rgba"] = rgba

                render_image = []
                for rendered_output_name in rendered_output_names:
                    if rendered_output_name not in outputs:
                        CONSOLE.rule("Error", style="red")
                        CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                        CONSOLE.print(
                            f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center"
                        )
                        sys.exit(1)
                    output_image = outputs[rendered_output_name]
                    is_depth = rendered_output_name.find("depth") != -1
                    if is_depth:
                        # NOTE Commented out to generate true depth
                        # output_image = (
                        #     colormaps.apply_depth_colormap(
                        #         output_image,
                        #         accumulation=outputs["accumulation"],
                        #         near_plane=depth_near_plane,
                        #         far_plane=depth_far_plane,
                        #         colormap_options=colormap_options,
                        #     )
                        #     .cpu()
                        #     .numpy()
                        # )
                        output_image = output_image.detach().cpu().numpy()
                    elif rendered_output_name == "rgba":
                        output_image = output_image.detach().cpu().numpy()
                    else:
                        output_image = (
                            colormaps.apply_colormap(
                                image=output_image,
                                colormap_options=colormap_options,
                            )
                            .cpu()
                            .numpy()
                        )
                    render_image.append(output_image)

                # Add closest training image to the right of the rendered image
                if render_nearest_camera:
                    assert train_dataset is not None
                    assert train_cameras is not None
                    img = train_dataset.get_image_float32(max_idx)
                    height = cameras.image_height[0]
                    # maintain the resolution of the img to calculate the width from the height
                    width = int(img.shape[1] * (height / img.shape[0]))
                    resized_image = torch.nn.functional.interpolate(
                        img.permute(2, 0, 1)[None], size=(int(height), int(width))
                    )[0].permute(1, 2, 0)
                    resized_image = (
                        colormaps.apply_colormap(
                            image=resized_image,
                            colormap_options=colormap_options,
                        )
                        .cpu()
                        .numpy()
                    )
                    render_image.append(resized_image)

                for idx, rendered_output_name in enumerate(rendered_output_names):
                    if output_format == "images":
                        if 'depth' in rendered_output_name:
                            cv2.imwrite(output_image_dir / rendered_output_name / f"{image_names[camera_idx]}.png", render_image[idx])
                        elif image_format == "png":
                            media.write_image(output_image_dir / rendered_output_name / f"{image_names[camera_idx]}.png", render_image[idx], fmt="png")
                        elif image_format == "jpeg":
                            media.write_image(
                                output_image_dir / rendered_output_name / f"{image_names[camera_idx]}.jpg", render_image[idx], fmt="jpeg", quality=jpeg_quality
                            )
                if output_format == "video":
                    if writer is None:
                        render_width = int(render_image.shape[1])
                        render_height = int(render_image.shape[0])
                        writer = stack.enter_context(
                            media.VideoWriter(
                                path=output_filename,
                                shape=(render_height, render_width),
                                fps=fps,
                            )
                        )
                    writer.add_image(render_image)

    table = Table(
        title=None,
        show_header=False,
        box=box.MINIMAL,
        title_style=style.Style(bold=True),
    )
    if output_format == "video":
        if cameras.camera_type[0] == CameraType.EQUIRECTANGULAR.value:
            CONSOLE.print("Adding spherical camera data")
            insert_spherical_metadata_into_file(output_filename)
        table.add_row("Video", str(output_filename))
    else:
        table.add_row("Images", str(output_image_dir))
    CONSOLE.print(Panel(table, title="[bold][green]:tada: Render Complete :tada:[/bold]", expand=False))

def transforms_to_camerapath(
        transforms_path: Path
) -> Union[List[str], Dict[Any]]:
    """
    This function converts transfomrs.json file to create camera-path for rendering.
    """

    assert input is not None, "Please specify input argument."

    # Load config
    transforms = json.loads(transforms_path.read_text())

    fov_x = np.rad2deg(2 * np.arctan(transforms["h"] / (2 * transforms["fl_y"])))
    aspect = transforms["w"] / transforms["h"]

    camerapath_file = {}
    camerapath_file["default_fov"] = fov_x
    camerapath_file["default_transition_sec"] = 2.0

    keyframes = []
    for i in range(2):
        keyframe = {"matrix": transforms["frames"][-1 * i]["transform_matrix"],
                    "aspect": aspect, 
                    "override_transition_enabled": False, 
                    "override_transition_sec": None}

        keyframes.append(keyframe)        

    camerapath_file["keyframes"] = keyframes
    camerapath_file["camera_type"] = "perspective"
    camerapath_file["render_height"] = transforms["h"]
    camerapath_file["render_width"] = transforms["w"]
    camerapath_file["cx"] = transforms["cx"]
    camerapath_file["cy"] = transforms["cy"]
    camerapath_file["fl_x"] = transforms["fl_x"]
    camerapath_file["fl_y"] = transforms["fl_y"]
    camerapath_file["fps"] = 3.0
    camerapath_file["seconds"] = 4.0
    camerapath_file["is_cycle"] = False
    camerapath_file["smoothness_value"] = 0.0

    camera_path = []
    image_names = []
    for frame in transforms["frames"]:
        camera = {"camera_to_world": frame["transform_matrix"],
                  "fov": fov_x,
                  "aspect": aspect}
        
        camera_path.append(camera)
        image_names.append(frame["file_path"].split('/')[-1].split('.')[0])

    camerapath_file["camera_path"] = camera_path
    
    return image_names, camerapath_file

@dataclass
class RenderFromTransforms(RenderCameraPath):
    """Render images from specfied camera transforms."""

    transforms_filename: Path = Path("transforms.json")
    """Filename of the camera transforms."""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""

    def main(self) -> None:
        """Main function."""
        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
        )

        install_checks.check_ffmpeg_installed()

        image_names, camera_path = transforms_to_camerapath(self.transforms_filename)

        seconds = camera_path["seconds"]
        crop_data = get_crop_from_json(camera_path)
        camera_path = get_path_from_json(camera_path)

        if (
            camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value
            or camera_path.camera_type[0] == CameraType.VR180_L.value
        ):
            # temp folder for writing left and right view renders
            temp_folder_path = self.output_path.parent / (self.output_path.stem + "_temp")

            Path(temp_folder_path).mkdir(parents=True, exist_ok=True)
            left_eye_path = temp_folder_path / "render_left.mp4"

            self.output_path = left_eye_path

            if camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value:
                CONSOLE.print("[bold green]:goggles: Omni-directional Stereo VR :goggles:")
            else:
                CONSOLE.print("[bold green]:goggles: VR180 :goggles:")

            CONSOLE.print("Rendering left eye view")

        # add mp4 suffix to video output if none is specified
        if self.output_format == "video" and str(self.output_path.suffix) == "":
            self.output_path = self.output_path.with_suffix(".mp4")

        if self.camera_idx is not None:
            camera_path.metadata = {"cam_idx": self.camera_idx}

        _render_trajectory_video(
            pipeline,
            camera_path,
            image_names=image_names,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            crop_data=crop_data,
            seconds=seconds,
            output_format=self.output_format,
            image_format=self.image_format,
            jpeg_quality=self.jpeg_quality,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            colormap_options=self.colormap_options,
            render_nearest_camera=self.render_nearest_camera,
            check_occlusions=self.check_occlusions,
        )

        if (
            camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value
            or camera_path.camera_type[0] == CameraType.VR180_L.value
        ):
            # declare paths for left and right renders

            left_eye_path = self.output_path
            right_eye_path = left_eye_path.parent / "render_right.mp4"

            self.output_path = right_eye_path

            if camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value:
                camera_path.camera_type[0] = CameraType.OMNIDIRECTIONALSTEREO_R.value
            else:
                camera_path.camera_type[0] = CameraType.VR180_R.value

            CONSOLE.print("Rendering right eye view")
            _render_trajectory_video(
                pipeline,
                camera_path,
                output_filename=self.output_path,
                rendered_output_names=self.rendered_output_names,
                rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
                crop_data=crop_data,
                seconds=seconds,
                output_format=self.output_format,
                image_format=self.image_format,
                jpeg_quality=self.jpeg_quality,
                depth_near_plane=self.depth_near_plane,
                depth_far_plane=self.depth_far_plane,
                colormap_options=self.colormap_options,
                render_nearest_camera=self.render_nearest_camera,
                check_occlusions=self.check_occlusions,
            )

            self.output_path = Path(str(left_eye_path.parent)[:-5] + ".mp4")

            if camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_R.value:
                # stack the left and right eye renders vertically for ODS final output
                ffmpeg_ods_command = ""
                if self.output_format == "video":
                    ffmpeg_ods_command = f'ffmpeg -y -i "{left_eye_path}" -i "{right_eye_path}" -filter_complex "[0:v]pad=iw:2*ih[int];[int][1:v]overlay=0:h" -c:v libx264 -crf 23 -preset veryfast "{self.output_path}"'
                    run_command(ffmpeg_ods_command, verbose=False)
                if self.output_format == "images":
                    # create a folder for the stacked renders
                    self.output_path = Path(str(left_eye_path.parent)[:-5])
                    self.output_path.mkdir(parents=True, exist_ok=True)
                    if self.image_format == "png":
                        ffmpeg_ods_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.png")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.png")}" -filter_complex vstack -start_number 0 "{str(self.output_path)+"//%05d.png"}"'
                    elif self.image_format == "jpeg":
                        ffmpeg_ods_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.jpg")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.jpg")}" -filter_complex vstack -start_number 0 "{str(self.output_path)+"//%05d.jpg"}"'
                    run_command(ffmpeg_ods_command, verbose=False)

                # remove the temp files directory
                if str(left_eye_path.parent)[-5:] == "_temp":
                    shutil.rmtree(left_eye_path.parent, ignore_errors=True)
                CONSOLE.print("[bold green]Final ODS Render Complete")
            else:
                # stack the left and right eye renders horizontally for VR180 final output
                self.output_path = Path(str(left_eye_path.parent)[:-5] + ".mp4")
                ffmpeg_vr180_command = ""
                if self.output_format == "video":
                    ffmpeg_vr180_command = f'ffmpeg -y -i "{right_eye_path}" -i "{left_eye_path}" -filter_complex "[1:v]hstack=inputs=2" -c:a copy "{self.output_path}"'
                    run_command(ffmpeg_vr180_command, verbose=False)
                if self.output_format == "images":
                    # create a folder for the stacked renders
                    self.output_path = Path(str(left_eye_path.parent)[:-5])
                    self.output_path.mkdir(parents=True, exist_ok=True)
                    if self.image_format == "png":
                        ffmpeg_vr180_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.png")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.png")}" -filter_complex hstack -start_number 0 "{str(self.output_path)+"//%05d.png"}"'
                    elif self.image_format == "jpeg":
                        ffmpeg_vr180_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.jpg")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.jpg")}" -filter_complex hstack -start_number 0 "{str(self.output_path)+"//%05d.jpg"}"'
                    run_command(ffmpeg_vr180_command, verbose=False)

                # remove the temp files directory
                if str(left_eye_path.parent)[-5:] == "_temp":
                    shutil.rmtree(left_eye_path.parent, ignore_errors=True)
                CONSOLE.print("[bold green]Final VR180 Render Complete")


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[RenderFromTransforms, tyro.conf.subcommand(name="render-from-transforms")],
        Annotated[RenderCameraPath, tyro.conf.subcommand(name="camera-path")],
        Annotated[RenderInterpolated, tyro.conf.subcommand(name="interpolate")],
        Annotated[SpiralRender, tyro.conf.subcommand(name="spiral")],
        Annotated[DatasetRender, tyro.conf.subcommand(name="dataset")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()