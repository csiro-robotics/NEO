import pyrealsense2 as rs
import numpy as np
import cv2
import os
import tyro

from pathlib import Path

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE

from experiment_config import NeoDataConfig


def read_realsense_bag(bag_file_path: Path, output_path: Path, rotate_images: bool, skip_frames: int=6):
    # Configure the pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable playback of the bag file
    config.enable_device_from_file(str(bag_file_path), repeat_playback=False)

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
                num = str(j)
                while len(num) < 4:
                    num = '0' + num

                if rotate_images:
                    color_image = cv2.rotate(color_image, cv2.ROTATE_180)
                    depth_image = cv2.rotate(depth_image, cv2.ROTATE_180)

                # Save the images
                cv2.imwrite(str(output_path / f'rgb/{num}.png'), color_image)
                cv2.imwrite(str(output_path / f'depth/{num}.png'), depth_image)

                # Update progress
                CONSOLE.print(f"Extracted frame {i}")
                j += 1
            i += 1
    finally:
        # Stop the pipeline
        CONSOLE.print(f"[bold green]Number of Frames:[/bold green] {j}")
        pipeline.stop()

def extract_images_rs(config: NeoDataConfig, skip_frames: int = 6) -> None:
    """Main function."""

    output_path = config.get_base_dir()

    # Need both before and after to be present in order to perform final alignment
    for stage in ["before", "after"]:
        bag_file_path = Path(f"{str(output_path).replace(str(config.root_input), f'{str(config.root_input)}/raw_bagfiles')}_{stage}.bag")

        if bag_file_path.exists():
            CONSOLE.print(f"[bold green]Output Path:[/bold green] {output_path}")
            CONSOLE.print(f"[bold green]Bag File Path:[/bold green] {bag_file_path}")
            
            read_realsense_bag(bag_file_path, output_path / stage, True, skip_frames=skip_frames)

        else:
            CONSOLE.print(f"Bag file does not exist: {bag_file_path}")
            return


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")

    extract_images_rs(
        tyro.cli(
            NeoDataConfig,
            description=convert_markup_to_ansi(__doc__),
        )
    )

if __name__ == "__main__":
    entrypoint()