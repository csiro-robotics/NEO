import cv2
import os
import json
import shutil
import tyro
import numpy as np
import pyrealsense2 as rs

from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage
from pathlib import Path
from scipy.optimize import minimize_scalar

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE

from experiment_config import NeoDataConfig


def read_realsense_bag(bag_file_path: Path, output_path: Path, rotate_images: bool, skip_frames: int=6, frame_list: list=None):
    # Configure the pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable playback of the bag file
    config.enable_device_from_file(str(bag_file_path), repeat_playback=False)

    # Start the pipeline
    pipeline.start(config)

    os.makedirs(output_path, exist_ok=True)
    if (output_path / 'rgb').exists():
        shutil.rmtree(output_path / 'rgb')
    if (output_path / 'depth').exists():
        shutil.rmtree(output_path / 'depth')
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
            
            if frame_list is not None:
                if i in frame_list:
                    num = str(j).zfill(4)
                    if rotate_images:
                        color_image = cv2.rotate(color_image, cv2.ROTATE_180)
                        depth_image = cv2.rotate(depth_image, cv2.ROTATE_180)

                    # Save the images
                    cv2.imwrite(str(output_path / f'rgb/{num}.png'), color_image)
                    cv2.imwrite(str(output_path / f'depth/{num}.png'), depth_image)

                    # Update progress
                    CONSOLE.print(f"Extracted frame {i}")
                    j += 1
            else:
                if i % skip_frames == 0:

                    num = str(j).zfill(4)
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

def read_mp4_file(mp4_file_path: Path, output_path: Path, skip_frames: int=6, frame_list: list=None):
    cap = cv2.VideoCapture(str(mp4_file_path))
    os.makedirs(output_path, exist_ok=True)
    if (output_path / 'rgb').exists():
        shutil.rmtree(output_path / 'rgb')
    os.makedirs(output_path / 'rgb', exist_ok=True)

    frame_count = 0
    saved_frame_count = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_list is not None:
            if frame_count in frame_list:
                num = str(saved_frame_count).zfill(4)
                cv2.imwrite(str(output_path / f'rgb/{num}.png'), frame)
                CONSOLE.print(f"Extracted frame {frame_count}")
                saved_frame_count += 1
        
        else:
            if frame_count % skip_frames == 0:
                num = str(saved_frame_count).zfill(4)
                cv2.imwrite(str(output_path / f'rgb/{num}.png'), frame)
                CONSOLE.print(f"Extracted frame {frame_count}")
                saved_frame_count += 1

        frame_count += 1

    cap.release()
    CONSOLE.print(f"[bold green]Number of Frames:[/bold green] {saved_frame_count}")

def read_ros2_bag(bag_file_path: Path, output_path: Path, rotate_images: bool, skip_frames: int=2, frame_list: list=None):

    # Create a directory for the output
    os.makedirs(output_path, exist_ok=True)
    if (output_path / 'rgb').exists():
        shutil.rmtree(output_path / 'rgb')
    if (output_path / 'depth').exists():
        shutil.rmtree(output_path / 'depth')
    os.makedirs(output_path / 'rgb', exist_ok=True)
    os.makedirs(output_path / 'depth', exist_ok=True)

    print("Reading bag2 file:", bag_file_path)
    with AnyReader([bag_file_path]) as reader:
        print("Available topics:")
        topics = sorted([(connection.topic, connection.msgtype) for connection in reader.connections])
        for topic, msgtype in topics:
            print(topic, msgtype)

        rgb_count = 0
        depth_count = 0
        saved_rgb_count = 1
        saved_depth_count = 1
        for connection, timestamp, rawdata in reader.messages():

            if connection.topic == '/camera/color/image_raw':
                if frame_list is not None:
                    if rgb_count in frame_list:
                        msg = reader.deserialize(rawdata, connection.msgtype)
                        img = message_to_cvimage(msg, 'bgr8')
                        if rotate_images:
                            img = cv2.rotate(img, cv2.ROTATE_180)
                        num = str(saved_rgb_count).zfill(4)
                        cv2.imwrite(f"{output_path}/rgb/frame_{num}.png", img)
                        saved_rgb_count += 1
                        rgb_count += 1
                    else:
                        rgb_count += 1
                else:
                    if rgb_count % skip_frames == 0:
                        msg = reader.deserialize(rawdata, connection.msgtype)
                        img = message_to_cvimage(msg, 'bgr8')
                        if rotate_images:
                            img = cv2.rotate(img, cv2.ROTATE_180)
                        num = str(saved_rgb_count).zfill(4)
                        cv2.imwrite(f"{output_path}/rgb/frame_{num}.png", img)
                        saved_rgb_count += 1
                        rgb_count += 1
                    else:
                        rgb_count += 1

            if connection.topic == '/camera/depth/image_raw':
                if frame_list is not None:
                    if depth_count in frame_list:
                        msg = reader.deserialize(rawdata, connection.msgtype)
                        img = message_to_cvimage(msg, '16UC1')
                        if rotate_images:
                            img = cv2.rotate(img, cv2.ROTATE_180)
                        num = str(saved_depth_count).zfill(4)
                        cv2.imwrite(f"{output_path}/depth/frame_{num}.png", img)
                        saved_depth_count += 1
                        depth_count += 1
                    else:
                        depth_count += 1
                else:
                    if depth_count % skip_frames == 0:
                        msg = reader.deserialize(rawdata, connection.msgtype)
                        img = message_to_cvimage(msg, '16UC1')
                        if rotate_images:
                            img = cv2.rotate(img, cv2.ROTATE_180)
                        num = str(saved_depth_count).zfill(4)
                        cv2.imwrite(f"{output_path}/depth/frame_{num}.png", img)
                        saved_depth_count += 1
                        depth_count += 1
                    else:
                        depth_count += 1

def select_frames_to_extract(config: NeoDataConfig, stage: str, num_frames: int = 250) -> list:

    path = config.get_base_dir() / stage

    assert path.exists(), "Before and After directories must exist."

    # Load the transforms
    with open(path / 'transforms.json', 'r') as file:
        transforms = json.load(file)

    # Sort frames by file path
    transforms["frames"] = sorted(transforms["frames"], key=lambda x: x["file_path"])

    # Calculate distance between frames
    distances = []
    for i in range(len(transforms["frames"]) - 1):
        first = np.array(transforms["frames"][i]["transform_matrix"])[:3, 3]
        second = np.array(transforms["frames"][i + 1]["transform_matrix"])[:3, 3]
        distances.append(np.linalg.norm(second - first))

    # Interpolate distances
    def interpolate_distances(distances):
        interpolated = []
        for i in range(len(distances) - 1):
            interpolated.append(distances[i])
            interpolated.extend(np.linspace(distances[i], distances[i + 1], num=6, endpoint=False)[1:])
        interpolated.append(distances[-1])
        return interpolated

    distances_inerpolated = interpolate_distances(distances)

    # Reduce number of frames
    def optimize_dist(dist, distances):
        selected_frames = []
        current_distance = 0

        for idx, distance in enumerate(distances):
            current_distance += distance
            if current_distance >= dist:
                selected_frames.append(idx)
                current_distance = 0

        return abs(len(selected_frames) - num_frames)

    result = minimize_scalar(optimize_dist, args=(distances_inerpolated))
    dist = result.x

    # Recalculate selected_frames with optimized dist_before
    selected_frames = []
    current_distance = 0

    for idx, distance in enumerate(distances_inerpolated):
        current_distance += distance
        if current_distance >= dist:
            selected_frames.append(idx)
            current_distance = 0

    CONSOLE.print(f"Selected {len(selected_frames)} frames out of {len(distances_inerpolated)}")

    return selected_frames

def extract_images(config: NeoDataConfig, skip_frames: int = 6, select_frames: list = True, num_frames: int = 250) -> None:
    """Main function."""

    output_path = config.get_base_dir()

    # Need both before and after to be present in order to perform final alignment
    for stage in ["before", "after"]:

        frame_list = None
        if select_frames:
            frame_list = select_frames_to_extract(config, stage=stage, num_frames=num_frames)

        bag_file_path = Path(f"{str(output_path).replace(str(config.root_input), f'{str(config.root_input)}/raw_bagfiles')}_{stage}.bag")
        bag2_file_path = Path(f"{str(output_path).replace(str(config.root_input), f'{str(config.root_input)}/raw_bagfiles')}_{stage}/{config.object_name}_{stage}_0.db3")
        mp4_file_path = Path(f"{str(output_path).replace(str(config.root_input), f'{str(config.root_input)}/raw_bagfiles')}_{stage}.mp4")

        if bag_file_path.exists():
            CONSOLE.print("Bag file found, processing...")
            
            read_realsense_bag(bag_file_path, output_path / stage, True, skip_frames=skip_frames, frame_list=frame_list)

        elif bag2_file_path.exists():
            CONSOLE.print("Bag2 file found, processing...")

            read_ros2_bag(bag2_file_path.parent, output_path / stage, True, skip_frames=skip_frames, frame_list=frame_list)
            
        elif mp4_file_path.exists():
            CONSOLE.print("MP4 file found, processing...")

            read_mp4_file(mp4_file_path, output_path / stage, skip_frames=skip_frames, frame_list=frame_list)

        else:
            raise FileNotFoundError(f"No bag or mp4 file found.")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")

    extract_images(
        tyro.cli(
            NeoDataConfig,
            description=convert_markup_to_ansi(__doc__),
        )
    )

if __name__ == "__main__":
    entrypoint()