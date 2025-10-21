import os
import json
import tyro
import numpy as np
import pandas as pd

from bagpy import bagreader
from scipy.spatial.transform import Rotation
from pathlib import Path

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE

from experiment_config import NeoDataConfig


def load_movement_from_manual(manual_file: Path) -> dict | None:
    """Loads movement data from a manual movement.txt file."""
    if not manual_file.exists():
        CONSOLE.print(f"[bold red]Fallback file 'movement.txt' not found. Exiting.[/bold red]")
        return None

    with open(manual_file, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    translation = [0.0, 0.0, 0.0]
    rotation_euler_deg = [0.0, 0.0, 0.0]

    section = None
    for line in lines:
        if "Translation:" in line:
            section = "translation"
        elif "Rotation:" in line:
            section = "rotation"
        elif ':' in line:
            key, value = line.split(':')
            value = float(value.strip())
            key = key.strip().lower()
            if section == "translation":
                if key == "x":
                    translation[0] = np.round(value / 1000, 3)
                elif key == "y":
                    translation[1] = np.round(value / 1000, 3)
                elif key == "z":
                    translation[2] = np.round(value / 1000, 3)
            elif section == "rotation":
                if key == "z":
                    rotation_euler_deg[2] = value  # Only Z axis assumed

    data = {
        'translation': translation,
        'rotation_euler_deg': rotation_euler_deg
    }

    return data


def load_movement_from_bag(bag_file_path: Path) -> dict | None:
    """Processes a .bag file to compute end-effector movement."""
    try:
        b = bagreader(bag_file_path)
    except Exception as e:
        CONSOLE.print(f"[bold red]Failed to load bag file: {e}[/bold red]")
        return None

    # Read arm data
    arm_topic = "/trolley/panda/armer/state"
    df = pd.read_csv(b.message_by_topic(arm_topic))

    # Read gripper data
    gripper_topic = "/trolley/panda/gripper_controller/joint_states"
    df_gripper = pd.read_csv(b.message_by_topic(gripper_topic))

    # Process arm data
    ee_columns = [f"ee_pose.pose.position.{x}" for x in 'xyz']
    ee_columns += [f"ee_pose.pose.orientation.{x}" for x in 'xyzw']
    df[ee_columns] = df[ee_columns].round(3)
    df = df[ee_columns + ['Time']]

    # Process gripper data
    gripper_columns = [f"position_{i}" for i in range(2)]
    df_gripper[gripper_columns] = df_gripper[gripper_columns].round(3)
    df_gripper = df_gripper[gripper_columns + ['Time']]

    # Detect grip closure period
    df_gripper['position_0'] = np.abs(df_gripper['position_0'] - df_gripper['position_0'][0]).round(3)
    closed_times = df_gripper[df_gripper['position_0'] == df_gripper[df_gripper['position_0'] > 0]['position_0'].value_counts().idxmax()]['Time']

    if closed_times.empty:
        CONSOLE.print("[bold red]No gripper closure detected. Exiting.[/bold red]")
        return None

    t_start, t_end = closed_times.iloc[0], closed_times.iloc[-1]
    arm_movement = df[(df['Time'] >= t_start) & (df['Time'] <= t_end)]

    if arm_movement.empty:
        CONSOLE.print("[bold red]No arm movement detected in the time window. Exiting.[/bold red]")
        return None

    pose_start = arm_movement.iloc[0]
    pose_end = arm_movement.iloc[-1]

    quat_start = Rotation.from_quat(pose_start[['ee_pose.pose.orientation.x', 'ee_pose.pose.orientation.y', 'ee_pose.pose.orientation.z', 'ee_pose.pose.orientation.w']].to_numpy())
    quat_end = Rotation.from_quat(pose_end[['ee_pose.pose.orientation.x', 'ee_pose.pose.orientation.y', 'ee_pose.pose.orientation.z', 'ee_pose.pose.orientation.w']].to_numpy())

    total_translation = ((pose_end[['ee_pose.pose.position.x', 'ee_pose.pose.position.y', 'ee_pose.pose.position.z']] -
                          pose_start[['ee_pose.pose.position.x', 'ee_pose.pose.position.y', 'ee_pose.pose.position.z']]).round(3)
                         * np.array([-1, -1, 1])).tolist()

    total_rotation = (quat_end * quat_start.inv()).as_euler('xyz', degrees=True)
    total_rotation = np.round(total_rotation, 3).tolist()

    CONSOLE.print(f"TRANSLATION: {total_translation} m")
    CONSOLE.print(f"ROTATION: {total_rotation} deg")

    return {
        'translation': total_translation,
        'rotation_euler_deg': total_rotation
    }


def extract_arm_movement(config: NeoDataConfig) -> None:
    """Main function."""

    output_path = config.get_base_dir()
    bag_file_path = Path(f"{str(output_path).replace(str(config.root_input), f'{str(config.root_input)}/raw_bagfiles')}_ros_topics.bag")

    if config.action == "remove":
        CONSOLE.print(f"[bold yellow]Remove action, exiting arm movement extraction![/bold yellow]")
        return

    if bag_file_path.exists():
        CONSOLE.print(f"Loading .bag file: {bag_file_path}")
        result = load_movement_from_bag(bag_file_path)
    else:
        CONSOLE.print(f"[bold red]File '{bag_file_path}' not found. Falling back to manual movement.txt input.[/bold red]")
        manual_file = bag_file_path.parent / 'movement.txt'
        result = load_movement_from_manual(manual_file)

    if result is None:
        return

    # Save output
    with open(output_path / 'object_reconfiguration.json', 'w') as f:
        json.dump(result, f, indent=4)

    CONSOLE.print(f"[bold green]Movement saved to object_reconfiguration.json[/bold green]")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    extract_arm_movement(
        tyro.cli(
            NeoDataConfig,
            description=convert_markup_to_ansi(__doc__),
        )
    )


if __name__ == "__main__":
    entrypoint()
