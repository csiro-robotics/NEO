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


def extract_arm_movement(config: NeoDataConfig) -> None:
    """Main function."""

    output_path = config.get_base_dir()
    bag_file_path = Path(f"{str(output_path).replace(str(config.root_input), f'{str(config.root_input)}/raw_bagfiles')}_ros_topics.bag")

    if config.action == "remove":
        CONSOLE.print(f"[bold yellow]Remove action, exiting arm movement extraction![/bold yellow]")
        return

    if os.path.exists(bag_file_path):
        CONSOLE.print(f"Loading .bag file: {bag_file_path}")
        b = bagreader(bag_file_path)
    else:
        CONSOLE.print(f"File '{bag_file_path}' does not exist. Exiting program.")
        return

    # Read arm data topic
    arm_topics = "/trolley/panda/armer/state"
    data = b.message_by_topic(arm_topics)
    df = pd.read_csv(data)

    # Read gripper data topic
    gripper_topic = "/trolley/panda/gripper_controller/joint_states"
    data = b.message_by_topic(gripper_topic)
    df_gripper = pd.read_csv(data)

    # Load end efector position and orientation into dataframe
    column_names = [f"ee_pose.pose.position.{x}" for x in ['x', 'y', 'z']]
    column_names.extend([f"ee_pose.pose.orientation.{x}" for x in ['x', 'y', 'z', 'w']])
    df[column_names] = df[column_names].round(3)
    column_names.extend(['Time'])
    df = df[column_names]

    # Load end gripper opening data
    column_names = [f"position_{x}" for x in range(2)]
    df_gripper[column_names] = df_gripper[column_names].round(3)
    column_names.extend(['Time'])
    df_gripper = df_gripper[column_names]

    # Select timestamps when gripper was closed
    df_gripper['position_0'] = np.abs(df_gripper['position_0'] - df_gripper['position_0'][0]).round(3)
    gripper_closed_data = df_gripper[df_gripper['position_0'] == df_gripper[df_gripper['position_0'] > 0]['position_0'].value_counts().idxmax()]['Time']

    time_closing = gripper_closed_data.iloc[0]
    time_opening = gripper_closed_data.iloc[-1]

    arm_movement = df[np.logical_and(df['Time'] >= time_closing, df['Time'] <= time_opening)]
    movement_start = arm_movement.iloc[0].round(3)
    movement_end = arm_movement.iloc[-1].round(3)

    quat_start = Rotation.from_quat(movement_start[['ee_pose.pose.orientation.x', 'ee_pose.pose.orientation.y', 'ee_pose.pose.orientation.z', 'ee_pose.pose.orientation.w']].to_numpy())
    quat_end = Rotation.from_quat(movement_end[['ee_pose.pose.orientation.x', 'ee_pose.pose.orientation.y', 'ee_pose.pose.orientation.z', 'ee_pose.pose.orientation.w']].to_numpy())
    
    # Compute final translation and rotation 
    total_travel = ((np.round((movement_end - movement_start)[['ee_pose.pose.position.x', 'ee_pose.pose.position.y', 'ee_pose.pose.position.z']], 3)) * np.array([-1, -1, 1])).tolist()
    total_rotation = quat_start * quat_end.inv()
    total_rotation = np.round(total_rotation.as_euler('xyz', degrees=True), 3).tolist()

    CONSOLE.print(f"TRANSLATION: {total_travel} m")
    CONSOLE.print(f"ROTATION: {total_rotation} deg")

    # Save data to file
    data = {'translation': total_travel, 'rotation_euler_deg': total_rotation}
    with open(output_path / 'object_reconfiguration.json', 'w') as f:
        json.dump(data, f, indent=4)

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")

    extract_arm_movement(
        tyro.cli(
            NeoDataConfig,
            description=convert_markup_to_ansi(__doc__),
        )
    )

if __name__ == "__main__":
    entrypoint()