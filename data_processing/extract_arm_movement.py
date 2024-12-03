import bagpy
import os
import sys
import json
import numpy as np
import pandas as pd

from bagpy import bagreader
from scipy.spatial.transform import Rotation
from argparse import ArgumentParser
from pathlib import Path

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--data_type", type=str, choices=["real", "simulated"],
                    help="specify if data is real or simulated")
    parser.add_argument("--scene_name", "-n", type=str, 
                        help="name of scene. Used in naming all related files")
    parser.add_argument("--root_bag", default="data/raw_bagfiles",
                        help="absolute path of root location of bag files")
    parser.add_argument("--skip_frames", default=6, type=int, 
                        help="number of frames to be skipped from original video")
    parser.add_argument("--action", choices=["remove", "translate", 
                                        "rotate", "translate_and_rotate",
                                        "mix"])
    parser.add_argument("--scene_type", choices=["single", "multiple_one", 
                                                "multiple_sequential", 
                                                "multiple_multiaction"])
    parser.add_argument("--sequence_num", type=int)
    parser.add_argument("--object_name", type=str, nargs="+")
    parser.add_argument("--root_output", default="data/neo_dataset",
                        help="absolute path of output for dataset")
    args=parser.parse_args()

    action_type = args.action
    scene_name = args.scene_name if args.scene_type not in ["multiple_sequential", "multiple_multiaction"] \
        else f"{args.scene_name}-{args.sequence_num:04}"
    if args.scene_type == "single":
        object_name = args.scene_name
    elif not args.object_name:
        raise ValueError("Cannot have action on multiple object scene without defining object being acted upon")
    else:
        object_name = "-".join([obj for obj in args.object_name])
    output_path = Path(args.root_output) / args.data_type / args.action / args.scene_type / scene_name / object_name / "after"
    os.makedirs(output_path, exist_ok=True)

    bag_file_path = os.path.join(args.root_bag, action_type, args.scene_type, f"{scene_name}_ros_topics.bag")

    print(f"BAG FILE: {bag_file_path}")
    print(f"OUTPUT PATH: {output_path}")

    if args.action == "remove":
        print(f"Remove action, exiting arm movement extraction!")
        sys.exit()

    if os.path.exists(bag_file_path):
        print(f"Loading .bag file: {bag_file_path}")
        b = bagreader(bag_file_path)
    else:
        print(f"File '{bag_file_path}' does not exist. Exiting program.")
        sys.exit()

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
    total_travel = (np.round(np.abs(movement_end - movement_start)[['ee_pose.pose.position.x', 'ee_pose.pose.position.y', 'ee_pose.pose.position.z']], 3)).tolist()

    total_rotation = quat_start * quat_end.inv()
    total_rotation = np.round(total_rotation.as_euler('xyz', degrees=True), 3).tolist()

    print(f"TRANSLATION: {total_travel} m")
    print(f"ROTATION: {total_rotation} deg")

    # Save data to file
    data = {'translation': total_travel, 'rotation_euler_deg': total_rotation}
    with open(output_path / 'object_reconfiguration.json', 'w') as f:
        json.dump(data, f, indent=4)