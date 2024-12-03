#!/bin/bash

# Default values for arguments
data_type=""
scene_name=""
action=""
scene_type=""
object_name=""
root_input=""

# Parse the command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data_type) data_type="$2"; shift ;;
        --scene_name) scene_name="$2"; shift ;;
        --action) action="$2"; shift ;;
        --scene_type) scene_type="$2"; shift ;;
        --object_name) object_name="$2"; shift ;;
        --root_input) root_input="$2"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Directory containing the folders
base_dir=outputs/"$data_type"/"$action"/"$scene_type"/"$scene_name"/"$object_name"/before/lerf

# Find all timestamped directories within the base directory
timestamp_dirs=$(find "$base_dir" -type d -regextype posix-egrep -regex '.*[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{6}')

# If no timestamp directories are found, exit
if [[ -z "$timestamp_dirs" ]]; then
  echo "No timestamp directories found in $base_dir"
  exit 0
fi

# Sort directories based on timestamp in folder names (YYYY-MM-DD_HHMMSS)
newest_dir_before=$(echo "$timestamp_dirs" | sort -t'_' -k1,1 -k2,2 -r | head -n 1)

echo ns-export pointcloud \
    --load-config "$newest_dir_before"/config.yml \
    --output-dir "${newest_dir_before//outputs/exports}" \
    --num-points 1000000 \
    --remove-outliers True \
    --normal-method open3d \
    --save-world-frame True  \
    --obb_center 0.2700000107 0.2800000012 0.1099999994 \
    --obb_rotation 0.0000000000 0.0000000000 0.0000000000 \
    --obb_scale 0.7500000000 0.7799999714 0.4000000060

# Directory containing the folders
base_dir=outputs/"$data_type"/"$action"/"$scene_type"/"$scene_name"/"$object_name"/after/nerfacto

# Find all timestamped directories within the base directory
timestamp_dirs=$(find "$base_dir" -type d -regextype posix-egrep -regex '.*[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{6}')

# If no timestamp directories are found, exit
if [[ -z "$timestamp_dirs" ]]; then
  echo "No timestamp directories found in $base_dir"
  exit 0
fi

# Sort directories based on timestamp in folder names (YYYY-MM-DD_HHMMSS)
newest_dir_after=$(echo "$timestamp_dirs" | sort -t'_' -k1,1 -k2,2 -r | head -n 1)

echo ns-export pointcloud \
    --load-config "$newest_dir_after"/config.yml \
    --output-dir "${newest_dir_after//outputs/exports}" \
    --num-points 1000000 \
    --remove-outliers True \
    --normal-method open3d \
    --save-world-frame True  \
    --obb_center 0.2700000107 0.2800000012 0.1099999994 \
    --obb_rotation 0.0000000000 0.0000000000 0.0000000000 \
    --obb_scale 0.7500000000 0.7799999714 0.4000000060

# Evaluate
python3 neo/eval/eval_arucos.py \
    --path_to_pointcloud_before "${newest_dir_before//outputs/exports}"/point_cloud.ply \
    --path_to_pointcloud_after "${newest_dir_after//outputs/exports}"/point_cloud.ply