#!/bin/bash

# Default values for arguments
data_type=""
scene_name=""
action=""
scene_type=""
object_name=""
root_output=""
skip_colmap=""

# Parse the command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data_type) data_type="$2"; shift ;;
        --scene_name) scene_name="$2"; shift ;;
        --action) action="$2"; shift ;;
        --scene_type) scene_type="$2"; shift ;;
        --object_name) object_name="$2"; shift ;;
        --root_output) root_output="$2"; shift ;;
        --skip_colmap) skip_colmap=true ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Run image extraction, COLMAP and scene matching
python3 data_processing/process_bags.py \
    --data_type "$data_type" \
    --scene_name "$scene_name" \
    --action "$action" \
    --scene_type "$scene_type" \
    --object_name "$object_name" \
    --root_output "$root_output" \
    ${skip_colmap:+--skip_colmap}  # Only include --skip_colmap if it's true

# Extract arm movement if applicable
python3 data_processing/extract_arm_movement.py \
    --data_type "$data_type" \
    --scene_name "$scene_name" \
    --action "$action" \
    --scene_type "$scene_type" \
    --object_name "$object_name" \
    --root_output "$root_output"