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

# Train LERF
ns-train lerf \
    --data "$root_input"/"$data_type"/"$action"/"$scene_type"/"$scene_name"/"$object_name"/before \
    --output-dir outputs/"$data_type"/"$action"/"$scene_type"/"$scene_name" \
    --experiment-name "$object_name"/before \
    --viewer.quit-on-train-completion True \
    nerfstudio-data \
    --orientation-method none \
    --center-method none \
    --auto-scale-poses False \
    --train-split-fraction 0.99

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
newest_dir=$(echo "$timestamp_dirs" | sort -t'_' -k1,1 -k2,2 -r | head -n 1)

# Get the last folder from the newest directory
timestamp=$(basename "$newest_dir")

# Train GT scene
ns-train nerfacto \
    --data "$root_input"/"$data_type"/"$action"/"$scene_type"/"$scene_name"/"$object_name"/after \
    --output-dir outputs/"$data_type"/"$action"/"$scene_type"/"$scene_name" \
    --experiment-name "$object_name"/after \
    --timestamp "$timestamp" \
    --viewer.quit-on-train-completion True \
    nerfstudio-data \
    --orientation-method none \
    --center-method none \
    --auto-scale-poses False \
    --train-split-fraction 0.99

# Convert LERF to Neo
ns-convert --input $newest_dir