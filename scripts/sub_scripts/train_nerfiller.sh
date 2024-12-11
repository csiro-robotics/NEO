#!/bin/bash

# Default values for arguments
data_type=""
scene_name=""
action=""
scene_type=""
object_name=""

# Parse the command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data_type) data_type="$2"; shift ;;
        --scene_name) scene_name="$2"; shift ;;
        --action) action="$2"; shift ;;
        --scene_type) scene_type="$2"; shift ;;
        --object_name) object_name="$2"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Directory containing the folders
base_dir=outputs/"$data_type"/"$action"/"$scene_type"/"$scene_name"/"$object_name"/before/neo

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

# Prepare datasets for training
python3 neo/scripts/prepare_data_nerfiller.py \
    --load_config "$base_dir"/"$timestamp"/config.yml

# Train NeRFiller base
ns-train nerfacto-nerfiller \
    --data outputs/"$data_type"/"$action"/"$scene_type"/"$scene_name"/"$object_name"/before/nerfacto-nerfiller/"$timestamp"/cache \
    --output-dir outputs/"$data_type"/"$action"/"$scene_type"/"$scene_name" \
    --experiment-name "$object_name"/before \
    --timestamp "$timestamp" \
    --viewer.quit-on-train-completion True \
    --pipeline.model.camera-optimizer.mode off \
    --pipeline.datamanager.masks-on-gpu True \
    --pipeline.datamanager.images-on-gpu True \
    nerfstudio-data \
    --orientation-method none \
    --center-method none \
    --auto-scale-poses False \
    --train-split-fraction 0.99
    
# Train NeRFiller
ns-train grid-prior-du \
    --data outputs/"$data_type"/"$action"/"$scene_type"/"$scene_name"/"$object_name"/before/nerfacto-nerfiller/"$timestamp"/cache \
    --output-dir outputs/"$data_type"/"$action"/"$scene_type"/"$scene_name" \
    --experiment-name "$object_name"/before \
    --timestamp "$timestamp" \
    --viewer.quit-on-train-completion True \
    --pipeline.model.camera-optimizer.mode off \
    --pipeline.datamanager.masks-on-gpu True \
    --pipeline.datamanager.images-on-gpu True \
    nerfstudio-data \
    --orientation-method none \
    --center-method none \
    --auto-scale-poses False \
    --train-split-fraction 0.99