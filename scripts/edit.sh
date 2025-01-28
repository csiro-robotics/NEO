#!/bin/bash

set -e  # Exit the script if any command fails

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

# Import the time logging function
source scripts/sub_scripts/log_time.sh
log_file=outputs/"$data_type"/"$action"/"$scene_type"/"$scene_name"/"$object_name"/times.json

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

log_time "$log_file" "Neo_edit" \
    python3 neo/scripts/apply_neo_edit.py \
        --load_config "$newest_dir"/config.yml