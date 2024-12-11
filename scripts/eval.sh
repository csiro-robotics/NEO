#!/bin/bash

# Default values for arguments
data_type=""
scene_name=""
action=""
scene_type=""
object_name=""
root_input=""
dry_run=""

# Parse the command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data_type) data_type="$2"; shift ;;
        --scene_name) scene_name="$2"; shift ;;
        --action) action="$2"; shift ;;
        --scene_type) scene_type="$2"; shift ;;
        --object_name) object_name="$2"; shift ;;
        --root_input) root_input="$2"; shift ;;
        --dry_run) dry_run=true ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Directory containing the folders
base_dir=outputs/"$data_type"/"$action"/"$scene_type"/"$scene_name"/"$object_name"/before/replace-nerf

# Find all timestamped directories within the base directory
timestamp_dirs=$(find "$base_dir" -type d -regextype posix-egrep -regex '.*[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{6}')

# If no timestamp directories are found, exit
if [[ -z "$timestamp_dirs" ]]; then
    echo "No timestamp directories found in $base_dir"
    exit 0
fi

# Sort directories based on timestamp in folder names (YYYY-MM-DD_HHMMSS)
newest_dir=$(echo "$timestamp_dirs" | sort -t'_' -k1,1 -k2,2 -r | head -n 1)

# Evaluate Neo
ns-eval-neo --config-path "$newest_dir"/config.yml \
    ${dry_run:+--dry_run True} \
    --verbose True

# Replace "replace-nerf" with "nerfacto-nerfiller" in the newest directory path
selected_dir="${newest_dir/replace-nerf/nerfacto-nerfiller}"

# Evaluate NeRFiller baseline
ns-eval-neo --config-path "$selected_dir"/config.yml \
    ${dry_run:+--dry_run True} \
    --verbose True

# Replace "replace-nerf" with "grid-prior-du" in the newest directory path
selected_dir="${newest_dir/replace-nerf/grid-prior-du}"

# Evaluate NeRFiller baseline
ns-eval-neo --config-path "$selected_dir"/config.yml \
    ${dry_run:+--dry_run True} \
    --verbose True