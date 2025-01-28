#!/bin/bash

set -e  # Exit the script if any command fails

# Default values for arguments
data_type=""
scene_name=""
action=""
scene_type=""
object_name=""
root_input=""
lerf_prompt=""

# Parse the command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --root_input) root_input="$2"; shift ;;
        --data_type) data_type="$2"; shift ;;
        --scene_name) scene_name="$2"; shift ;;
        --action) action="$2"; shift ;;
        --scene_type) scene_type="$2"; shift ;;
        --object_name) object_name="$2"; shift ;;
        --lerf_prompt) lerf_prompt="$2"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Import the time logging function
source scripts/sub_scripts/log_time.sh
log_file=outputs/"$data_type"/"$action"/"$scene_type"/"$scene_name"/"$object_name"/times.json

log_time "$log_file" "Lerftogo_get_bbox" \
    lerftogo-neo --root_input "$root_input" \
                 --data_type "$data_type" \
                 --scene_name "$scene_name" \
                 --action "$action" \
                 --scene_type "$scene_type" \
                 --object_name "$object_name" \
                 --lerf_prompt \'"$lerf_prompt"\'
