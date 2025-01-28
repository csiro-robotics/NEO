#!/bin/bash

set -e  # Exit the script if any command fails

# Default values for arguments
root_input=""
data_type=""
scene_name=""
action=""
scene_type=""
object_name=""
lerf_prompt=""
skip_colmap=""
dry_run=""

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
        --skip_colmap) skip_colmap=true ;;
        --dry_run) dry_run=true ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Import the time logging function
source scripts/sub_scripts/log_time.sh
log_file=outputs/"$data_type"/"$action"/"$scene_type"/"$scene_name"/"$object_name"/times.json

# ./scripts/prepare_data.sh --root_input $root_input \
#                           --data_type $data_type \
#                           --scene_name $scene_name \
#                           --action $action \
#                           --scene_type $scene_type \
#                           --object_name $object_name \
#                           ${skip_colmap:+--skip_colmap}  # Only include --skip_colmap if it's true

log_time "$log_file" "All_get_bbox" \
    ./scripts/sub_scripts/get_bbox.sh --root_input $root_input \
                                      --data_type $data_type \
                                      --scene_name $scene_name \
                                      --action $action \
                                      --scene_type $scene_type \
                                      --object_name $object_name \
                                      --lerf_prompt \'"$lerf_prompt"\'

log_time "$log_file" "All_neo_edit" \
    ./scripts/edit.sh --data_type $data_type \
                      --scene_name $scene_name \
                      --action $action \
                      --scene_type $scene_type \
                      --object_name $object_name \
                      --root_input $root_input

log_time "$log_file" "All_nerfiller_training" \
    ./scripts/sub_scripts/train_nerfiller.sh --data_type $data_type \
                                             --scene_name $scene_name \
                                             --action $action \
                                             --scene_type $scene_type \
                                             --object_name $object_name

log_time "$log_file" "All_eval" \
    ./scripts/eval.sh --data_type $data_type \
                      --scene_name $scene_name \
                      --action $action \
                      --scene_type $scene_type \
                      --object_name $object_name \
                      --root_input $root_input \
                      ${dry_run:+--dry_run}  # Only include --dry_run if it's true