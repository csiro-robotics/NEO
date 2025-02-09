#!/bin/bash

# ./scripts/run_editing.sh --root_input data \
#                          --data_type real \
#                          --scene_name mustard_bottle \
#                          --action remove \
#                          --scene_type single \
#                          --object_name mustard_bottle \
#                          --lerf_prompt 'Yellow mustard bottle' \
#                          --skip_colmap

# ./scripts/run_editing.sh --root_input data \
#                          --data_type real \
#                          --scene_name orange \
#                          --action remove \
#                          --scene_type single \
#                          --object_name orange \
#                          --lerf_prompt 'Orange' \
#                          --skip_colmap

# ./scripts/run_editing.sh --root_input data \
#                          --data_type real \
#                          --scene_name red_robot \
#                          --action remove \
#                          --scene_type single \
#                          --object_name red_robot \
#                          --lerf_prompt 'Red robot' \
#                          --skip_colmap

# ./scripts/run_editing.sh --root_input data \
#                          --data_type real \
#                          --scene_name spam \
#                          --action remove \
#                          --scene_type single \
#                          --object_name spam \
#                          --lerf_prompt 'Spam' \
#                          --skip_colmap

# ./scripts/run_editing.sh --root_input data \
#                          --data_type real \
#                          --scene_name tomato_soup_can \
#                          --action remove \
#                          --scene_type single \
#                          --object_name tomato_soup_can \
#                          --lerf_prompt 'Tomato soup can' \
#                          --skip_colmap

# ./scripts/run_editing.sh --root_input data \
#                          --data_type real \
#                          --scene_name mustard_bottle \
#                          --action translate_and_rotate \
#                          --scene_type single \
#                          --object_name mustard_bottle \
#                          --lerf_prompt 'Yellow mustard bottle'

./scripts/run_editing.sh --root_input data \
                         --data_type real \
                         --scene_name red_robot \
                         --action translate_and_rotate \
                         --scene_type single \
                         --object_name red_robot \
                         --lerf_prompt 'Red robot'