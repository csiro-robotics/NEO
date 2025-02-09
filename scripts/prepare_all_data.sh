#!/bin/bash

# ./scripts/prepare_data.sh --root_input data \
#                           --data_type real \
#                           --scene_name mustard_bottle \
#                           --action remove \
#                           --scene_type single \
#                           --object_name mustard_bottle \
#                           --skip_colmap

# ./scripts/prepare_data.sh --root_input data \
#                           --data_type real \
#                           --scene_name orange \
#                           --action remove \
#                           --scene_type single \
#                           --object_name orange \
#                           --skip_colmap

# ./scripts/prepare_data.sh --root_input data \
#                           --data_type real \
#                           --scene_name red_robot \
#                           --action remove \
#                           --scene_type single \
#                           --object_name red_robot \
#                           --skip_colmap

# ./scripts/prepare_data.sh --root_input data \
#                           --data_type real \
#                           --scene_name spam \
#                           --action remove \
#                           --scene_type single \
#                           --object_name spam \
#                           --skip_colmap

# ./scripts/prepare_data.sh --root_input data \
#                           --data_type real \
#                           --scene_name tomato_soup_can \
#                           --action remove \
#                           --scene_type single \
#                           --object_name tomato_soup_can \
#                           --skip_colmap

# ./scripts/prepare_data.sh --root_input data \
#                           --data_type real \
#                           --scene_name mustard_bottle \
#                           --action translate_and_rotate \
#                           --scene_type single \
#                           --object_name mustard_bottle

./scripts/prepare_data.sh --root_input data \
                            --data_type real \
                            --scene_name red_robot \
                            --action translate_and_rotate \
                            --scene_type single \
                            --object_name red_robot \
                            --skip_colmap