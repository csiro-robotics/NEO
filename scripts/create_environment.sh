#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

conda env create -f "${ROOT_DIR}/environment.yml"
cd ${ROOT_DIR}/submodules/neo_lerftogo/robot_lerf/graspnet_baseline/pointnet2/
conda run -n neo python setup.py build_ext --inplace
cd -
cd ${ROOT_DIR}/submodules/neo_lerftogo/robot_lerf/graspnet_baseline/knn/
conda run -n neo python setup.py build_ext --inplace
cd -
conda run -n neo bash "${ROOT_DIR}/scripts/install_tinycudann.sh"
