#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

conda env create -f "${ROOT_DIR}/environment.yml"
conda run -n neo bash "${ROOT_DIR}/scripts/install_tinycudann.sh"
