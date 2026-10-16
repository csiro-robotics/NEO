#!/bin/bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "NERFILLER EXPORTS"
echo "${ROOT_DIR}/submodules/neo_nerfiller/ZoeDepth/zoedepth"
export PYTHONPATH="${PYTHONPATH}:${ROOT_DIR}/submodules/neo_nerfiller/ZoeDepth/zoedepth"
export PYTHONPATH="${PYTHONPATH}:${ROOT_DIR}/submodules/neo_nerfiller/ZoeDepth"
export PYTHONPATH="${PYTHONPATH}:${ROOT_DIR}/submodules/neo_nerfiller/MiDaS"
export PYTHONPATH="${PYTHONPATH}:${ROOT_DIR}/submodules/neo_nerfiller/MiDaS/midas"