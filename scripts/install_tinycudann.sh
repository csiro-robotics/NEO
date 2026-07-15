#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${ROOT_DIR}/submodules/tinycudann/bindings/torch"

cleanup() {
    rm -rf \
        "${SOURCE_DIR}/build" \
        "${SOURCE_DIR}/tinycudann.egg-info" \
        "${SOURCE_DIR}/tinycudann/__pycache__" \
        "${SOURCE_DIR}/dependencies" \
        "${SOURCE_DIR}/src"
}
trap cleanup EXIT

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "Activate the neo Conda environment before building tiny-cuda-nn." >&2
    exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc was not found. Create or update the environment from environment.yml." >&2
    exit 1
fi

TCNN_CUDA_ARCHITECTURES="$(python - <<'PY'
import sys

import torch

if not torch.cuda.is_available():
    sys.exit("CUDA is not visible. Run this installer on a machine with an NVIDIA GPU.")

architectures = sorted(
    {
        major * 10 + minor
        for major, minor in (
            torch.cuda.get_device_capability(index)
            for index in range(torch.cuda.device_count())
        )
    }
)
print(";".join(str(architecture) for architecture in architectures))
PY
)"

export TCNN_CUDA_ARCHITECTURES
export CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX}}"
export MAX_JOBS="${MAX_JOBS:-1}"
export CC="${CC:-${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc}"
export CXX="${CXX:-${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++}"
export LIBRARY_PATH="${CONDA_PREFIX}/lib/stubs${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LDFLAGS="-L${CONDA_PREFIX}/lib/stubs ${LDFLAGS:-}"

echo "Building tiny-cuda-nn for compute capabilities ${TCNN_CUDA_ARCHITECTURES}."
cd "${SOURCE_DIR}"
python -m pip uninstall --yes tinycudann >/dev/null 2>&1 || true
python -m pip install --no-build-isolation --no-deps --force-reinstall .
python -c "import tinycudann; print('tiny-cuda-nn import OK')"
