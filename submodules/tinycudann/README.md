# Vendored tiny-cuda-nn build sources

This directory contains the minimal source set required to build the
tiny-cuda-nn 1.7 PyTorch extension locally. It was copied from upstream commit
`075158a70b87dba8729188a9cadc9411cfa4b71d` and is not a Git submodule.

Vendored build dependencies:

- CUTLASS commit `1eb6355182a5124639ce9d3ff165732a94ed9a70`
- fmt commit `b0c8263cb26ea178d3a5df1b984e1a61ef578950`

Documentation, examples, tests, and unrelated CMake dependencies were omitted.
The original licenses are retained. No compiled Python or CUDA extension is
stored here. Run `scripts/install_tinycudann.sh` from an activated NEO
environment to detect the local GPU architecture, compile the extension, and
install it into that environment.

Upstream: <https://github.com/NVlabs/tiny-cuda-nn>
