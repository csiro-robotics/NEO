# NEO: NeRF It Once, Edit It Many Times

Language-guided, persistent NeRF editing for continuous object manipulation.

NEO starts from a single scene scan and predicts future scene states without
re-scanning after every robot action. It combines:

1. region-aware neural-field resampling for object removal;
2. progressive, multi-view-consistent diffusion inpainting; and
3. teacher-student distillation for persistent object relocation.

The edited NeRF can be converted back to a language field and used as the input
to the next manipulation step.

## Release Status

This branch contains the NEO implementation and paper experiment manifests.
The NEO-Dataset, paper checkpoints, and final license are separate release
artifacts and are not yet included in this repository. See
[Dataset](docs/DATASET.md) and [Reproducibility](docs/REPRODUCIBILITY.md) before
attempting the full paper evaluation.

## Requirements

- Linux and Python 3.10
- NVIDIA GPU with CUDA support
- approximately 24 GB VRAM for `neo`; more memory and time for `neo-big`
- COLMAP and FFmpeg for data preparation and rendering
- Hugging Face access for the configured Stable Diffusion inpainting model

The paper environment used the `gs` conda environment, PyTorch 2.6, CUDA 12.4,
and pinned forks of Nerfstudio, LERF, and NeRFiller. Exact Git revisions are in
the `paper` optional dependency group in `pyproject.toml`.

## Installation

```bash
conda env create -f environment.yml
conda activate gs
ns-install-cli
```

For a manual installation, create a Python 3.10 environment, install a PyTorch
2.6 build appropriate for the host NVIDIA driver, then run
`pip install -e ".[paper,capture]"` and `ns-install-cli`.

Download the Segment Anything ViT-H checkpoint:

```bash
mkdir -p ckpts
wget -O ckpts/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Verify the installation without starting a training run:

```bash
python -c "import neo.neo_config; print('NEO import OK')"
neo-remove --help
neo-reconfigure --help
neo-assembly --help
```

Do not add host-specific `libcuda.so` symlinks to the conda environment. The
installed PyTorch build and NVIDIA driver must be mutually compatible.

## Data Layout

NEO expects registered `before` and `after` Nerfstudio datasets:

```text
data/real/<action>/single/<scene>/<object>/
├── before/
│   ├── images/
│   ├── depth/
│   └── transforms.json
├── after/
│   ├── images/
│   ├── depth/
│   └── transforms.json
└── bbox.json                 # generated during language-guided localization
```

Raw RealSense, ROS 2, MP4, and robot-motion inputs are described in
[docs/DATASET.md](docs/DATASET.md).

## Run NEO

Paper configurations are plain YAML under `experiments/paper`. Paths are
relative to the repository by default and can be changed in the YAML files.

Object removal:

```bash
neo-remove --load-config paper/removal/tomato_soup_can.yaml
```

Object relocation:

```bash
neo-reconfigure --load-config paper/relocation/banana.yaml
```

Four-step assembly:

```bash
neo-assembly --load-config paper/assembly.yaml
```

Each runner exposes individual pipeline switches:

```bash
neo-remove --help
neo-reconfigure --help
neo-assembly --help
```

Outputs mirror the input hierarchy below `outputs/`. A run contains the
initial LERF model, NEO conversion, inpainted sparse-guided NeRF, distilled
`replace-nerf` model, bounding boxes, and logs.

## Custom Scenes

For a prepared Nerfstudio scene, train the initial language field, localize the
object with a text prompt, and launch an edit through the corresponding runner.
The bounding-box viewer supports translation and rotation offsets for rigid
objects. NEO currently assumes a static background and does not model
articulated or deformable objects.

Language localization and language-field loopback require the pinned
`csiro-robotics/lerf_bbox` fork, which provides `lerftogo-neo`, `lerf-lite`, and
`lerf-lite-lan`. NeRFiller baselines require the pinned
`csiro-robotics/neo_nerfiller` fork.

## Paper Experiments

- Object removal: five single-object scenes.
- Pick and place: six single-object relocation scenes.
- Assembly: four consecutive edits from one initial scan.
- NEO-Dataset: twelve tabletop scenes in total, consisting of eleven
  single-object scenes and one multi-object scene.

Evaluation uses PSNR, SSIM, depth RMSE, and photometric reprojection error over
the full image and the edited `Out`/`In` regions. Exact scene lists, metric
defaults, and known paper-to-code discrepancies are documented in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Repository Layout

```text
neo/                 Nerfstudio models, pipelines, rendering, and metrics
scripts/             Installed data, editing, evaluation, and experiment CLIs
experiments/paper/   Portable paper experiment manifests
docs/                Dataset and reproducibility documentation
tests/               CPU-safe release checks
notebooks/           Development and data-inspection notebooks
```

## Citation

See [CITATION.cff](CITATION.cff), or cite:

```bibtex
@article{zielinski2026neo,
  title   = {NEO: NeRF It Once, Edit It Many Times for Continuous Object Manipulation},
  author  = {Zielinski, Mikolaj and Hall, David and Belter, Dominik and Moghadam, Peyman},
  year    = {2026}
}
```

## License

A public software and dataset license has not yet been assigned. Until the
copyright holders add license files, the repository should be treated as
source-available for review rather than an open-source distribution.
