# NEO-Dataset

NEO-Dataset contains twelve registered tabletop scenes: eleven single-object
scenes used by removal and relocation experiments and one multi-object assembly
scene. Scenes were captured with Franka Emika Panda and UR5 setups under
controlled illumination. Four table-mounted ArUco markers align the before and
after captures; marker images are not used for NeRF training.

## Public Release Assets

The dataset archive URL, checksum manifest, terms of use, and paper checkpoints
are not present in this repository. Add them here before creating the public
release. A release archive should contain raw captures, prepared Nerfstudio
datasets, movement transforms, bounding-box annotations, train/evaluation
splits, and a machine-readable checksum file.

## Prepared Scene

Each prepared scene follows this structure:

```text
data/real/<action>/single/<scene>/<object>/
├── before/{images,depth,transforms.json}
├── after/{images,depth,transforms.json}
└── movement.json or movement.txt
```

`remove` scenes contain registered captures before and after physical removal.
`translate_and_rotate` scenes additionally provide the known rigid 6-DoF
object transform. Depth images use millimetres on disk; evaluation converts
them with a scale of `0.001`.

## Raw Capture Naming

Unless `raw_bagfiles_root` is overridden, raw files are resolved beneath
`<root_input>/raw_bagfiles` using the same relative scene path. Supported input
forms include:

```text
<scene-path>_before.bag
<scene-path>_after.bag
<scene-path>_before.mp4
<scene-path>_after.mp4
<scene-path>_before/<object>_before_0.db3
<scene-path>_after/<object>_after_0.db3
<scene-path>_ros_topics.bag
```

## Registration

Data preparation extracts frames, runs `ns-process-data images`, detects ArUco
markers, and applies a common scale and world frame. Keep `raw_transforms.json`
alongside each registered `transforms.json` so registration can be audited.

Before publication, document camera calibration, marker size, capture frame
selection, object-motion convention, and the license of every YCB-derived
asset included in the archive.
