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

Each prepared scene follows this structure :

```text
<neo_data_root>/processed/<action>/<scene_name>/<object>/
├── before/{images,depth,transforms.json}
├── after/{images,depth,transforms.json}
└── object_reconfiguration.json
```

`remove` scenes contain registered captures before and after physical removal and do not include a `movement.txt` file.
`reconfigure` scenes additionally include the `object_reconfiguration.json` file. 
Depth images are stored in mms and are converted to metres to match scale used for the underlying NF scene.

## Raw Capture Naming

Unless `raw_bagfiles_root` is overridden, raw files are resolved beneath
`<neo_data_root>/raw_bagfiles` using the same relative scene path. Supported input
forms include:

```
<scene-path>_before.bag
<scene-path>_after.bag
<scene-path>_before.mp4
<scene-path>_after.mp4
<scene-path>_before/<object>_before_0.db3
<scene-path>_after/<object>_after_0.db3
<scene-path>_ros_topics.bag
```

## Registration

The `before` recording captures the scene before manipulation. The robot then
moves or removes the object, and the `after` recording captures the changed
scene along the same camera trajectory.

Data preparation extracts frames, runs `ns-process-data images`, and uses the
four table-mounted ArUco markers to align both captures at a common scale and
in the same coordinate system. Keep `raw_transforms.json` alongside each
registered `transforms.json` to preserve the original camera poses.

Before publication, document camera calibration, marker size, capture frame
selection, object-motion convention, and the license of every YCB-derived
asset included in the archive.
