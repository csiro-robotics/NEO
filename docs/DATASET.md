# NEO-Dataset

NEO-Dataset contains twelve registered tabletop scenes: eleven single-object
scenes used by removal and relocation experiments and one multi-object assembly
scene. Scenes were captured with Franka Emika Panda and UR5 setups under
controlled illumination. Four table-mounted ArUco markers align the before and
after captures; marker images are not used for NeRF training.

## Dataset structure
Full dataset structure under the root Neo dataset folder (`<neo_data_root>`) is outlined in the figure below

<p align="center">
   <img src=figs/file_structure.png width=1280>
</p>

## Definitions

Term | Definition
:----|:----------
`<scene>` | Name of the tabletop scene
`<object>` | Name of object being manipulated
`object_reconfiguration.json` | JSON file containing translation and Euler rotations within the tabletop coordinate frame for the object being reconfigured
`transforms.json` | Camera poses in the tabletop coordinate frame for processed data. Stored in nerfstudio-compatible format.
`raw_transforms.json` | Initial COLMAP camera poses before conversion to tabletop co-ordinates. Stored in nerfstudio-compatible format.
`depth` | Folder of extracted depth images (640x480) captured by RGB-D sensor. Images are saved in 16-bit PNG format with pixel values representing depth in mms
`images` | Folder of extracted colour images (640x480) captured by RGB-D sensor
`movement.txt` | Movement file saved during data capture describing an object's translation along the XYZ axis and rotation about the Z axis
`<object>_<before/after>.bag` | ROS1 bagfile containing camera data for the image scan of the scene before/after object manipulation has occurred
`<object>_ros_topics.bag` | ROS1 bagfile containingall topics recorded during object manipulation process
`<object>_<before/after>_0.db3` | ROS2 bagfile containing data for the image scan of the scene before/after manipulation of `<object>` has occurred ("assembly" for base assembly scene)
`metadata.yaml` | Metadata pertaining to the ROS2 bagfile's contents

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
