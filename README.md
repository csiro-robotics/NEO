# Neo: NeRF it once, edit twice

Fast editing pipeline for nerfacto models

Register the method within Nerfstudio

```bash
cd Neo/
pip install -e .
ns-install-cli
```

You have to download [SAM ViT-H](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) model weights and put them in ``ckpts`` folder. So the folder structure should look like this

```
Neo/
├── ckpts/
│ └── sam_vit_h_4b8939.pth
├── neo/
│ └── ...
├── pyproject.toml
└── README.md
```

## Using pretrained Nerfacto-based models

If you want to use different Nerfacto based model just convert the config file with command
```bash
ns-convert --input CONFIG_INPUT_PATH
```
Converted config will be saved at previous output location in neo folder

## Running Editing Pipeline

To run all parts of the editing pipeline just run the command as follows

```bash
./scripts/run_all.sh --root_input <data_directory> --data_type <data_type> --scene_name <scene_name> --action <action> --scene_type <scene_type> --object_name <object_name> --lerf_prompt '<lerf_prompt>' [--skip_colmap] [--dry_run]
```

> **Note:** Make sure that `lerfprompt` is enclosed within single quotes ('').

Arguments are:
- `--root_input`: The root directory where input data is stored.
- `--data_type`: The type of data being used (e.g., real, synthetic).
- `--scene_name`: The name of the scene to be processed.
- `--action`: The action to be performed (e.g., remove, translate).
- `--scene_type`: The type of scene (e.g., single, multiple_one).
- `--object_name`: The name of the object to be edited.
- `--lerf_prompt`: The prompt for the LERF model to identify the object.
- `--skip_colmap`: Flag to skip the COLMAP step in the pipeline.
- `--dry_run`: Flag to skip the rendering of images in evaluation step.

## Loopback edited model

In order to load edited lerf model use the command:

```bash
ns-loopback lerf --input <path_to_folder_containing_config_file>
```