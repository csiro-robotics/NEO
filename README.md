
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

## Install nerfstudio with conda using CUDA 12.4
```bash
conda create --name nerfstudio -y python=3.10
conda activate nerfstudio
pip3 install torch torchvision torchaudio
conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit
conda install ninja git
export CUDA_HOME=$HOME/.conda/envs/nerfstudio
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
ln -s /usr/lib64/libcuda.so.565.77 ~/.conda/envs/nerfstudio/lib/libcuda.so

git clone https://github.com/nerfstudio-project/nerfstudio.git
cd nerfstudio
pip install --upgrade pip setuptools
pip install -e .
```


## Using pretrained Nerfacto-based models

If you want to use different Nerfacto based model just convert the config file with command
```bash
ns-convert --input CONFIG_INPUT_PATH
```
Converted config will be saved at previous output location in neo folder

## Loopback edited model

In order to load edited lerf model use the command:

```bash
ns-loopback lerf --input <path_to_folder_containing__replace_nerf_config_file> [--verbose bool]
```

## Time logging

For time logging you will need to install

```bash
sudo apt-get install bc jq
```

## Running our experiments

First step is to prepare default models for further editing. This step will load images, calcualte colmap for ground truth and basic dataset. It will also load the movement from robot rosbag file is the file is provided. During this step basic LERF model is trainied and ground truth nerfacto model.

```bash
./scripts/prepare_data.sh --root_input <data_directory> --data_type <data_type> --scene_name <scene_name> --action <action> --scene_type <scene_type> --object_name <object_name> [--skip_colmap]
```

Arguments are:
- `--root_input`: The root directory where input data is stored.
- `--data_type`: The type of data being used (e.g., real, synthetic).
- `--scene_name`: The name of the scene to be processed.
- `--action`: The action to be performed (e.g., remove, translate).
- `--scene_type`: The type of scene (e.g., single, multiple_one).
- `--object_name`: The name of the object to be edited.
- `--skip_colmap`: Flag to skip the COLMAP step in the pipeline.

Now you can optionally run `lerftogo` command and check how good your language prompt is.

Then we can run editing and evaluation with command.

```bash
./scripts/run_editing.sh --root_input <data_directory> --data_type <data_type> --scene_name <scene_name> --action <action> --scene_type <scene_type> --object_name <object_name> --lerf_prompt '<lerf_prompt>' [--skip_colmap] [--dry_run]
```

> **Note:** Make sure that `lerfprompt` is enclosed within single quotes ('').

Arguments are:
- Same as before +
- `--lerf_prompt`: The prompt for the LERF model to identify the object.
- `--dry_run`: Flag to skip the rendering of images in evaluation step.

Last step is to update langiage field with the command

```bash
ns-loopback lerf --input <data_directory>/<data_type>/<action>/<scene_type>/<scene_name>/<object_name>/befoe/replace-nerf/<timestamp> [--verbose bool]
```
Now this new field will be saved under the same lerf-lite directory under different timestamp