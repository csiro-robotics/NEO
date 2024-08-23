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

If you are running Neo inside docker container or remote server you need to run
```bash
sudo apt-get install -y xvfb
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
```

## Using pretrained Nerfacto-based models

If you want to use different Nerfacto based model just convert the config file with command
```bash
ns-convert --input CONFIG_INPUT_PATH
```
Converted config will be saved at previous output location in neo folder

ns-render camera-path --load-config outputs/figurines/neo/2024-08-04_225712/config.yml --camera-path-filename /workspace/Neo/data/figurines/camera_paths/2024-08-15-02-21-06.json --output-path renders/figurines/2024-08-15-02-21-06.mp4