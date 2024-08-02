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
├── tests/
├── pyproject.toml
└── README.md
```

If you are running Neo inside docker container or remote server you need to run
```bash
sudo apt-get install -y xvfb
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
```