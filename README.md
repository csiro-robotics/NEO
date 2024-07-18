# Neo: NeRF it once, edit twice

Fast editing pipeline for nerfacto models

Register the method within Nerfstudio

```bash
cd Neo/
pip install -e .
ns-install-cli
```

```bash
neo --data data/bouquet --timestamp "" --pipeline.datamanager.pixel-sampler.ignore-mask True
```

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install transformers diffusers accelerate
```

If running inside docker container or remote server you need to run
```bash
sudo apt-get install -y xvfb
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
```