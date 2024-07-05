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