import torch
import tyro
import yaml
import os
import shutil

from dataclasses import dataclass, fields
from pathlib import Path

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig

from neo.neo_pipeline import NeoPipelineConfig
from neo.neo_model import NeoModelConfig


@dataclass
class ConvertConfig:
    input: Path
    """Path to the folder containing config.yaml file and nerfstudio_models"""
    # output: Path
    # """Path to the output where converted model will be saved"""

def main(
        config: ConvertConfig
):
    """
    Convert any Nerfacto based model to Neo model suitable for editing.
    """

    assert config.input is not None, "Please specify --input. Type 'ns-convert -h' for more help."

    # Load config
    nerfacto_config_path = config.input / "config.yml"
    nerfacto_config = yaml.load(nerfacto_config_path.read_text(), Loader=yaml.Loader)

    assert nerfacto_config is not None, f"Could not find the config file in specified directory: {nerfacto_config_path}"
    assert isinstance(nerfacto_config, TrainerConfig), "Config is not the instance of TrainerConfig and thus cannot be converted."

    # Change model name in config and copy the content of input folder to its new location 
    old_base_dir = nerfacto_config.get_base_dir()
    nerfacto_config.method_name = "neo"
    shutil.copytree(old_base_dir, nerfacto_config.get_base_dir(), dirs_exist_ok=True)

    # Copy pipeline settings
    neo_pipeline_config = NeoPipelineConfig()
    neo_pipeline_config.bbox_data_path = nerfacto_config.get_base_dir()

    # Copy datamanager settings
    datamanager_config = VanillaDataManagerConfig()
    for field in fields(VanillaDataManagerConfig):
        if field.name != '_target':
            setattr(datamanager_config, field.name, getattr(nerfacto_config.pipeline.datamanager, field.name))

    # Copy model settings
    neo_model_config = NeoModelConfig()
    for field in fields(NeoModelConfig):
        setattr(neo_model_config, field.name, getattr(nerfacto_config.pipeline.model, field.name))
        
    neo_pipeline_config.datamanager = datamanager_config
    nerfacto_config.pipeline = neo_pipeline_config

    # Save changed config
    nerfacto_config.save_config()

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(ConvertConfig))

if __name__ == '__main__':
    entrypoint()