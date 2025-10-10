import torch
import tyro
import yaml
import os
import shutil
import torch

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Union
from typing_extensions import Annotated
from abc import ABC, abstractmethod

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig

from neo.neo_pipeline import NeoPipelineConfig
from neo.neo_model import NeoModelConfig
from neo.neo_trainer import NeoTrainerConfig
from neo.neo_config import neo_method, neo_big_method
from neo.utils.inpainting import InpainterConfig


@dataclass
class BaseConvert:
    load_config: Path
    """Path to the folder containing config.yaml file and nerfstudio_models"""

    @abstractmethod
    def main(self):
        assert self.load_config is not None, "Please specify --load_config. Type 'ns-convert -h' for more help."

        # Load config
        self.config = yaml.load(self.load_config.read_text(), Loader=yaml.Loader)

        assert self.config is not None, f"Could not find the config file in specified directory: {self.load_config}"
        assert isinstance(self.config, TrainerConfig), "Config is not the instance of TrainerConfig and thus cannot be converted."


@dataclass
class ConvertToNeo(BaseConvert):
    def main(self):
        """
        Convert any Nerfacto based model to Neo model suitable for editing.
        """
        super().main()

        # Change model name in config and copy the content of load_config folder to its new location 
        old_base_dir = self.config.get_base_dir()
        self.config.method_name = "neo"
        shutil.copytree(old_base_dir, self.config.get_base_dir(), dirs_exist_ok=True)

        # Copy pipeline settings
        pipeline_config = neo_method.config.pipeline
        pipeline_config.bbox_data_path = self.config.get_base_dir()

        # Assign Neo trainer
        self.config = NeoTrainerConfig(**self.config.__dict__)
        self.config._target = neo_method.config._target

        # Copy optimizers settings
        self.config.optimizers = neo_method.config.optimizers

        # Copy datamanager settings
        datamanager_config = VanillaDataManagerConfig()
        for field in fields(VanillaDataManagerConfig):
            if field.name not in ['_target']:
                if field.name in ['train_num_rays_per_batch', 'eval_num_rays_per_batch']:
                    setattr(datamanager_config, field.name, getattr(neo_method.config.pipeline.datamanager, field.name))
                else:
                    setattr(datamanager_config, field.name, getattr(self.config.pipeline.datamanager, field.name))

        # Copy model settings
        model_config = NeoModelConfig()
        for field in fields(NeoModelConfig):
            if field.name not in ['_target']:
                if field.name in ['eval_num_rays_per_chunk']:
                    setattr(model_config, field.name, getattr(neo_method.config.pipeline.model, field.name))
                else:
                    setattr(model_config, field.name, getattr(self.config.pipeline.model, field.name))
        
        pipeline_config.datamanager = datamanager_config
        pipeline_config.model = model_config
        self.config.pipeline = pipeline_config

        # Save changed config
        self.config.save_config()

        # Copy datapareser transforms file if exists
        if (old_base_dir / "dataparser_transforms.json").exists():
            shutil.copyfile(old_base_dir / "dataparser_transforms.json", self.config.get_base_dir() / "dataparser_transforms.json")


@dataclass
class ConvertToNeoBig(BaseConvert):
    def main(self):
        """
        Convert any Nerfacto based model to Neo model suitable for editing.
        """
        super().main()

        # Change model name in config and copy the content of load_config folder to its new location 
        old_base_dir = self.config.get_base_dir()
        self.config.method_name = "neo-big"
        shutil.copytree(old_base_dir, self.config.get_base_dir(), dirs_exist_ok=True)

        # Copy pipeline settings
        pipeline_config = neo_big_method.config.pipeline
        pipeline_config.bbox_data_path = self.config.get_base_dir()

        # Assign Neo trainer
        self.config = NeoTrainerConfig(**self.config.__dict__)
        self.config._target = neo_big_method.config._target

        # Copy optimizers settings
        self.config.optimizers = neo_big_method.config.optimizers

        # Copy datamanager settings
        datamanager_config = VanillaDataManagerConfig()
        for field in fields(VanillaDataManagerConfig):
            if field.name not in ['_target']:
                if field.name in ['train_num_rays_per_batch', 'eval_num_rays_per_batch']:
                    setattr(datamanager_config, field.name, getattr(neo_big_method.config.pipeline.datamanager, field.name))
                else:
                    setattr(datamanager_config, field.name, getattr(self.config.pipeline.datamanager, field.name))

        # Copy model settings
        model_config = NeoModelConfig()
        for field in fields(NeoModelConfig):
            if field.name not in ['_target']:
                if field.name in ['eval_num_rays_per_chunk']:
                    setattr(model_config, field.name, getattr(neo_big_method.config.pipeline.model, field.name))
                else:
                    setattr(model_config, field.name, getattr(self.config.pipeline.model, field.name))
        
        pipeline_config.datamanager = datamanager_config
        pipeline_config.model = model_config
        self.config.pipeline = pipeline_config

        # Save changed config
        self.config.save_config()

        # Copy datapareser transforms file if exists
        if (old_base_dir / "dataparser_transforms.json").exists():
            shutil.copyfile(old_base_dir / "dataparser_transforms.json", self.config.get_base_dir() / "dataparser_transforms.json")


@dataclass
class ConvertToNerfacto(BaseConvert):
    def main(self):
        """
        Convert Replace-NeRF model to Nerfacto model.
        """
        super().main()

        # Change model name in config and copy the content of load_config folder to its new location
        old_base_dir = self.config.get_base_dir()
        old_base_dir = Path(old_base_dir) / "replace-nerf"
        self.config.method_name = "nerfacto"
        shutil.copytree(old_base_dir, self.config.get_base_dir(), dirs_exist_ok=True)

        # Load the Neo config
        neo_config = yaml.load(Path(str(old_base_dir.parent / "config.yml")).read_text(), Loader=yaml.Loader)

        # Load the .ckpt torch model
        ckpt_path = next((self.config.get_base_dir() / "nerfstudio_models").glob("*.ckpt"))
        model = torch.load(ckpt_path, weights_only=False)

        keys_to_delete = []
        for line in model["pipeline"]:
            if "copy" in line or "replace" in line:
                keys_to_delete.append(line)
                
        for key in keys_to_delete:
            del model["pipeline"][key]

        appearance_embedding = model["pipeline"]["_model.field.embedding_appearance.embedding.weight"]
        pose_adjustment = model["pipeline"].get("_model.camera_optimizer.pose_adjustment", None)

        if pose_adjustment is not None:
            model["pipeline"]["_model.camera_optimizer.pose_adjustment"] = pose_adjustment[:appearance_embedding.shape[0] - pose_adjustment.shape[0], :]

        # Save the modified model
        torch.save(model, ckpt_path)

        # Copy pipeline settings
        pipeline_config = VanillaPipelineConfig()

        # Copy datamanager settings
        datamanager_config = ParallelDataManagerConfig()
        for field in fields(ParallelDataManagerConfig):
            if field.name != '_target':
                if getattr(self.config.pipeline.datamanager, field.name, None) is not None:
                    setattr(datamanager_config, field.name, getattr(self.config.pipeline.datamanager, field.name))

        # Copy model settings
        model_config = NerfactoModelConfig()
        for field in fields(NerfactoModelConfig):
            if field.name != '_target':
                setattr(model_config, field.name, getattr(self.config.pipeline.model, field.name))
        model_config.average_init_density = neo_config.pipeline.model.average_init_density
        
        # Change data path
        self.config.data = neo_config.data
        pipeline_config.model = model_config
        pipeline_config.datamanager = datamanager_config
        pipeline_config.datamanager.data = self.config.data
        pipeline_config.datamanager.dataparser.data = self.config.data
        self.config.pipeline = pipeline_config

        # Save changed config
        self.config.save_config()

        # Copy datapareser transforms file if exists
        if (old_base_dir / "dataparser_transforms.json").exists():
            shutil.copyfile(old_base_dir / "dataparser_transforms.json", self.config.get_base_dir() / "dataparser_transforms.json")



Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ConvertToNeo, tyro.conf.subcommand(name="to-neo")],
        Annotated[ConvertToNeoBig, tyro.conf.subcommand(name="to-neo-big")],
        Annotated[ConvertToNerfacto, tyro.conf.subcommand(name="to-nerfacto")],
    ]
]

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()

if __name__ == '__main__':
    entrypoint()