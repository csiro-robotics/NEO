import tyro
import yaml
import os
import shutil

from dataclasses import dataclass
from pathlib import Path
from typing import Union
from typing_extensions import Annotated
from abc import abstractmethod

from neo.utils.commands import run_command
from datetime import datetime


@dataclass
class BaseLoopback:

    load_config: Path
    """Path to the folder containing config.yaml file and nerfstudio_models"""
    verbose: bool = False
    """If set to True, the output will be more verbose"""

    @abstractmethod
    def main(self):
        NotImplementedError("This method should be implemented in the subclass.")


@dataclass
class LoopbackNerfacto(BaseLoopback):

    def main(self):
        
        convert_command = [
            "ns-convert", "to-nerfacto",
            "--load-config", self.load_config,
        ]

        run_command(convert_command, verbose=self.verbose)


@dataclass
class LoopbackLERF(BaseLoopback):

    def main(self):
        
        # Convert to Nerfacto
        convert_command = [
            "ns-convert", "to-nerfacto",
            "--load-config", self.load_config,
        ]

        run_command(convert_command, verbose=self.verbose)

        model_type = "neo-big" if "neo-big" in str(self.load_config) else "neo"

        # Load config to get data path
        neo_config = yaml.load(Path(str(self.load_config)).read_text(), Loader=yaml.Loader)
        nerfacto_path = Path(str(self.load_config.parent).replace(model_type, "nerfacto"))
        render_path = nerfacto_path / "cache"
        os.makedirs(render_path, exist_ok=True)
        shutil.copy(neo_config.data / "transforms.json", render_path / "transforms.json")

        render_command = [
            "ns-render-neo", "render-from-transforms",
            "--load-config", nerfacto_path / "config.yml",
            "--output-path", render_path,
            "--output-format", "images",
            "--image-format", "png",
            "--rendered-output-names", "rgb", "depth",
            "--transforms-filename", neo_config.data / "transforms.json",
        ]

        run_command(render_command, verbose=self.verbose)

        # Rename folder
        if (render_path / "rgb").exists():
            destination_path = render_path / "images"
            if destination_path.exists():
                shutil.rmtree(destination_path)
            shutil.copytree(render_path / "rgb", destination_path, dirs_exist_ok=True)
            shutil.rmtree(render_path / "rgb")

        # Train language field
        train_command = [
            "ns-train", "lerf-lite-lan",
            "--data", render_path,
            "--output-dir", self.load_config.parents[4],
            "--experiment-name", neo_config.experiment_name,
            "--timestamp", neo_config.timestamp,
            "--max-num-iterations", "3000",
            "--pipeline.model.camera-optimizer.mode", neo_config.pipeline.model.camera_optimizer.mode,
            "--pipeline.model-type", model_type,
            "--viewer.quit-on-train-completion", "True",
            "nerfstudio-data",
            "--orientation-method", neo_config.pipeline.datamanager.dataparser.orientation_method,
            "--center-method", neo_config.pipeline.datamanager.dataparser.center_method,
            "--auto-scale-poses", neo_config.pipeline.datamanager.dataparser.auto_scale_poses,
            "--train-split-fraction", neo_config.pipeline.datamanager.dataparser.train_split_fraction,
        ]

        run_command(train_command, verbose=self.verbose)

        # Move model to its new location
        if "neo-big" in str(self.load_config):
            new_config_path = Path(str(self.load_config).replace("neo-big", "lerf-lite-lan"))
        else:
            new_config_path = Path(str(self.load_config).replace("neo", "lerf-lite-lan"))
        lerf_lite_lan_config = yaml.load(new_config_path.read_text(), Loader=yaml.Loader)
        lerf_lite_lan_config.method_name = "lerf-lite"

        new_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        lerf_lite_lan_config.timestamp = new_timestamp
        output_path = str(lerf_lite_lan_config.pipeline.output_path).replace("lerf-lite-lan", "lerf-lite").replace(neo_config.timestamp, new_timestamp)
        lerf_lite_lan_output_path = Path(lerf_lite_lan_config.pipeline.output_path)
        lerf_lite_lan_config.pipeline.output_path = output_path 
        lerf_lite_lan_config.pipeline.model.output_folder = output_path

        # Overwrite config.yml
        with open(new_config_path, 'w') as config_file:
            yaml.dump(lerf_lite_lan_config, config_file)

        # Copy the timestamp folder to lerf-lite location
        shutil.copytree(lerf_lite_lan_output_path, output_path, dirs_exist_ok=True)

        # Remove lerf-lite-lan folder
        shutil.rmtree(lerf_lite_lan_output_path.parent)

        # Convert to Neo
        if "neo-big" in str(self.load_config):
            convert_command = [
                "ns-convert", "to-neo-big",
                "--load-config", Path(output_path) / "config.yml",
            ]
        else:
            convert_command = [
                "ns-convert", "to-neo",
                "--load-config", Path(output_path) / "config.yml",
            ]

        run_command(convert_command, verbose=self.verbose)


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[LoopbackNerfacto, tyro.conf.subcommand(name="nerfacto")],
        Annotated[LoopbackLERF, tyro.conf.subcommand(name="lerf")],
    ]
]

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()

if __name__ == '__main__':
    entrypoint()
