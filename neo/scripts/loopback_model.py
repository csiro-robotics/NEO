import tyro
import yaml
import os
import shutil

from dataclasses import dataclass
from pathlib import Path
from typing import Union
from typing_extensions import Annotated
from abc import abstractmethod

from nerfstudio.utils.scripts import run_command


@dataclass
class BaseLoopback:

    input: Path
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
            f"ns-convert to-nerfacto",
            f"--input {self.input}",
        ]
        convert_command = " ".join(convert_command)

        run_command(convert_command, verbose=self.verbose)


@dataclass
class LoopbackLERF(BaseLoopback):

    def main(self):
        
        convert_command = [
            f"ns-convert to-nerfacto",
            f"--input {self.input}",
        ]
        convert_command = " ".join(convert_command)

        # run_command(convert_command, verbose=self.verbose)

        # Load config to get data path
        neo_config = yaml.load(Path(str(self.input / "config.yml").replace("replace-nerf", "neo")).read_text(), Loader=yaml.Loader)
        nerfacto_path = Path(str(self.input).replace("replace-nerf", 'nerfacto'))
        render_path = nerfacto_path / "cache"
        os.makedirs(render_path, exist_ok=True)
        shutil.copy(neo_config.data / "transforms.json", render_path / "transforms.json")

        render_command = [
            "ns-render-neo render-from-transforms",
            f"--load-config {nerfacto_path / 'config.yml'}",
            f"--output-path {render_path}",
            "--output-format images",
            "--image-format png",
            "--rendered-output-names rgb depth",
            f"--transforms-filename {neo_config.data}/transforms.json"
        ]
        render_command = " ".join(render_command)

        # run_command(render_command, verbose=self.verbose)

        # Rename folder
        if (render_path / "rgb").exists():
            shutil.move(render_path / "rgb", render_path / "images")

        # Train language field
        train_command = [
            "ns-train lerf-lite-lan",
            f"--data {render_path}",
            f"--output-dir {self.input.parents[3]}",
            f"--experiment_name {neo_config.experiment_name}",
            f"--timestamp {neo_config.timestamp}",
            "--max-num-iterations 3000",
            f"--pipeline.model.camera-optimizer.mode {neo_config.pipeline.model.camera_optimizer.mode}",
            "--viewer.quit-on-train-completion True",
            "nerfstudio-data",
            f"--orientation-method {neo_config.pipeline.datamanager.dataparser.orientation_method}",
            f"--center-method {neo_config.pipeline.datamanager.dataparser.center_method}",
            f"--auto-scale-poses {neo_config.pipeline.datamanager.dataparser.auto_scale_poses}",
            f"--train-split-fraction {neo_config.pipeline.datamanager.dataparser.train_split_fraction}"
        ]
        train_command = " ".join(train_command)

        run_command(train_command, verbose=self.verbose)


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