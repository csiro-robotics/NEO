import tyro
import time
import yaml
import os

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Union

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE

from pipeline_progress_bar import PipelineProgress
from experiment_config import NeoDataConfig
from data_preparation.calculate_colmap import calculate_colmap
from data_preparation.extract_images import extract_images_rs
from data_preparation.match_arucos import match_arucos
from data_preparation.extract_arm_movement import extract_arm_movement


@dataclass
class PipelineElements():

    proces_bags: bool = True
    """Process bags"""
    colmap: bool = True
    """Run COLMAP"""
    match_arucos: bool = True
    """Match Arucos"""
    extract_arm_movement: bool = True
    """Extract arm movement"""

@dataclass
class ExperimentConfig(NeoDataConfig, InstantiateConfig):
    """Configuration for running experiments."""

    _target: Type = field(default_factory=lambda: Experiment)
    """target class to instantiate"""
    experiment_name: str = "default"
    """Name of the experiment"""
    load_config: Union[str, None] = None
    """Load configuration file from name"""
    generate_config: bool = False
    """Generate configuration file and exit program"""
    pipeline_elements: PipelineElements = field(default_factory=PipelineElements)
    """Pipeline elements to run"""

    def save_config(self) -> None:
        """Save config to base directory"""
        base_dir =Path("experiments")
        assert base_dir is not None, f"Data directory does not exist: {base_dir}"
        base_dir.mkdir(parents=True, exist_ok=True)
        config_yaml_path = base_dir / f"{self.experiment_name}.yml"
        CONSOLE.log(f"Saving config to: {config_yaml_path}")
        config_yaml_path.write_text(yaml.dump(self), "utf8")

def not_implemented(config: ExperimentConfig):
    print("Not implemented")
    time.sleep(3)

class Experiment():
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run(self):
        process_bags = {"name": "Process Bags", "active": self.config.pipeline_elements.proces_bags, "idx": 1, "func": extract_images_rs}
        colmap = {"name": "COLMAP", "active": self.config.pipeline_elements.colmap, "idx": 2, "func": calculate_colmap}
        matching_arucos = {"name": "Match Arucos", "active": self.config.pipeline_elements.match_arucos, "idx": 3, "func": match_arucos}
        extracting_arm_movement = {"name": "Extract Arm Movement", "active": self.config.pipeline_elements.extract_arm_movement, "idx": 4, "func": extract_arm_movement}
        
        pipeline_elements = [[process_bags, "→", colmap, "→", matching_arucos],
                            [None, "", None, "↘", extracting_arm_movement]]

        with PipelineProgress(pipeline_elements, console=CONSOLE) as progress:
            for item in progress:
                progress.start_step()
                
                item["func"](self.config)

                progress.end_step()


def main(config: ExperimentConfig) -> None:
    """Main function."""

    # Create experiments directory if it does not exist.
    os.makedirs("experiments", exist_ok=True)

    # Load configuration from file.
    if config.load_config is not None:
        config_path = Path(f"experiments/{config.load_config}")
        assert config_path.exists(), f"Config file does not exist: {config_path}"

        CONSOLE.log(f"Loading pre-set config from: experiments/{config.load_config}")
        with open(f"experiments/{config.load_config}", "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)

    else:
        # Save configuration to file.
        config.save_config()
        
    config.print_to_terminal()
    if config.generate_config:
        CONSOLE.log("Configuration file generated. Exiting program.")
        return

    # Instantiate the target class.
    target = config.setup()

    # Run the experiment.
    target.run()


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")

    AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
        tyro.conf.FlagConversionOff[
            ExperimentConfig
        ]
    ]

    main(
        tyro.cli(
            AnnotatedBaseConfigUnion,
            description=convert_markup_to_ansi(__doc__),
        )
    )

if __name__ == "__main__":
    entrypoint()