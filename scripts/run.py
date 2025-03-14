import tyro
import yaml
import os

from dataclasses import dataclass, field
from pathlib import Path
from pipeline_progress_bar import PipelineProgress
from typing import Type, Literal, Union

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class PipelineElements():

    proces_bags: bool = True
    """Process bags"""
    colmap: bool = True
    """Run COLMAP"""
    extract_arm_movement: bool = True
    """Extract arm movement"""

@dataclass
class ExperimentConfig(InstantiateConfig):
    """Configuration for running experiments."""

    _target: Type = field(default_factory=lambda: Experiment)
    """target class to instantiate"""
    experiment_name: str = "default"
    """Name of the experiment"""
    load_config: Union[str, None] = None
    """Load configuration file from name"""
    generate_config: bool = False
    """Generate configuration file and exit program"""
    root_input: Path = Path("data")
    """Root directory for input data"""
    data_type: Literal["real", "synthetic"] = "real"
    """Type of data to process"""
    action: Literal["remove", "translate", "rotate", "translate_and_rotate", "mix"] = "remove"
    """Action to perform"""
    scene_type: Literal["single", "multiple_one", "multiple_sequential", "multiple_multiaction"] = "single"
    """Type of scene"""
    scene_name: Union[str, None]= None
    """Name of scene"""
    object_name: Union[str, None] = None
    """Name of manipulated object"""
    skip_colmap: bool = False
    """Skip colmap processing"""
    pipeline_elements: PipelineElements = field(default_factory=PipelineElements)
    """Pipeline elements to run"""

    def get_base_dir(self):
        return self.root_input / self.data_type / self.action / self.scene_type / self.scene_name / self.object_name
    
    def print_to_terminal(self) -> None:
        """Helper to pretty print config to terminal"""
        CONSOLE.rule("Config")
        CONSOLE.print(self)
        CONSOLE.rule("")

    def save_config(self) -> None:
        """Save config to base directory"""
        base_dir =Path("experiments")
        assert base_dir is not None, f"Data directory does not exist: {base_dir}"
        base_dir.mkdir(parents=True, exist_ok=True)
        config_yaml_path = base_dir / f"{self.experiment_name}.yml"
        CONSOLE.log(f"Saving config to: {config_yaml_path}")
        config_yaml_path.write_text(yaml.dump(self), "utf8")


class Experiment():
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run(self):
        process_bags = {"name": "Process Bags", "active": self.config.pipeline_elements.proces_bags, "idx": 1}
        colmap = {"name": "COLMAP", "active": self.config.pipeline_elements.colmap, "idx": 2}
        extract_arm_movement = {"name": "Extract Arm Movement", "active": self.config.pipeline_elements.extract_arm_movement, "idx": 3}
        ready_datasets = {"name": "Ready Datasets", "active": True, "idx": 4}

        if self.config.pipeline_elements.proces_bags:
            command = [
                "python3 data_processing/process_bags.py",
                f"--data_type {self.config.data_type}",
                f"--scene_name {self.config.scene_name}",
                f"--action {self.config.action}",
                f"--scene_type {self.config.scene_type}",
                f"--object_name {self.config.object_name}",
                f"--root_input {self.config.root_input}",
                f"--root_bag raw_bagfiles",
                f"--skip_frames 6",
                # ${skip_colmap:+--skip_colmap}  # Only include --skip_colmap if it's true
            ]
        
        pipeline_elements = [[process_bags, "→", colmap, "→", ready_datasets],
                            [None, "", extract_arm_movement, "↗", None]]

        with PipelineProgress(pipeline_elements) as progress:
            for item in progress:
                progress.start_step()
                
                # Run command item["command"]

                progress.end_step()


def main(config: ExperimentConfig) -> None:
    """Main function."""

    # Create experiments directory if it does not exist.
    os.makedirs("experiments", exist_ok=True)

    # Load configuration from file.
    if config.load_config is not None:
        assert os.path.exists(f"experiments/{config.load_config}.yml"), f"Config file does not exist: experiments/{config.load_config}"

        CONSOLE.log(f"Loading pre-set config from: experiments/{config.load_config}")
        with open(f"experiments/{config.load_config}.yml", "r") as f:
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