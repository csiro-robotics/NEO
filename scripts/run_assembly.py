import tyro
import os

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type, Union, List

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE

from scripts.pipeline_progress_bar import PipelineProgress
from scripts.experiment_config import NeoDataConfig
from neo.utils.experiment_io import load_experiment_config, save_experiment_config


@dataclass
class DataPreperationElements():
    """Data preparation elements to run"""

    _execute_all: bool = True
    """Execute all data preparation elements"""
    process_bags: bool = True
    """Process bags"""
    colmap_1: bool = True
    """Run COLMAP"""
    colmap_2: bool = True
    """Run COLMAP"""
    select_frames: bool = True
    """Select frames"""
    match_arucos: bool = True
    """Match Arucos"""
    extract_arm_movement: bool = True
    """Extract arm movement"""
    train_lerf: bool = True
    """Train LERF"""

@dataclass
class TrainingElements():
    """Training elements to run"""

    _execute_all: bool = True
    """Execute all training elements"""
    prepare_subassembly: bool = True
    """Prepare subassembly"""
    generate_bbox: bool = True
    """Generate BBoxes"""
    train_neo: bool = True
    """Train Neo"""
    loopback_neo: bool = True
    """Loopback Neo"""

@dataclass
class PipelineElements():

    data_preparation: DataPreperationElements = field(default_factory=DataPreperationElements)
    """Data preparation elements to run"""
    training: TrainingElements = field(default_factory=TrainingElements)
    """Training elements to run"""

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
    lerf_prompts: Optional[List[str]] = field(default_factory=lambda: ["What is the object in the scene?"])
    """Prompt for LERF-TOGO"""
    lerf_prompt: Optional[str] = None
    """Prompt for LERF-TOGO"""
    object_names: Optional[List[str]] = field(default_factory=lambda: ["object"])
    """Names of the objects in the scene"""
    object_name: Optional[str] = "assembly"
    """Name of the object in the scene"""
    steps_to_run: Optional[List[int]] = None
    """Steps to run in the experiment"""
    nerfiller_device: str = "cuda:0"
    """What device should nerfiller be run on (change to cpu for low-power devices)"""

    def save_config(self) -> None:
        """Save config to base directory"""
        base_dir =Path("experiments")
        assert base_dir is not None, f"Data directory does not exist: {base_dir}"
        base_dir.mkdir(parents=True, exist_ok=True)
        config_yaml_path = base_dir / f"{self.experiment_name}.yml"
        CONSOLE.log(f"Saving config to: {config_yaml_path}")
        save_experiment_config(self, config_yaml_path)

def dummy_fnc(config: ExperimentConfig):
    pass

class Experiment():
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def _execute_pipeline(self, pipeline_elements):
        """Execute a pipeline of elements."""
        with PipelineProgress(pipeline_elements, console=CONSOLE) as progress:
            for item in progress:
                progress.start_step()
                if item.get("kwargs"):
                    item["func"](self.config, **item["kwargs"])
                else:
                    item["func"](self.config)
                progress.end_step()

    def run(self):

        # Experiment setup
        if self.config.pipeline_elements.data_preparation._execute_all:
            from scripts.data_preparation.calculate_colmap import calculate_colmap
            from scripts.data_preparation.extract_arm_movement import extract_arm_movement
            from scripts.data_preparation.extract_images import extract_images
            from scripts.data_preparation.match_arucos import match_arucos
            from scripts.data_preparation.train_lerf import train_lerf

            elements = self.config.pipeline_elements.data_preparation

            process_bags = {"name": "Process Bags", "active": elements.process_bags, "idx": 1, "func": extract_images, "kwargs": {"skip_frames": 6, "select_frames": False, "num_frames": 250, "skip_after": True}}
            colmap_1 = {"name": "COLMAP", "active": elements.colmap_1, "idx": 2, "func": calculate_colmap, "kwargs": {"skip_after": True}}
            selecting_frames = {"name": "Select Frames", "active": elements.select_frames, "idx": 3, "func": extract_images, "kwargs": {"skip_frames": 6, "select_frames": True, "num_frames": 250, "skip_after": True}}
            colmap_2 = {"name": "COLMAP", "active": elements.colmap_2, "idx": 4, "func": calculate_colmap, "kwargs": {"skip_after": True}}
            matching_arucos = {"name": "Match Arucos", "active": elements.match_arucos, "idx": 5, "func": match_arucos, "kwargs": {"skip_after": True}}
            extracting_arm_movement = {"name": "Extract Arm Movement", "active": elements.extract_arm_movement, "idx": 6, "func": extract_arm_movement}
            ready_datasets = {"name": "Ready Datasets", "active": True, "idx": 7, "func": dummy_fnc}
            training_lerf = {"name": "Training LERF", "active": elements.train_lerf, "idx": 8, "func": train_lerf, "kwargs": {"camera_optimizer": "off"}}
            
            pipeline_elements = [[process_bags, "→", colmap_1, "→", selecting_frames, "→", colmap_2, "→", matching_arucos, "→", ready_datasets, "→", training_lerf],
                                [None, "", None, "", None, "", None, "↘", extracting_arm_movement, "↗", None]]

            self._execute_pipeline(pipeline_elements)

        for step in self.config.steps_to_run:
            self.config.object_name = self.config.object_names[step]
            self.config.lerf_prompt = self.config.lerf_prompts[step]

            # Training editing networks
            if self.config.pipeline_elements.training._execute_all:
                from scripts.data_preparation.prepare_subassembly import prepare_subassembly
                from scripts.editing.edit import edit
                from scripts.editing.generate_bbox import generate_bbox
                from scripts.editing.loopback import loopback

                elements = self.config.pipeline_elements.training

                preparing_subassembly = {"name": "Prepare Subassembly", "active": elements.prepare_subassembly, "idx": 1, "func": prepare_subassembly, "kwargs": {"first_step": step==0, "previous_object_name": self.config.object_names[step-1] if step>0 else None}}
                generating_bbox = {"name": "Generate BBox", "active": elements.generate_bbox, "idx": 2, "func": generate_bbox}
                train_neo = {"name": "Editing Neo", "active": elements.train_neo, "idx": 3, "func": edit}
                loopback_neo = {"name": "Looping Neo", "active": elements.loopback_neo, "idx": 4, "func": loopback}
                edited_object = {"name": self.config.lerf_prompt, "active": False, "idx": 5, "func": dummy_fnc}

                pipeline_elements = [[preparing_subassembly, "→", generating_bbox, "→", train_neo, "→", loopback_neo],
                                    [edited_object, "", None, "", None, "", None]]

                self._execute_pipeline(pipeline_elements)


def main(config: ExperimentConfig) -> None:
    """Main function."""

    # Create experiments directory if it does not exist.
    os.makedirs("experiments", exist_ok=True)

    # Load configuration from file.
    if config.load_config is not None:
        config_path = Path(f"experiments/{config.load_config}")
        assert config_path.exists(), f"Config file does not exist: {config_path}"

        CONSOLE.log(f"Loading pre-set config from: experiments/{config.load_config}")
        config = load_experiment_config(config, config_path)

    else:
        # Save configuration to file.
        config.set_timestamp()
        config.save_config()
        
    config.print_to_terminal()
    if config.generate_config:
        config.generate_config = False
        config.save_config()
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
