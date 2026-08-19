import tyro
import os

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Union

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
    train_gt: bool = True
    """Train GT"""

@dataclass
class TrainingElements():
    """Training elements to run"""

    _execute_all: bool = True
    """Execute all training elements"""
    generate_bbox: bool = True
    """Generate BBoxes"""
    train_neo: bool = True
    """Train Neo"""

@dataclass
class EvaluationElements():
    """Evaluation elements to run"""

    _execute_all: bool = True
    """Execute all evaluation elements"""
    prepare_nerfiller: bool = True
    """Prepare Nerfiller"""
    train_nerfiller: bool = True
    """Train Nerfiller"""
    train_nerfiller_no_depth: bool = True
    """Train Nerfiller without depth"""
    evaluate: bool = True
    """Evaluate"""

@dataclass
class PipelineElements():

    data_preparation: DataPreperationElements = field(default_factory=DataPreperationElements)
    """Data preparation elements to run"""
    training: TrainingElements = field(default_factory=TrainingElements)
    """Training elements to run"""
    evaluation: EvaluationElements = field(default_factory=EvaluationElements)
    """Evaluation elements to run"""

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
    lerf_prompt: str = "What is the object in the scene?"
    """Prompt for LERF-TOGO"""

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
        # Experiment setupe
        if self.config.pipeline_elements.data_preparation._execute_all:
            from scripts.data_preparation.calculate_colmap import calculate_colmap
            from scripts.data_preparation.extract_arm_movement import extract_arm_movement
            from scripts.data_preparation.extract_images import extract_images
            from scripts.data_preparation.match_arucos import match_arucos
            from scripts.data_preparation.train_gt import train_gt
            from scripts.data_preparation.train_lerf import train_lerf

            elements = self.config.pipeline_elements.data_preparation

            process_bags = {"name": "Process Bags", "active": elements.process_bags, "idx": 1, "func": extract_images, "kwargs": {"skip_frames": 6, "select_frames": False, "num_frames": 250}}
            colmap_1 = {"name": "COLMAP", "active": elements.colmap_1, "idx": 2, "func": calculate_colmap}
            selecting_frames = {"name": "Select Frames", "active": elements.select_frames, "idx": 3, "func": extract_images, "kwargs": {"skip_frames": 6, "select_frames": True, "num_frames": 250}}
            colmap_2 = {"name": "COLMAP", "active": elements.colmap_2, "idx": 4, "func": calculate_colmap}
            matching_arucos = {"name": "Match Arucos", "active": elements.match_arucos, "idx": 5, "func": match_arucos}
            extracting_arm_movement = {"name": "Extract Arm Movement", "active": elements.extract_arm_movement, "idx": 6, "func": extract_arm_movement}
            ready_datasets = {"name": "Ready Datasets", "active": True, "idx": 7, "func": dummy_fnc}
            training_lerf = {"name": "Training LERF", "active": elements.train_lerf, "idx": 8, "func": train_lerf, "kwargs": {"camera_optimizer": "off"}}
            training_gt = {"name": "Training GT", "active": elements.train_gt, "idx": 9, "func": train_gt, "kwargs": {"camera_optimizer": "off"}}
            
            pipeline_elements = [[process_bags, "→", colmap_1, "→", selecting_frames, "→", colmap_2, "→", matching_arucos, "→", ready_datasets, "→", training_lerf],
                                [None, "", None, "", None, "", None, "↘", extracting_arm_movement, "↗", None, "↘", training_gt]]

            self._execute_pipeline(pipeline_elements)

        # Training editing networks
        if self.config.pipeline_elements.training._execute_all:
            from scripts.editing.edit import edit
            from scripts.editing.generate_bbox import generate_bbox

            elements = self.config.pipeline_elements.training

            generating_bbox = {"name": "Generate BBox", "active": elements.generate_bbox, "idx": 1, "func": generate_bbox}
            train_neo = {"name": "Editing Neo", "active": elements.train_neo, "idx": 2, "func": edit}
            
            pipeline_elements = [[generating_bbox, "→", train_neo]]
            
            self._execute_pipeline(pipeline_elements)

        # Evaluation
        if self.config.pipeline_elements.evaluation._execute_all:
            from scripts.data_preparation.prepare_nerfiller import prepare_nerfiller
            from scripts.evaluation.eval import eval
            from scripts.evaluation.train_nerfiller import train_nerfiller
            from scripts.evaluation.train_nerfiller_no_depth import train_nerfiller_no_depth

            elements = self.config.pipeline_elements.evaluation
            preparing_nerfiller = {"name": "Prepare Nerfiller", "active": elements.prepare_nerfiller, "idx": 1, "func": prepare_nerfiller}
            training_nerfiller = {"name": "Training Nerfiller", "active": elements.train_nerfiller, "idx": 2, "func": train_nerfiller}
            training_nerfiller_no_depth = {"name": "Training Nerfiller No Depth", "active": elements.train_nerfiller_no_depth, "idx": 3, "func": train_nerfiller_no_depth}
            evaluating = {"name": "Evaluation", "active": elements.evaluate, "idx": 4, "func": eval}
            
            pipeline_elements = [[preparing_nerfiller, "→", training_nerfiller, "→", evaluating],
                                [None, "↘", training_nerfiller_no_depth, "↗", None]]
            
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
