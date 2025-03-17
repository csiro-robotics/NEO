import tyro
import yaml
import os

from dataclasses import dataclass, field
from pathlib import Path
from pipeline_progress_bar import PipelineProgress
from typing import Type, Literal, Union
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class NeoDataConfig():
    """Configuration for running experiments."""

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

    def get_base_dir(self):
        return self.root_input / self.data_type / self.action / self.scene_type / self.scene_name / self.object_name
    
    def print_to_terminal(self) -> None:
        """Helper to pretty print config to terminal"""
        CONSOLE.rule("Config")
        CONSOLE.print(self)
        CONSOLE.rule("")