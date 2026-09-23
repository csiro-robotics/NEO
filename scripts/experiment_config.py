from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Literal, Union
from nerfstudio.utils.rich_utils import CONSOLE


TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")


@dataclass
class NeoDataConfig():
    """Configuration for running experiments."""

    neo_data_root: Path = Path("data")
    """Root directory for neo data"""
    outputs_root: Path = Path("outputs")
    """Root directory for trained models and editing outputs"""
    raw_bagfiles_root: Union[Path, None] = None
    """Optional raw capture root; defaults to <neo_data_root>/raw_bagfiles"""
    action: Literal["remove", "reconfigure", "assembly"] = "remove"
    """Action to perform"""
    scene_name: Union[str, None]= None
    """Name of scene"""
    object_name: Union[str, None] = None
    """Name of manipulated object"""
    timestamp: Union[str, None] = None
    """Timestamp of the experiment"""

    def get_base_dir(self):
        return self.neo_data_root / "processed" / self.action / self.scene_name / self.object_name

    def get_relative_scene_dir(self) -> Path:
        """Return the scene path relative to the processed data root."""
        return self.get_base_dir().relative_to(self.neo_data_root / "processed")

    def get_output_dir(self) -> Path:
        """Return the model/output directory corresponding to the data scene."""
        return self.outputs_root / self.get_relative_scene_dir()

    def get_raw_capture_base(self) -> Path:
        """Return the raw capture path without a stage suffix."""
        raw_root = self.raw_bagfiles_root or self.neo_data_root / "raw_bagfiles"
        return raw_root / self.get_relative_scene_dir()
    
    def print_to_terminal(self) -> None:
        """Helper to pretty print config to terminal"""
        CONSOLE.rule("Config")
        CONSOLE.print(self)
        CONSOLE.rule("")

    def set_timestamp(self):
        """Set the timestamp of the experiment"""
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    def get_timestamp(self):
        """Get the timestamp of the experiment"""
        if self.timestamp is None:
            base_dir = self.get_output_dir() / "before" / "lerf-lite"

            if base_dir.exists():
                timestamp_dirs = sorted(
                    (
                        path
                        for path in base_dir.iterdir()
                        if path.is_dir() and TIMESTAMP_PATTERN.fullmatch(path.name)
                    ),
                    reverse=True,
                )
                if timestamp_dirs:
                    self.timestamp = timestamp_dirs[0].name

            if self.timestamp is None:
                CONSOLE.print("[bold yellow]No timestamp directories found! Generating new timestamp.[/bold yellow]")
                self.set_timestamp()

        return self.timestamp
