from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Union
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
    timestamp: Union[str, None] = None
    """Timestamp of the experiment"""

    def get_base_dir(self):
        return self.root_input / self.data_type / self.action / self.scene_type / self.scene_name / self.object_name
    
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
            # Find the latest timestamped directory
            base_dir = self.get_base_dir() / "before" / "lerf-lite"

            if base_dir.exists():
                timestamp_dirs = sorted(base_dir.glob("*_[0-9]*"), reverse=True)
                if not timestamp_dirs:
                    CONSOLE.print("[bold red]No timestamp directories found! Exiting.[/bold red]")
                    return

                self.timestamp = timestamp_dirs[0].name
                
            else:
                CONSOLE.print("[bold yellow]No timestamp directories found! Generating new timestamp.[/bold yellow]")
                self.set_timestamp()

        return self.timestamp