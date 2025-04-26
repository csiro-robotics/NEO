import tyro

from pathlib import Path

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.scripts import run_command

from experiment_config import NeoDataConfig


def generate_bbox(config: NeoDataConfig) -> None:
    """Main function that executes the training pipeline."""

    # Run LERF-TOGO
    CONSOLE.print("[bold green]Obtaining BBoxes...[/bold green]")
    command = [
        "lerftogo-neo",
        f"--root_input {config.root_input}",
        f"--data_type {config.data_type}",
        f"--scene_name {config.scene_name}",
        f"--action {config.action}",
        f"--scene_type {config.scene_type}",
        f"--object_name {config.object_name}",
        f"--lerf_prompt \'{config.lerf_prompt}\'"
    ]
    command = " ".join(command)

    run_command(command, verbose=True)

def entrypoint():
    """Entrypoint for CLI."""
    tyro.extras.set_accent_color("bright_yellow")

    generate_bbox(
        tyro.cli(
            NeoDataConfig,
            description=convert_markup_to_ansi(__doc__),
        )
    )

if __name__ == "__main__":
    entrypoint()
