import tyro

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE
from neo.utils.commands import run_command

from scripts.experiment_config import NeoDataConfig


def generate_bbox(config: NeoDataConfig) -> None:
    """Main function that executes the training pipeline."""

    # Run LERF-TOGO
    CONSOLE.print("[bold green]Obtaining BBoxes...[/bold green]")
    command = [
        "lerftogo-neo",
        "--root_input", config.root_input,
        "--data_type", config.data_type,
        "--scene_name", config.scene_name,
        "--action", config.action,
        "--scene_type", config.scene_type,
        "--object_name", config.object_name,
        "--lerf_prompt", config.lerf_prompt,
        "--model_types", "neo", "neo-big",
    ]
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
