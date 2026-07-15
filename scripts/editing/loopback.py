import tyro

from typing import Annotated

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE
from neo.utils.commands import run_command

from scripts.experiment_config import NeoDataConfig


def loopback(config: NeoDataConfig, model_type: Annotated["neo", "neo-big"] = "neo") -> None:
    """Main function."""

    config_path = config.get_output_dir() / "before" / model_type / config.get_timestamp() / "config.yml"

    # Evaluate LERF
    CONSOLE.print("[bold green]Evaluating...[/bold green]")
    command = [
        "ns-loopback", "lerf",
        "--load-config", config_path,
        "--verbose", "True",
    ]
    run_command(command, verbose=True)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")

    loopback(
        tyro.cli(
            NeoDataConfig,
            description=convert_markup_to_ansi(__doc__),
        )
    )

if __name__ == "__main__":
    entrypoint()
