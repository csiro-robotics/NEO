import tyro

from typing import Annotated

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.scripts import run_command

from experiment_config import NeoDataConfig


def loopback(config: NeoDataConfig, model_type: Annotated["neo", "neo-big"] = "neo") -> None:
    """Main function."""

    base_dir = config.get_base_dir()

    config_path = str(base_dir / f"before/{model_type}/{config.get_timestamp()}/config.yml").replace(str(config.root_input), 'outputs')

    # Evaluate LERF
    CONSOLE.print("[bold green]Evaluating...[/bold green]")
    command = [
        "ns-loopback lerf",
        f"--load-config {config_path}",
        "--verbose True"
    ]
    command = " ".join(command)

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