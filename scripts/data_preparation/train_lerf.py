import tyro

from pathlib import Path

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.scripts import run_command

from experiment_config import NeoDataConfig


def train_lerf(config: NeoDataConfig) -> None:
    """Main function that executes the training pipeline."""
    base_dir = Path(config.get_base_dir())

    # Train LERF
    CONSOLE.print("[bold green]Training LERF...[/bold green]")
    command = [
        "ns-train lerf-lite",
        f"--data {base_dir}/before",
        f"--output-dir {str(base_dir.parent).replace(str(config.root_input), 'outputs')}",
        f"--experiment-name {config.object_name}/before",
        f"--timestamp {config.get_timestamp()}",
        "--viewer.quit-on-train-completion True",
        "nerfstudio-data",
        "--orientation-method none",
        "--center-method none",
        "--auto-scale-poses False",
        "--train-split-fraction 0.99"
    ]
    command = " ".join(command)

    run_command(command, verbose=True)

    # Convert LERF to Neo
    CONSOLE.print("[bold green]Converting LERF to Neo...[/bold green]")
    run_command(f"ns-convert to-neo --input {str(base_dir).replace(str(config.root_input), 'outputs')}/before/lerf-lite/{config.get_timestamp()}", verbose=True)

def entrypoint():
    """Entrypoint for CLI."""
    tyro.extras.set_accent_color("bright_yellow")

    train_lerf(
        tyro.cli(
            NeoDataConfig,
            description=convert_markup_to_ansi(__doc__),
        )
    )

if __name__ == "__main__":
    entrypoint()
