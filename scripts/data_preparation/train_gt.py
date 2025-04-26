import tyro

from pathlib import Path

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.scripts import run_command

from experiment_config import NeoDataConfig


def train_gt(config: NeoDataConfig) -> None:
    """Main function that executes the training pipeline."""
    base_dir = Path(config.get_base_dir())

    # Train GT scene
    CONSOLE.print("[bold green]Training GT Nerfacto...[/bold green]")
    command = [
        "ns-train nerfacto",
        f"--data {base_dir}/after",
        f"--output-dir {str(base_dir.parent).replace(str(config.root_input), 'outputs')}",
        f"--experiment-name {config.object_name}/after",
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

def entrypoint():
    """Entrypoint for CLI."""
    tyro.extras.set_accent_color("bright_yellow")

    train_gt(
        tyro.cli(
            NeoDataConfig,
            description=convert_markup_to_ansi(__doc__),
        )
    )

if __name__ == "__main__":
    entrypoint()
