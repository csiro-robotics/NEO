import tyro

from pathlib import Path

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.scripts import run_command

from experiment_config import NeoDataConfig


def train_nerfiller(config: NeoDataConfig) -> None:
    """Main function that executes the training pipeline."""
    base_dir = Path(config.get_base_dir())

    # Train LERF
    CONSOLE.print("[bold green]Training Nerfiller base...[/bold green]")
    command = [
        "ns-train nerfacto-nerfiller",
        f"--data {str(base_dir).replace(str(config.root_input), 'outputs')}/before/nerfacto-nerfiller/{config.get_timestamp()}/cache",
        f"--output-dir {str(base_dir.parent).replace(str(config.root_input), 'outputs')}",
        f"--experiment-name {config.object_name}/before",
        f"--timestamp {config.get_timestamp()}",
        "--viewer.quit-on-train-completion True",
        "--pipeline.model.camera-optimizer.mode off",
        "--pipeline.datamanager.masks-on-gpu True",
        "--pipeline.datamanager.images-on-gpu True",
        "nerfstudio-data",
        "--orientation-method none",
        "--center-method none",
        "--auto-scale-poses False",
        "--train-split-fraction 0.99"
    ]
    command = " ".join(command)

    run_command(command, verbose=True)

    CONSOLE.print("[bold green]Training Nerfiller...[/bold green]")
    command = [
        "ns-train grid-prior-du",
        f"--data {str(base_dir).replace(str(config.root_input), 'outputs')}/before/nerfacto-nerfiller/{config.get_timestamp()}/cache",
        f"--output-dir {str(base_dir.parent).replace(str(config.root_input), 'outputs')}",
        f"--experiment-name {config.object_name}/before",
        f"--timestamp {config.get_timestamp()}",
        "--viewer.quit-on-train-completion True",
        "--pipeline.model.camera-optimizer.mode off",
        "--pipeline.datamanager.masks-on-gpu True",
        "--pipeline.datamanager.images-on-gpu True",
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

    train_nerfiller(
        tyro.cli(
            NeoDataConfig,
            description=convert_markup_to_ansi(__doc__),
        )
    )

if __name__ == "__main__":
    entrypoint()
