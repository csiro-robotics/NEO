import tyro

from pathlib import Path

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE
from neo.utils.commands import run_command

from scripts.experiment_config import NeoDataConfig


def train_gt(config: NeoDataConfig, **kwargs) -> None:
    """Main function that executes the training pipeline."""
    base_dir = Path(config.get_base_dir())

    # Train GT scene
    CONSOLE.print("[bold green]Training GT Nerfacto...[/bold green]")
    command = [
        "ns-train", "nerfacto",
        "--data", base_dir / "after",
        "--output-dir", config.get_output_dir().parent,
        "--experiment-name", f"{config.object_name}/after",
        "--timestamp", config.get_timestamp(),
        "--viewer.quit-on-train-completion", "True",
    ]

    if kwargs.get("camera_optimizer", False):
        command.extend(["--pipeline.model.camera-optimizer.mode", kwargs["camera_optimizer"]])

    command.extend([
        "nerfstudio-data",
        "--orientation-method", "none",
        "--center-method", "none",
        "--auto-scale-poses", "False",
        "--train-split-fraction", "0.99"
    ])

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
