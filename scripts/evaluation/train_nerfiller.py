import tyro

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE
from neo.utils.commands import run_command

from scripts.experiment_config import NeoDataConfig


def train_nerfiller(config: NeoDataConfig) -> None:
    """Main function that executes the training pipeline."""
    output_dir = config.get_output_dir()
    cache_dir = output_dir / "before" / "nerfacto-nerfiller" / config.get_timestamp() / "cache"

    def train(method_name):
        run_command([
            "ns-train", method_name,
            "--data", cache_dir,
            "--output-dir", output_dir.parent,
            "--experiment-name", f"{config.object_name}/before",
            "--timestamp", config.get_timestamp(),
            "--viewer.quit-on-train-completion", "True",
            "--pipeline.model.camera-optimizer.mode", "off",
            "nerfstudio-data",
            "--orientation-method", "none",
            "--center-method", "none",
            "--auto-scale-poses", "False",
            "--train-split-fraction", "0.99",
        ])

    # Train NeRFiller baseline
    CONSOLE.print("[bold green]Training Nerfiller base...[/bold green]")
    train("nerfacto-nerfiller")

    # Train NeRFiller
    CONSOLE.print("[bold green]Training Nerfiller...[/bold green]")
    train("grid-prior-du")


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
