import tyro

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE
from neo.utils.commands import run_command

from scripts.experiment_config import NeoDataConfig


def train_nerfiller_no_depth(config: NeoDataConfig, device='cuda:0') -> None:
    """Main function that executes the training pipeline."""
    output_dir = config.get_output_dir()
    cache_dir = output_dir / "before" / "nerfacto-nerfiller-no-depth" / config.get_timestamp() / "cache"

    def train(method_name, output_method_name=None):
        command = [
            "ns-train", method_name,
            "--data", cache_dir,
            "--output-dir", output_dir.parent,
            "--experiment-name", f"{config.object_name}/before",
            "--timestamp", config.get_timestamp(),
        ]
        if method_name == 'grid-prior-du-no-depth':
            command.extend([
                '--pipeline.rgb-inpaint-device', device, 
                "--pipeline.depth-inpaint-device", device, 
                "--pipeline.upscale-device", device
            ])
        if output_method_name is not None:
            command.extend(["--method-name", output_method_name])
        command.extend([
            "--viewer.quit-on-train-completion", "True",
            "--pipeline.model.camera-optimizer.mode", "off",
            "nerfstudio-data",
            "--orientation-method", "none",
            "--center-method", "none",
            "--auto-scale-poses", "False",
            "--train-split-fraction", "0.99",
        ])
        run_command(command)

    # Train NeRFiller baseline
    CONSOLE.print("[bold green]Training Nerfiller base...[/bold green]")
    train("nerfacto-nerfiller", "nerfacto-nerfiller-no-depth")

    # Train NeRFiller without depth
    CONSOLE.print("[bold green]Training Nerfiller...[/bold green]")
    train("grid-prior-du-no-depth")


def entrypoint():
    """Entrypoint for CLI."""
    tyro.extras.set_accent_color("bright_yellow")

    train_nerfiller_no_depth(
        tyro.cli(
            NeoDataConfig,
            description=convert_markup_to_ansi(__doc__),
        )
    )

if __name__ == "__main__":
    entrypoint()
