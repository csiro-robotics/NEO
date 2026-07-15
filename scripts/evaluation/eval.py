import tyro

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE
from neo.utils.commands import run_command

from scripts.experiment_config import NeoDataConfig


def eval(config: NeoDataConfig) -> None:
    """Main function that executes the training pipeline."""
    output_dir = config.get_output_dir()
    config_path = output_dir / "before" / "neo" / config.get_timestamp() / "replace-nerf" / "config.yml"

    def evaluate(path):
        run_command(["ns-eval-neo", "--config-path", path, "--dry-run", "False", "--verbose", "True"])

    # Evaluate Neo
    CONSOLE.print("[bold green]Evaluating...[/bold green]")
    evaluate(config_path)

    # Evaluate NeRFiller baseline
    evaluate(output_dir / "before" / "nerfacto-nerfiller" / config.get_timestamp() / "config.yml")

    # Evaluate NeRFiller
    evaluate(output_dir / "before" / "grid-prior-du" / config.get_timestamp() / "config.yml")

    # Evaluate NeRFiller without depth
    evaluate(output_dir / "before" / "grid-prior-du-no-depth" / config.get_timestamp() / "config.yml")


def entrypoint():
    """Entrypoint for CLI."""
    tyro.extras.set_accent_color("bright_yellow")

    eval(
        tyro.cli(
            NeoDataConfig,
            description=convert_markup_to_ansi(__doc__),
        )
    )

if __name__ == "__main__":
    entrypoint()
