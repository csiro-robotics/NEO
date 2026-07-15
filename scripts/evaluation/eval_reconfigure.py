import tyro
import shutil

from pathlib import Path

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE
from neo.utils.commands import run_command

from scripts.experiment_config import NeoDataConfig


def eval(config: NeoDataConfig) -> None:
    """Main function that executes the training pipeline."""
    output_dir = config.get_output_dir()
    config_path = output_dir / "before" / "neo" / config.get_timestamp() / "replace-nerf" / "config.yml"

    def evaluate(path: Path) -> None:
        run_command(["ns-eval-neo", "--config-path", path, "--dry-run", "False", "--verbose", "True"])

    # Evaluate LERF
    CONSOLE.print("[bold green]Evaluating...[/bold green]")
    selected_dir = output_dir / "before" / "lerf-lite" / config.get_timestamp() / "config.yml"
    # output_path = str(Path(selected_dir.replace("outputs", "eval")).parents[1])
    # command = [
    #     "ns-eval",
    #     f"--load-config {selected_dir}",
    #     f"--output-path {output_path}/results.json",
    #     f"--render-output-path {output_path.replace('eval', 'renders')}",
    # ]
    shutil.copy(config_path.parents[1] / "bbox.json", selected_dir.parent / "bbox.json")
    evaluate(selected_dir)

    # Evaluate Neo
    evaluate(config_path)

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
