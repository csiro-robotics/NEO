import tyro
import shutil

from pathlib import Path

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.scripts import run_command

from experiment_config import NeoDataConfig


def eval(config: NeoDataConfig) -> None:
    """Main function that executes the training pipeline."""
    base_dir = config.get_base_dir()
    config_path = str(base_dir / f"before/neo/{config.get_timestamp()}/replace-nerf/config.yml").replace(str(config.root_input), 'outputs')

    # Evaluate LERF
    CONSOLE.print("[bold green]Evaluating...[/bold green]")
    selected_dir = config_path.replace("neo", "lerf-lite").replace("/replace-nerf", "")
    # output_path = str(Path(selected_dir.replace("outputs", "eval")).parents[1])
    # command = [
    #     "ns-eval",
    #     f"--load-config {selected_dir}",
    #     f"--output-path {output_path}/results.json",
    #     f"--render-output-path {output_path.replace('eval', 'renders')}",
    # ]
    shutil.copy(Path(config_path).parents[1] / "bbox.json", Path(selected_dir).parent / "bbox.json")
    command = [
        "ns-eval-neo",
        f"--config-path {selected_dir}",
        "--dry_run False",
        "--verbose True"
    ]
    command = " ".join(command)

    run_command(command, verbose=True)

    # Evaluate Neo
    command = [
        "ns-eval-neo",
        f"--config-path {config_path}",
        "--dry_run False",
        "--verbose True"
    ]
    command = " ".join(command)

    run_command(command, verbose=True)

    # Evaluate Neo-Big
    selected_dir = config_path.replace("neo", "neo-big")
    command = [
        "ns-eval-neo",
        f"--config-path {selected_dir}",
        "--dry_run False",
        "--verbose True"
    ]
    command = " ".join(command)

    run_command(command, verbose=True)


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
