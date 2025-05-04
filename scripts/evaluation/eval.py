import tyro

from pathlib import Path

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.scripts import run_command

from experiment_config import NeoDataConfig


def eval(config: NeoDataConfig) -> None:
    """Main function that executes the training pipeline."""
    base_dir = config.get_base_dir()
    config_path = str(base_dir / f"before/neo/{config.get_timestamp()}/replace-nerf/config.yml").replace(str(config.root_input), 'outputs')

    # Evaluate Neo
    CONSOLE.print("[bold green]Evaluating...[/bold green]")
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

    # Evaluate NeRFiller baseline
    selected_dir = config_path.replace(f"neo/{config.get_timestamp()}/replace-nerf", f"nerfacto-nerfiller/{config.get_timestamp()}")
    command = [
        "ns-eval-neo",
        f"--config-path {selected_dir}",
        "--dry_run False",
        "--verbose True"
    ]
    command = " ".join(command)

    run_command(command, verbose=True)

    # Evaluate NeRFiller
    selected_dir = config_path.replace(f"neo/{config.get_timestamp()}/replace-nerf", f"grid-prior-du/{config.get_timestamp()}")
    command = [
        "ns-eval-neo",
        f"--config-path {selected_dir}",
        "--dry_run False",
        "--verbose True"
    ]
    command = " ".join(command)

    run_command(command, verbose=True)

    # Evaluate NeRFiller without depth
    selected_dir = config_path.replace(f"neo/{config.get_timestamp()}/replace-nerf", f"grid-prior-du-no-depth/{config.get_timestamp()}")
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
