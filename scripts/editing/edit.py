import tyro
import os

from typing import Annotated
from pathlib import Path

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.scripts import run_command

from experiment_config import NeoDataConfig

def edit(config: NeoDataConfig, model_type: Annotated["neo", "neo-big"] = "neo") -> None:
    """Main function."""

    base_dir = config.get_base_dir()

    # Convert LERF to Neo
    CONSOLE.print("[bold green]Converting LERF to Neo...[/bold green]")
    assert os.path.exists(f"{str(base_dir).replace(str(config.root_input), 'outputs')}/before/lerf-lite"), "LERF model not found. Please run the training script first."
    run_command(f"ns-convert to-{model_type} --input {str(base_dir).replace(str(config.root_input), 'outputs')}/before/lerf-lite/{config.get_timestamp()}", verbose=True)

    # Load model
    CONSOLE.print("[bold green]Editing...[/bold green]")
    config, pipeline, ckpt_path, step = eval_setup(Path(str(base_dir / f"before/{model_type}/{config.get_timestamp()}/config.yml").replace(str(config.root_input), 'outputs')), eval_num_rays_per_chunk=None, test_mode='test')
    pipeline.apply_changes_hook()

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")

    edit(
        tyro.cli(
            NeoDataConfig,
            description=convert_markup_to_ansi(__doc__),
        )
    )

if __name__ == "__main__":
    entrypoint()