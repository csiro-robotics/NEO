import tyro
import os
import shutil

from typing import Annotated
from pathlib import Path

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.scripts import run_command

from experiment_config import NeoDataConfig

def prepare_gt(config: NeoDataConfig, model_type: Annotated["neo", "neo-big"] = "neo") -> None:
    """Main function."""

    base_dir = config.get_base_dir()

    # Convert LERF to Neo
    CONSOLE.print("[bold green]Converting LERF to Neo...[/bold green]")
    assert os.path.exists(f"{str(base_dir).replace(str(config.root_input), 'outputs')}/before/lerf-lite"), "LERF model not found. Please run the training script first."
    run_command(f"ns-convert to-{model_type} --load-config {str(base_dir).replace(str(config.root_input), 'outputs')}/before/lerf-lite/{config.get_timestamp()}/config.yml", verbose=True)

    # Move converted model to GT folder
    CONSOLE.print("[bold green]Moving converted model to GT folder...[/bold green]")
    gt_dir = str(base_dir / f"before/{model_type}").replace(str(config.root_input), 'outputs')
    out_dir = Path(gt_dir.replace("before", "after"))
    if not os.path.exists(out_dir.parent):
        os.makedirs(out_dir.parent)

    shutil.copytree(gt_dir, out_dir, dirs_exist_ok=True)
    shutil.rmtree(gt_dir)

    # Update config.yml inside the moved directory
    config_path = out_dir / f"{config.get_timestamp()}/config.yml"
    if config_path.exists():
        text = config_path.read_text(encoding="utf-8")
        new_text = text.replace("before", "after")
        if new_text != text:
            config_path.write_text(new_text, encoding="utf-8")
            CONSOLE.print(f"[bold green]Patched config: {config_path}[/bold green]")
        else:
            CONSOLE.print(f"[bold yellow]No occurrences of 'before' found in {config_path}.[/bold yellow]")
    else:
        CONSOLE.print(f"[bold red]Config file not found at {config_path}.[/bold red]")

    # Copy the data
    shutil.copytree(config.get_base_dir() / "before", config.get_base_dir() / "after", dirs_exist_ok=True)
    bbox_path = config.get_base_dir() / "bbox.json"
    if bbox_path.exists():
        shutil.copy(bbox_path, config_path.parent)
    else:
        raise FileNotFoundError(f"Bounding box file not found at {bbox_path}.")

    # Edit model
    CONSOLE.print("[bold green]Editing...[/bold green]")
    _, pipeline, _, _ = eval_setup(Path(str(config.get_base_dir() / f"after/{model_type}/{config.get_timestamp()}/config.yml").replace(str(config.root_input), 'outputs')), eval_num_rays_per_chunk=None, test_mode='test')
    pipeline.apply_changes_hook()

    # Convert neo to nerfacto
    CONSOLE.print("[bold green]Converting Neo to Nerfacto...[/bold green]")
    run_command(f"ns-convert to-nerfacto --load-config {str(config.get_base_dir()).replace(str(config.root_input), 'outputs')}/after/{model_type}/{config.get_timestamp()}/config.yml", verbose=True)

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")

    prepare_gt(
        tyro.cli(
            NeoDataConfig,
            description=convert_markup_to_ansi(__doc__),
        )
    )

if __name__ == "__main__":
    entrypoint()