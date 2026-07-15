import tyro
import os
import shutil

from typing import Annotated
from pathlib import Path

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.eval_utils import eval_setup
from neo.utils.commands import run_command

from scripts.experiment_config import NeoDataConfig

def prepare_gt(config: NeoDataConfig, model_type: Annotated["neo", "neo-big"] = "neo") -> None:
    """Main function."""

    data_dir = config.get_base_dir()
    output_dir = config.get_output_dir()

    # Convert LERF to Neo
    CONSOLE.print("[bold green]Converting LERF to Neo...[/bold green]")
    lerf_config = output_dir / "before" / "lerf-lite" / config.get_timestamp() / "config.yml"
    if not lerf_config.exists():
        raise FileNotFoundError(f"LERF model not found at {lerf_config}. Run the LERF training step first.")
    run_command(["ns-convert", f"to-{model_type}", "--load-config", lerf_config], verbose=True)

    # Move converted model to GT folder
    CONSOLE.print("[bold green]Moving converted model to GT folder...[/bold green]")
    gt_dir = output_dir / "before" / model_type
    out_dir = output_dir / "after" / model_type
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
    shutil.copytree(data_dir / "before", data_dir / "after", dirs_exist_ok=True)
    bbox_path = data_dir / "bbox.json"
    if bbox_path.exists():
        shutil.copy(bbox_path, config_path.parent)
    else:
        raise FileNotFoundError(f"Bounding box file not found at {bbox_path}.")

    # Edit model
    CONSOLE.print("[bold green]Editing...[/bold green]")
    edited_config = output_dir / "after" / model_type / config.get_timestamp() / "config.yml"
    _, pipeline, _, _ = eval_setup(edited_config, eval_num_rays_per_chunk=None, test_mode='test')
    pipeline.apply_changes_hook()

    # Convert neo to nerfacto
    CONSOLE.print("[bold green]Converting Neo to Nerfacto...[/bold green]")
    run_command(["ns-convert", "to-nerfacto", "--load-config", edited_config], verbose=True)

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
