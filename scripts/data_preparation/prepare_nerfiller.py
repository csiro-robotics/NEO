import json
import yaml
import shutil
import os

from pathlib import Path

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.utils.rich_utils import CONSOLE

import tyro

from pathlib import Path

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE

from experiment_config import NeoDataConfig


def prepare_nerfiller(config: NeoDataConfig) -> None:

    base_dir = config.get_base_dir()
    config_path = Path(str(base_dir / f"before/neo/{config.get_timestamp()}/config.yml").replace(str(config.root_input), 'outputs'))

    # Load save config
    CONSOLE.print(f"Loading config from {config_path}")
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)

    masks_path = config.get_base_dir() / "cache"
    if not masks_path.exists():
        CONSOLE.print(f"[bold red]Error: Masks path {masks_path} does not exist. Train Neo model first.")
        exit(1)
    output_path = Path(str(masks_path).replace("neo", "nerfacto-nerfiller"))

    # Copy masks, images, depth and transforms.json to the new folder
    CONSOLE.print(f"[bold green]Preparing Nerfiller...[/bold green]")
    CONSOLE.print(f"Copying masks, images, depth and transforms.json to {output_path}")
    
    def copy_and_rename(src_dir, dst_dir):
        dst_dir.mkdir(parents=True, exist_ok=True)
        for file in src_dir.iterdir():
            if file.is_file():
                new_name = file.name
                if new_name.startswith("train_"):
                    new_name = new_name[len("train_") :]
                elif new_name.startswith("eval_"):
                    new_name = new_name[len("eval_") :]
                shutil.copy(file, dst_dir / new_name)

    # Copy masks, images, and depth, renaming files to remove "train_" and "eval_" from the beginning
    copy_and_rename(masks_path / "masks", output_path / "masks")
    copy_and_rename(config.data / "images", output_path / "images")
    copy_and_rename(config.data / "depth", output_path / "depth")
    shutil.copy(config.data / "transforms.json", output_path / "transforms.json")

    # Remove only files from data_path containing "dedicated" in their name
    for file in output_path.glob("**/*dedicated*"):
        if file.is_file():
            os.remove(file)

    # Read json file update and save it
    CONSOLE.print(f"Updating transforms.json")
    with open(output_path / "transforms.json", "r") as f:
        transforms = json.load(f)

    for frame in transforms["frames"]:
        frame["mask_path"] = frame["file_path"].replace("images", "masks")
        frame["depth_file_path"] = frame["file_path"].replace("images", "depth")

    with open(output_path / "transforms.json", "w") as f:
        json.dump(transforms, f, indent=4)

    # Prepare Nerfiller without depth
    CONSOLE.print(f"[bold green]Preparing Nerfiller without depth...[/bold green]")
    output_path_no_depth = Path(str(output_path).replace("nerfacto-nerfiller", "nerfacto-nerfiller-no-depth"))
    shutil.copytree(output_path, output_path_no_depth, dirs_exist_ok=True)
    shutil.rmtree(output_path_no_depth / "depth")

    # Read json file update and save it
    CONSOLE.print(f"Updating transforms.json")
    with open(output_path_no_depth / "transforms.json", "r") as f:
        transforms = json.load(f)

    for frame in transforms["frames"]:
        del frame["depth_file_path"]

    with open(output_path_no_depth / "transforms.json", "w") as f:
        json.dump(transforms, f, indent=4)

    CONSOLE.print(f"[bold green]Done![/bold green]")


def entrypoint():
    """Entrypoint for CLI."""
    tyro.extras.set_accent_color("bright_yellow")

    prepare_nerfiller(
        tyro.cli(
            NeoDataConfig,
            description=convert_markup_to_ansi(__doc__),
        )
    )

if __name__ == "__main__":
    entrypoint()