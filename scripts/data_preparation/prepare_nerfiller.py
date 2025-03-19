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
    CONSOLE.print(f"Copying masks, images, depth and transforms.json to {output_path}")
    shutil.copytree(masks_path / "masks", output_path / "masks", dirs_exist_ok=True)
    shutil.copytree(config.data / "images", output_path / "images", dirs_exist_ok=True)
    shutil.copytree(config.data / "depth", output_path / "depth", dirs_exist_ok=True)
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
        frame["depth_file_path"] = frame["file_path"].replace("depth/frame_", "")

    with open(output_path / "transforms.json", "w") as f:
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