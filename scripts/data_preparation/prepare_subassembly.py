import tyro
import json
import shutil
import numpy as np

from typing import Optional
from pathlib import Path

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.scripts import run_command

from scripts.experiment_config import NeoDataConfig


def load_movement_from_manual(manual_file: Path) -> dict | None:
    """Loads movement data from a manual movement.txt file."""
    if not manual_file.exists():
        CONSOLE.print(f"[bold red]Fallback file 'movement.txt' not found. Exiting.[/bold red]")
        return None

    with open(manual_file, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    translation = [0.0, 0.0, 0.0]
    rotation_euler_deg = [0.0, 0.0, 0.0]

    section = None
    for line in lines:
        if "Translation:" in line:
            section = "translation"
        elif "Rotation:" in line:
            section = "rotation"
        elif ':' in line:
            key, value = line.split(':')
            value = float(value.strip())
            key = key.strip().lower()
            if section == "translation":
                if key == "x":
                    translation[0] = np.round(value / 1000, 3)
                elif key == "y":
                    translation[1] = np.round(value / 1000, 3)
                elif key == "z":
                    translation[2] = np.round(value / 1000, 3)
            elif section == "rotation":
                if key == "z":
                    rotation_euler_deg[2] = value  # Only Z axis assumed

    data = {
        'translation': translation,
        'rotation_euler_deg': rotation_euler_deg
    }

    return data


def extract_arm_movement(config: NeoDataConfig) -> None:
    """Main function."""

    output_path = config.get_base_dir()
    movement_file_path = config.get_raw_capture_base() / "movement.txt"
    result = load_movement_from_manual(movement_file_path)

    if result is None:
        raise FileNotFoundError("Movement data could not be loaded from manual file.")

    if output_path.exists() is False:
        output_path.mkdir(parents=True, exist_ok=True)

    # Save output
    with open(output_path / 'object_reconfiguration.json', 'w') as f:
        json.dump(result, f, indent=4)

    CONSOLE.print(f"[bold green]Movement saved to object_reconfiguration.json[/bold green]")


def prepare_subassembly(config: NeoDataConfig, first_step: bool = False, previous_object_name: Optional[str] = None) -> None:
    """Main function."""

    base_dir = config.get_base_dir()

    extract_arm_movement(config)

    if first_step:
        # Copy the data
        shutil.copytree(base_dir.parent / "assembly" / "before", base_dir / "before", dirs_exist_ok=True)
        shutil.copytree(base_dir.parent / "assembly" / "after", base_dir / "after", dirs_exist_ok=True)

        # Prepare LERF-Lite
        lerf_lite_dir = config.get_output_dir().parent
        shutil.copytree(lerf_lite_dir / f"assembly/before/lerf-lite", lerf_lite_dir / f"{config.object_name}/before/lerf-lite", dirs_exist_ok=True)

        config_lerf_lite, _, _, _ = eval_setup(lerf_lite_dir / f"{config.object_name}/before/lerf-lite/{config.timestamp}/config.yml", eval_num_rays_per_chunk=None, test_mode='test')

        config_lerf_lite.experiment_name = config_lerf_lite.experiment_name.replace("assembly", config.object_name)
        new_output_path = config_lerf_lite.pipeline.output_path.replace("assembly/before", f"{config.object_name}/before")
        config_lerf_lite.pipeline.output_path = new_output_path
        config_lerf_lite.pipeline.model.output_folder = new_output_path

        config_lerf_lite.save_config()

    else:
        # Copy the data from previous step
        shutil.copytree(base_dir.parent / "assembly" / "before", base_dir / "before", dirs_exist_ok=True)
        shutil.copytree(base_dir.parent / "assembly" / "after", base_dir / "after", dirs_exist_ok=True)

        lerf_lite_dir = config.get_output_dir().parent

        edited_images_dir = lerf_lite_dir / f"{previous_object_name}/before/nerfacto/{config.timestamp}/cache"
        shutil.copytree(edited_images_dir / "images", base_dir / "before" / "images", dirs_exist_ok=True)
        shutil.copytree(edited_images_dir / "depth", base_dir / "before" / "depth", dirs_exist_ok=True)
        
        # Prepare LERF-Lite
        new_timestamp = [d.name for d in (lerf_lite_dir / f"{previous_object_name}/before/lerf-lite").iterdir() if d.is_dir() and d.name != config.timestamp][0]
        shutil.copytree(lerf_lite_dir / f"{previous_object_name}/before/lerf-lite/{new_timestamp}", lerf_lite_dir / f"{config.object_name}/before/lerf-lite/{new_timestamp}", dirs_exist_ok=True)

        config_lerf_lite, _, _, _ = eval_setup(lerf_lite_dir / f"{config.object_name}/before/lerf-lite/{new_timestamp}/config.yml", eval_num_rays_per_chunk=None, test_mode='test')

        config_lerf_lite.experiment_name = config_lerf_lite.experiment_name.replace(previous_object_name, config.object_name)
        new_output_path = config_lerf_lite.pipeline.output_path.replace(f"{previous_object_name}/before", f"{config.object_name}/before").replace(new_timestamp, config.timestamp)
        config_lerf_lite.pipeline.output_path = new_output_path
        config_lerf_lite.pipeline.model.output_folder = new_output_path
        config_lerf_lite.timestamp = config.timestamp
        config_lerf_lite.data = base_dir / "before"
        config_lerf_lite.pipeline.datamanager.data = base_dir / "before"
        config_lerf_lite.pipeline.datamanager.dataparser.data = base_dir / "before"
        config_lerf_lite.load_dir = Path(str(config_lerf_lite.load_dir).replace(previous_object_name, config.object_name).replace(new_timestamp, config.timestamp).replace("nerfacto", "lerf-lite"))

        shutil.copytree(str(lerf_lite_dir / f"{config.object_name}/before/lerf-lite/{new_timestamp}"), str(lerf_lite_dir / f"{config.object_name}/before/lerf-lite/{config.timestamp}"), dirs_exist_ok=True)
        config_lerf_lite.save_config()
        shutil.rmtree(str(lerf_lite_dir / f"{config.object_name}/before/lerf-lite/{new_timestamp}"))


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")

    prepare_subassembly(
        tyro.cli(
            NeoDataConfig,
            description=convert_markup_to_ansi(__doc__),
        )
    )

if __name__ == "__main__":
    entrypoint()
