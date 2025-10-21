import tyro
import subprocess

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.scripts import run_command

from experiment_config import NeoDataConfig

def calculate_colmap(config: NeoDataConfig, skip_after: bool = False) -> None:
    """Main function."""

    output_path = config.get_base_dir()

    # Need both before and after to be present in order to perform final alignment
    for stage in ["before", "after"]:

        if stage == "after" and skip_after:
            continue

        # Check and remove any files or folders other than 'rgb'
        for item in (output_path / stage).iterdir():
            if item.name != "rgb" and item.name != "depth":
                CONSOLE.print(f"Removing {item}")
                if item.is_dir():
                    subprocess.run(f"rm -rf {item}", shell=True)
                else:
                    item.unlink()

        # Use COLMAP to get initial positions
        command = f"ns-process-data images --data {output_path / stage / 'rgb'} --output-dir {output_path / stage}"
        run_command(command, verbose=True)

        subprocess.run(f"rm -rf {output_path / stage /'rgb'}", shell=True)
        subprocess.run(f"cp {output_path / stage / 'transforms.json'} {output_path / stage / 'raw_transforms.json'}", shell=True)

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")

    calculate_colmap(
        tyro.cli(
            NeoDataConfig,
            description=convert_markup_to_ansi(__doc__),
        )
    )

if __name__ == "__main__":
    entrypoint()