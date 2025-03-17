import tyro
import subprocess

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE

from experiment_config import NeoDataConfig


def calculate_colmap(config: NeoDataConfig) -> None:
    """Main function."""

    output_path = config.get_base_dir()

    # Need both before and after to be present in order to perform final alignment
    for stage in ["before", "after"]:
        # Use COLMAP to get initial positions
        def run_subprocess(command):
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True)
            for line in process.stdout:
                CONSOLE.print(line, end="")
            for line in process.stderr:
                CONSOLE.print(line, end="", style="bold red")
            process.stdout.close()
            process.stderr.close()
            process.wait()

        command = f"ns-process-data images --data {output_path / stage / 'rgb'} --output-dir {output_path / stage}"
        run_subprocess(command)

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