import tyro

from pathlib import Path

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.eval_utils import eval_setup

from experiment_config import NeoDataConfig

def edit(config: NeoDataConfig) -> None:
    """Main function."""

    base_dir = config.get_base_dir()

    # Load model
    CONSOLE.print("[bold green]Editing...[/bold green]")
    config, pipeline, ckpt_path, step = eval_setup(Path(str(base_dir / f"before/neo/{config.get_timestamp()}/config.yml").replace(str(config.root_input), 'outputs')), eval_num_rays_per_chunk=None, test_mode='test')
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