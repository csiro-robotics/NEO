import tyro

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.eval_utils import eval_setup
from neo.utils.commands import run_command

from scripts.experiment_config import NeoDataConfig

def edit(config: NeoDataConfig) -> None:
    """Main function."""

    output_dir = config.get_output_dir()

    # Convert LERF to Neo
    CONSOLE.print("[bold green]Converting LERF to Neo...[/bold green]")
    lerf_config = output_dir / "before" / "lerf-lite" / config.get_timestamp() / "config.yml"
    if not lerf_config.exists():
        raise FileNotFoundError(f"LERF model not found at {lerf_config}. Run the LERF training step first.")
    run_command(["ns-convert", "to-neo", "--load-config", lerf_config], verbose=True)

    # Load model
    CONSOLE.print("[bold green]Editing...[/bold green]")
    model_config = output_dir / "before" / "neo" / config.get_timestamp() / "config.yml"
    trainer_config, pipeline, ckpt_path, step = eval_setup(model_config, eval_num_rays_per_chunk=None, test_mode='test')
    replace_nerf_config_attrs = [
        "replace_nerf_use_field_distillation",
        "replace_nerf_field_distillation_steps",
        "replace_nerf_use_render_rgb_distillation",
        "replace_nerf_use_render_depth_distillation",
    ]
    for attr in replace_nerf_config_attrs:
        if hasattr(config, attr) and hasattr(pipeline.config, attr):
            setattr(pipeline.config, attr, getattr(config, attr))
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
