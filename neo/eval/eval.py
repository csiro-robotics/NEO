import os
import tyro
import yaml
import json

from dataclasses import dataclass
from pathlib import Path

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.utils.rich_utils import CONSOLE, status
from nerfstudio.utils.scripts import run_command

from neo.eval.datasets import RealsenseDataset, NerfstudioDataset


@dataclass
class EvalConfig:
    model_config_path: Path
    """Path to the edited model config"""


def main(
        config: EvalConfig
):
    """
    Evaluation pipeline for Neo models.
    """

    # Create eval path
    os.makedirs('eval', exist_ok=True)

    # Load edited model config
    model_config = yaml.load(config.model_config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(model_config, TrainerConfig)

    with open(model_config.data / 'transforms.json') as f:
        transforms = json.load(f)

    # Get rendering command
    render_output_path = Path(f"renders/{'/'.join(str(model_config.get_base_dir()).split('/')[1:])}")

    train_replace_nerf_cmd = [
        "ns-render-neo render-from-transforms",
        f"--load-config {model_config.get_base_dir() / 'config.yml'}",
        f"--output-path {render_output_path}",
        "--output-format images",
        "--image-format png",
        "--rendered-output-names rgb depth",
        f"--transforms-filename {model_config.data / 'transforms.json'}"
    ]

    train_replace_nerf_cmd = " ".join(train_replace_nerf_cmd)

    # Render images
    dry_run = True
    verbose = True
    
    if not dry_run:
        with status(msg="[bold green]Rendering images", spinner="moon", verbose=verbose):
            run_command(train_replace_nerf_cmd, verbose=verbose)

    else:
        print("----------------------------------------------------------------------------------------------------------------")
        print("Render command:")
        print(train_replace_nerf_cmd)
        print("----------------------------------------------------------------------------------------------------------------")

    # Create datasets
    gt_dataset = RealsenseDataset(model_config.data, model_config.experiment_name)
    dataset = NerfstudioDataset(render_output_path, model_config.experiment_name, transforms["fl_x"], transforms["fl_y"])

    print(gt_dataset[0])
    print(dataset[0])

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(EvalConfig))

if __name__ == '__main__':
    entrypoint()