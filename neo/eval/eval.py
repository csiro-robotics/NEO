import os
import tyro
import yaml
import json

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.utils.rich_utils import CONSOLE, status
from nerfstudio.utils.scripts import run_command

from neo.eval.datasets import RealsenseDataset, NerfstudioDataset
from neo.eval.eval_utils import calcualte_error
from neo.eval.metrics import (
    PSNRModule,
    SSIMModule,
    LPIPSModule,
    CorrsModule,
    MAEModule,
)


@dataclass
class EvalConfig:
    model_config_path: Path
    """Path to the edited model config"""
    batch_size: int = 32
    """Size of batch for calculating metrics. Bigger batch mesans faster calculation"""


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
    gt_data_path = Path(str(model_config.data).replace('before', 'after'))

    train_replace_nerf_cmd = [
        "ns-render-neo render-from-transforms",
        f"--load-config {str(model_config.get_base_dir()).replace('neo', 'replace-nerf')}/config.yml",
        f"--output-path {render_output_path}",
        "--output-format images",
        "--image-format png",
        "--rendered-output-names rgb depth",
        f"--transforms-filename {gt_data_path}/transforms.json"
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

    # Create datasets and dataloaders
    gt_dataset = RealsenseDataset(gt_data_path, model_config.experiment_name)
    dataset = NerfstudioDataset(render_output_path, model_config.experiment_name, transforms["fl_x"], transforms["fl_y"])

    gt_dataloader = DataLoader(gt_dataset, config.batch_size, shuffle=False)
    dataloader = DataLoader(dataset, config.batch_size, shuffle=False)

    # Calculate metrics
    results = {model_config.experiment_name: {}}
    calcualte_error(PSNRModule(), dataloader, gt_dataloader, results)
    calcualte_error(MAEModule(), dataloader, gt_dataloader, results)

    # Save results
    eval_output_path = Path(f"eval/{'/'.join(str(model_config.get_base_dir()).split('/')[1:])}")
    os.makedirs(eval_output_path, exist_ok=True)

    with open(eval_output_path / "results.json", "w") as f:
        json.dump(results, f, indent=4)

    CONSOLE.print("[bold green]Finished calculations, results are:")
    CONSOLE.print(results)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(EvalConfig))

if __name__ == '__main__':
    entrypoint()