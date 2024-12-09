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

def execute_command(command: str, dry_run: bool = False, verbose: bool = False) -> None:
    """
    Execute a command in the shell.
    """
    if not dry_run:
        with status(msg="[bold green]Rendering images", spinner="moon", verbose=verbose):
            run_command(command, verbose=verbose)

    else:
        print("----------------------------------------------------------------------------------------------------------------")
        print("Render command:")
        print(command)
        print("----------------------------------------------------------------------------------------------------------------")


@dataclass
class EvalConfig:
    model_config_path: Path
    """Path to the edited model config"""
    batch_size: int = 32
    """Size of batch for calculating metrics. Bigger batch mesans faster calculation"""
    dry_run: bool = False
    """If True, only print the commands that would be executed"""
    verbose: bool = False
    """If True, print additional information"""


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
    
    # Render full images
    render_command = [
        "ns-render-neo render-from-transforms",
        f"--load-config {str(model_config.get_base_dir()).replace('neo', 'replace-nerf')}/config.yml",
        f"--output-path {render_output_path}/full_edited",
        "--output-format images",
        "--image-format png",
        "--rendered-output-names rgb depth",
        f"--transforms-filename {gt_data_path}/transforms.json"
    ]

    render_command = " ".join(render_command)
    execute_command(render_command, dry_run=config.dry_run, verbose=config.verbose)

    # Crop parameters
    radius = 0.5
    crop = True
    crop_center = (0.27, 0.28, 0.11)
    crop_scale = (0.75, 0.78, 0.40)
    crop_rot = (0.0, 0.0, 0.0)

    # Render cropped images from edited NeRF    
    render_command = [
        "ns-render-neo spiral",
        f"--load-config {str(model_config.get_base_dir()).replace('neo', 'replace-nerf')}/config.yml",
        f"--output-path {render_output_path}/cropped_edited",
        "--output-format images",
        "--image-format png",
        "--rendered-output-names rgb depth",
        f"--radius {radius}",
        f"--crop {crop}",
        f"--crop-center {' '.join(map(str, crop_center))}",
        f"--crop-scale {' '.join(map(str, crop_scale))}",
        f"--crop-rot {' '.join(map(str, crop_rot))}"
    ]

    render_command = " ".join(render_command)
    execute_command(render_command, dry_run=config.dry_run, verbose=config.verbose)

    # Render cropped images from GT NeRF
    render_command = [
        "ns-render-neo spiral",
        f"--load-config {str(model_config.get_base_dir()).replace('neo', 'nerfacto').replace('before', 'after')}/config.yml",
        f"--output-path {render_output_path}/cropped_gt",
        "--output-format images",
        "--image-format png",
        "--rendered-output-names rgb depth",
        f"--radius {radius}",
        f"--crop {crop}",
        f"--crop-center {' '.join(map(str, crop_center))}",
        f"--crop-scale {' '.join(map(str, crop_scale))}",
        f"--crop-rot {' '.join(map(str, crop_rot))}"
    ]

    render_command = " ".join(render_command)
    execute_command(render_command, dry_run=config.dry_run, verbose=config.verbose)

    # Create datasets and dataloaders
    experiment_name = Path(model_config.experiment_name).parent.name
    dataset_full_gt = RealsenseDataset(gt_data_path, experiment_name + "_full")
    dataset_full_edited = NerfstudioDataset(render_output_path / "full_edited", experiment_name + "_full", transforms["fl_x"], transforms["fl_y"])
    dataset_cropped_gt = NerfstudioDataset(render_output_path / "cropped_gt", experiment_name + "_cropped", transforms["fl_x"], transforms["fl_y"])
    dataset_cropped_edited = NerfstudioDataset(render_output_path / "cropped_edited", experiment_name + "_cropped", transforms["fl_x"], transforms["fl_y"])

    dataloader_full_gt = DataLoader(dataset_full_gt, config.batch_size, shuffle=False)
    dataloader_full_edited = DataLoader(dataset_full_edited, config.batch_size, shuffle=False)
    dataloader_cropped_gt = DataLoader(dataset_cropped_gt, config.batch_size, shuffle=False)
    dataloader_cropped_edted = DataLoader(dataset_cropped_edited, config.batch_size, shuffle=False)

    # Calculate metrics
    results = {experiment_name + "_full": {}, experiment_name + "_cropped": {}}
    calcualte_error(PSNRModule(), dataloader_full_edited, dataloader_full_gt, results)
    calcualte_error(MAEModule(), dataloader_full_edited, dataloader_full_gt, results)
    calcualte_error(PSNRModule(), dataloader_cropped_edted, dataloader_cropped_gt, results)
    calcualte_error(MAEModule(), dataloader_cropped_edted, dataloader_cropped_gt, results)

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