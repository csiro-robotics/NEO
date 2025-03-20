import os
import tyro
import yaml
import json
import shutil
import numpy as np

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Union, List, Tuple, Dict
from typing_extensions import Annotated

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.utils.rich_utils import CONSOLE, status
from nerfstudio.utils.scripts import run_command

from neo.utils.sparse_dataset import generate_masks
from neo.render.camera_paths import generate_hemisphere_transforms
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
class EvalBaseConfig:
    """
    Base config for evaluation.
    """

    config_path: Path
    """Path to the edited Neo model config"""
    batch_size: int = 32
    """Size of batch for calculating metrics. Bigger batch means faster calculation"""
    dry_run: bool = False
    """If True, only print the commands that would be executed"""
    verbose: bool = False
    """If True, print additional information"""
    crop: bool = True
    """If True, apply cropping to the rendered images"""
    crop_radius: float = 0.5
    """Radius for cropping the rendered images"""
    crop_center: tuple = (0.27, 0.28, 0.11)
    """Center coordinates for cropping the rendered images"""
    crop_scale: tuple = (0.75, 0.78, 0.40)
    """Scale factors for cropping the rendered images"""
    crop_rot: tuple = (0.0, 0.0, 0.0)
    """Rotation angles for cropping the rendered images"""

    def execute_command(self, command: str) -> None:
        """
        Execute a command in the shell.
        """
        if not self.dry_run:
            with status(msg="[bold green]Rendering images", spinner="moon", verbose=self.verbose):
                run_command(command, verbose=self.verbose)

        else:
            print("----------------------------------------------------------------------------------------------------------------")
            print("Render command:")
            print(command)
            print("----------------------------------------------------------------------------------------------------------------")

    def render_full(self, render_output_path: Path, is_edited: bool = True) -> None:
        """
        Render full images.
        """
        render_command = [
            "ns-render-neo render-from-transforms",
            f"--load-config {self.config_path if is_edited else str(self.config_path).replace('before', 'after').replace(self.model_config.method_name, 'nerfacto')}",
            f"--output-path {render_output_path}/full_{'edited' if is_edited else 'gt'}",
            "--output-format images",
            "--image-format png",
            "--rendered-output-names rgb depth",
            f"--transforms-filename {render_output_path}/full_gt/transforms.json"
        ]

        render_command = " ".join(render_command)
        self.execute_command(render_command)

    def render_cropped(self, render_output_path: Path, is_edited: bool = True) -> None:
        """
        Render cropped images.
        """
        render_command = [
            "ns-render-neo spiral",
            f"--load-config {self.config_path if is_edited else str(self.config_path).replace('before', 'after').replace(self.model_config.method_name, 'nerfacto')}",
            f"--output-path {render_output_path}/cropped_{'edited' if is_edited else 'gt'}",
            "--output-format images",
            "--image-format png",
            "--rendered-output-names rgb depth",
            f"--radius {self.crop_radius}",
            f"--crop {self.crop}",
            f"--crop-center {' '.join(map(str, self.crop_center))}",
            f"--crop-scale {' '.join(map(str, self.crop_scale))}",
            f"--crop-rot {' '.join(map(str, self.crop_rot))}"
        ]

        render_command = " ".join(render_command)
        self.execute_command(render_command)

    @abstractmethod
    def create_dataloaders(self, render_output_path: Path, gt_data_path: Path, exp_names: Tuple[str]) -> List[Tuple[DataLoader, DataLoader]]:
        """
        Create dataloaders and datasets.
        """
        pass

    def generate_masks_for_eval(self, transforms: Dict[str, float], render_output_path: Path, exp_names: Tuple[str]) -> None:
        """
        Generate masks for evaluation.
        """
        datamanager = self.model_config.pipeline.datamanager.setup(
            device='cuda:0', test_mode='test', world_size=1, local_rank=0
        )
        dataparser_output = datamanager.dataparser._generate_dataparser_outputs()

        bboxes_path = (Path(str(self.config_path).replace(self.model_config.method_name, 'neo')).parent / "bbox.json")
        if bboxes_path.is_file():
            bboxes = json.load(bboxes_path.open())['bboxes']
        else:
            raise FileNotFoundError(f"File {bboxes_path} not found. Check if the file exists or train Neo model first.")

        generate_masks(camera_data=transforms,
                       bboxes=bboxes,
                       output_path=render_output_path / f"{exp_names[0]}_masks",
                       dataparser_transform=dataparser_output.dataparser_transform,
                       dataparser_scale=dataparser_output.dataparser_scale)
        
        # Load the transforms.json file as camera_data
        transforms_path = Path(self.model_config.pipeline.datamanager.data) / "transforms.json"
        with open(transforms_path, "r") as f:
            camera_data = json.load(f)

        # Get dataparser transformation matrix
        dataparser_transform = dataparser_output.dataparser_transform.cpu().detach().numpy()
        if "applied_transform" in camera_data:
            dataparser_transform = dataparser_transform @ np.linalg.inv(np.concatenate([np.array(camera_data["applied_transform"]), np.array([[0, 0, 0, 1]])], axis=0))   
        
        
        frames = generate_hemisphere_transforms(radius=self.crop_radius,
                                                num_layers=3,
                                                num_per_layer=40,
                                                origin=self.crop_center,
                                                dataparser_scale=dataparser_output.dataparser_scale,
                                                dataparser_transform=dataparser_transform)
        image_names = [f"frame_{num:05d}" for num in range(len(frames))]

        frames_with_names = []
        for frame, name in zip(frames, image_names):
            frames_with_names.append({"file_path": f"images/{name}.png", "transform_matrix": frame})

        transforms["frames"] = frames_with_names

        generate_masks(camera_data=transforms,
                       bboxes=bboxes,
                       output_path=render_output_path / f"{exp_names[1]}_masks",
                       dataparser_transform=dataparser_output.dataparser_transform,
                       dataparser_scale=dataparser_output.dataparser_scale)

    def main(self) -> None:
        """
        Evaluation pipeline for models.
        """
        # Create eval path
        os.makedirs('eval', exist_ok=True)

        # Create experiment names
        exp_names = ("full", "cropped")

        # Load edited model config
        self.model_config = yaml.load(self.config_path.read_text(), Loader=yaml.Loader)
        assert isinstance(self.model_config, TrainerConfig)

        # Get rendering command
        base_path = str(self.model_config.get_base_dir())
        gt_data_path = Path(base_path.replace('outputs', 'data').replace('before', 'after')).parents[1]
        render_output_path = Path(base_path.replace('outputs', 'renders'))

        # Load focal lengths and transforms
        with open(self.model_config.data / 'transforms.json') as f:
            transforms = json.load(f)
            if 'frames' in transforms and 'fl_x' in transforms['frames'][0]:
                self.focal_lengths = (transforms['frames'][0]['fl_x'], transforms['frames'][0]['fl_y'])
            else:
                self.focal_lengths = (transforms['fl_x'], transforms['fl_y'])

            with open(gt_data_path / 'transforms.json') as f:
                transforms = json.load(f)
                if 'frames' in transforms and 'fl_x' in transforms['frames'][0]:
                    self.gt_focal_lengths = (transforms['frames'][0]['fl_x'], transforms['frames'][0]['fl_y'])
                else:
                    self.gt_focal_lengths = (transforms['fl_x'], transforms['fl_y'])

        # Copy GT data to render output path
        shutil.copytree(gt_data_path, render_output_path / f"{exp_names[0]}_gt", dirs_exist_ok=True)

        # Render full images
        self.render_full(render_output_path)

        # Render cropped images
        self.render_cropped(render_output_path)
        self.render_cropped(render_output_path, is_edited=False)

        # Generate masks
        self.generate_masks_for_eval(transforms, render_output_path, exp_names)

        # Create datasets and dataloaders
        dataloaders = self.create_dataloaders(render_output_path, exp_names)

        # Calculate metrics
        metrics = [PSNRModule(), MAEModule()]
        results = {exp_names[0]: {}, exp_names[1]: {}}
        for dataloader in dataloaders:
            for mertic in metrics:
                calcualte_error(mertic, dataloader[0], dataloader[1], results)
                calcualte_error(mertic, dataloader[0], dataloader[1], results, masked=True)

        # Save results
        eval_output_path = Path(f"eval/{'/'.join(str(self.model_config.get_base_dir()).split('/')[1:])}")
        os.makedirs(eval_output_path, exist_ok=True)

        with open(eval_output_path / "results.json", "w") as f:
            json.dump(results, f, indent=4)

        CONSOLE.print("[bold green]Finished calculations, results are:")
        CONSOLE.print(results)


@dataclass
class EvalNerfstudioConfig(EvalBaseConfig):

    def create_dataloaders(self, render_output_path: Path, exp_names: Tuple[str]) -> List[Tuple[DataLoader, DataLoader]]:
        """
        Create dataloaders and datasets.
        """

        dataset_full_edited = NerfstudioDataset(render_output_path / f"{exp_names[0]}_edited", exp_names[0], self.focal_lengths)
        dataset_full_gt = RealsenseDataset(render_output_path / f"{exp_names[0]}_gt", exp_names[0])
        
        dataset_cropped_edited = NerfstudioDataset(render_output_path / f"{exp_names[1]}_edited", exp_names[1], self.focal_lengths)
        dataset_cropped_gt = NerfstudioDataset(render_output_path / f"{exp_names[1]}_gt", exp_names[1], self.gt_focal_lengths)

        datasets = [(dataset_full_edited, dataset_full_gt), (dataset_cropped_edited, dataset_cropped_gt)]

        dataloaders = []
        for dataset in datasets:
            dataloaders.append((DataLoader(dataset[0], self.batch_size, shuffle=False), 
                                DataLoader(dataset[1], self.batch_size, shuffle=False)))
            
        return dataloaders


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[EvalNerfstudioConfig, tyro.conf.subcommand(name="nerfstudio")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == '__main__':
    entrypoint()