import torch
import cv2
import numpy as np
import os
import subprocess

from torch.utils.data import Dataset
from typing import List
from pathlib import Path

from neo.eval.eval_utils import depth_to_distance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseDataset(Dataset):
    """Base dataset class"""

    def __init__(self, path: Path, name: str) -> None:
        super().__init__()

        self.path = path
        self.name = name
        self.rgb_image_paths = []

    def __len__(self):

        return len(self.rgb_image_paths)

    def load_images_from_folder(self, folder_name: str) -> List[Path]:

        images = []
        for file_path in (self.path / folder_name).glob("*.png"):
            img = file_path
            if img is not None:
                images.append(img)

        assert len(images) > 0, f"Folder {self.path / folder_name} does not exist or contains no '.png' data!"

        images = sorted(images)

        return images
    
    def read_txt(self, path: Path) -> List[int]:

        content = []
        with open(path, 'r') as f:
            for line in f:
                if line != '\n':
                    content.append(float(line))

        return content

    def generate_video(self, experiment_name) -> None:

        os.makedirs(f"videos/{experiment_name}", exist_ok=True)

        name = self.name.replace("\\", "").replace("{", "").replace("}", "")

        command = [
            'ffmpeg',
            '-framerate', '24',
            '-pattern_type', 'glob',
            '-i', f"{self.rgb_image_paths[0].parent}/*.png",
            '-frames:v', '120',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-y', f'videos/{experiment_name}/{name}.mp4'
        ]
        
        subprocess.run(command) 


class RealsenseDataset(BaseDataset):
    """Dataset containing images from Realsense."""

    def __init__(self, path: Path, name: str):
        super().__init__(path, name)

        # Load images
        self.rgb_image_paths = self.load_images_from_folder("images")
        self.depth_image_paths = self.load_images_from_folder("depth")

    def __getitem__(self, idx: int):

        rgb_img = torch.tensor(cv2.imread(str(self.rgb_image_paths[idx])), device=device).permute(2, 0, 1) / 255
        depth_img = torch.tensor(np.int64(cv2.imread(str(self.depth_image_paths[idx]), cv2.IMREAD_ANYDEPTH)), device=device) / 1000
        
        return [rgb_img, depth_img]
    
    
class NerfstudioDataset(BaseDataset):
    """Dataset of images rendered from Nerfstudio."""

    def __init__(self, path: Path, name: str, fl_x: float, fl_y: float):
        super().__init__(path, name)

        self.fl_x = fl_x
        self.fl_y = fl_y

        # Load images
        self.rgb_image_paths = self.load_images_from_folder("rgb")
        self.depth_image_paths = self.load_images_from_folder("depth")

    def __getitem__(self, idx: int):

        rgb_img = torch.tensor(cv2.imread(str(self.rgb_image_paths[idx])), device=device).permute(2, 0, 1) / 255
        depth_img = torch.tensor(np.int64(cv2.imread(str(self.depth_image_paths[idx]), cv2.IMREAD_ANYDEPTH)), device=device) / 1000
        depth_img = depth_img[None, None, ...]
        depth_img = depth_to_distance(depth_img, fl_x=self.fl_x, fl_y=self.fl_y)
        depth_img = depth_img.squeeze(0).squeeze(0)
        
        return [rgb_img, depth_img]