"""
Neo Model.
"""
import json
import os
import numpy as np

from dataclasses import dataclass, field
from typing import Type, Optional
from scipy.spatial.transform import Rotation
from pathlib import Path

from neo.utils.bbox_utils import(
    BoundingBoxToolbox,
    BoundingBox
)

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig  # for subclassing Nerfacto model
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model


@dataclass
class NeoModelConfig(NerfactoModelConfig):
    """Neo Model Configuration."""

    _target: Type = field(default_factory=lambda: NeoModel)


class NeoModel(NerfactoModel):
    """Neo Model."""

    config: NeoModelConfig

    def populate_modules(self):
        super().populate_modules()

        # Check if the file exists
        bbox_data = None
        filename = self.kwargs['bbox_dir'] / 'bbox.json'
        if not os.path.exists(filename):
            print(f"File bbox.json does not exist. Creating a file.")

            with open(filename, 'w') as f:

                bbox_data = {
                    "labels": [
                        {
                            "id": "0",
                            "data": {
                                "position": [0.0, 0.0, 0.0],
                                "rotation": [0.0, 0.0, 0.0],
                                "quaternion": [1.0, 0.0, 0.0, 0.0],
                                "scale": [0.1, 0.1, 0.1]
                            }
                        }
                    ]
                }

                json.dump(bbox_data, f, indent=2)

        else:
            print(f"File bbox.json already exists. Loading the file.")

            with open(filename, 'r') as f:
                bbox_data = json.load(f)

        bbox_data = bbox_data['labels'][0]['data']
        # self.bbox = Cube(bbox_data['scale'], bbox_data['position'], bbox_data['rotation'])

        dimensions = tuple(np.array(bbox_data['scale']) * 10)
        deg = tuple(np.round(np.rad2deg(bbox_data['rotation'])[::-1] * np.array([1, -1, 1])))
        rot = Rotation.from_euler('xyz', deg, degrees=True)
        wxyz = tuple(rot.as_quat(False))
        position = tuple(np.array(bbox_data['position']) * 10)

        self.bbox_tool = BoundingBoxToolbox("Box", cb_hook=self.move_box, resize_hook=self.resize_box, dimensions=dimensions, rotation=deg, position=position)
        self.bbox_vis = BoundingBox("Box", cb_hook=self.display_wireframe, dimensions=dimensions, wxyz=wxyz, position=position)

    def update_bbox_save_file(self):

        with open(self.config.bbox_path / 'bbofx.json', 'w') as f:

            position = tuple(np.array(self.bbox_tool.position_handle.value) / 10)
            rotation = tuple(np.deg2rad(self.bbox_tool.rotation_handle.value[::-1] * np.array([1, -1, 1])))
            rot = Rotation.from_euler('zyx', rotation, degrees=False)
            quaternion = tuple(rot.as_quat(False))
            dimensions = tuple(np.array(self.bbox_tool.dimensions_handle.value) / 10)

            bbox_data = {
                "labels": [
                    {
                        "id": "0",
                        "data": {
                            "position": position,
                            "rotation": rotation,
                            "quaternion": quaternion,
                            "scale": dimensions
                        }
                    }
                ]
            }

            json.dump(bbox_data, f, indent=2)

    def move_box(self, element):
        self.bbox_vis.box_handle.position = element.position_handle.value
        rot = Rotation.from_euler('xyz', element.rotation_handle.value, degrees=True)
        rot_quat = rot.as_quat(False)
        self.bbox_vis.box_handle.wxyz = rot_quat

        self.update_bbox_save_file()

    def resize_box(self, element):

        assert self.bbox_tool is not None
        
        viser_server = self.bbox_vis.viser_server
        prev_wireframe = bool(self.bbox_vis.wireframe)

        self.bbox_vis.box_handle.remove()
        dimensions = element.dimensions_handle.value
        rot = Rotation.from_euler('xyz', element.rotation_handle.value, degrees=True)
        wxyz = tuple(rot.as_quat(False))
        position = element.position_handle.value

        self.bbox_vis = BoundingBox("Box", cb_hook=self.display_wireframe, dimensions=dimensions, wxyz=wxyz, position=position, wireframe=prev_wireframe)

        assert viser_server is not None

        self.bbox_vis.install(viser_server)
        self.update_bbox_save_file()

    def display_wireframe(self, element):

        viser_server = element.viser_server
        prev_wireframe = not bool(element.wireframe)

        element.box_handle.remove()
        dimensions = self.bbox_tool.dimensions_handle.value
        rot = Rotation.from_euler('xyz', self.bbox_tool.rotation_handle.value, degrees=True)
        wxyz = tuple(rot.as_quat(False))
        position = self.bbox_tool.position_handle.value

        self.bbox_vis = BoundingBox("Box", cb_hook=self.display_wireframe, dimensions=dimensions, wxyz=wxyz, position=position, wireframe=prev_wireframe)

        assert viser_server is not None

        self.bbox_vis.install(viser_server)