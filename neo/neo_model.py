"""
Neo Model.
"""
import json
import os
import numpy as np

from dataclasses import dataclass, field
from typing import Type, Optional, Tuple, List
from scipy.spatial.transform import Rotation
from pathlib import Path

from neo.utils.bbox_utils import(
    BBoxData,
    BoundingBoxToolbox,
    BoundingBox
)

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig  # for subclassing Nerfacto model


@dataclass
class NeoModelConfig(NerfactoModelConfig):
    """Neo Model Configuration."""

    _target: Type = field(default_factory=lambda: NeoModel)


class NeoModel(NerfactoModel):
    """Neo Model."""

    config: NeoModelConfig

    def populate_modules(self):
        super().populate_modules()

        # Load bbox data
        self.bbox_vis: List[BoundingBox] = []
        self.bboxes = {"bboxes": []}
        self.bbox_dir = self.kwargs['bbox_dir'] / 'bbox.json'
        if not os.path.exists(self.bbox_dir):
            print(f"File bbox.json does not exist. Creating a file.")

            with open(self.bbox_dir, 'w') as f:
                json.dump(self.bboxes, f, indent=2)

        else:
            print(f"File bbox.json already exists. Loading the file.")

            with open(self.bbox_dir, 'r') as f:
                self.bboxes = json.load(f)

        if len(self.bboxes['bboxes']) > 0:

            bbox_data = self.bboxes['bboxes'][0]['data']
            dimensions = tuple(np.array(bbox_data['scale']) * 10)
            rotation = tuple(np.round(np.rad2deg(bbox_data['rotation'])[::-1] * np.array([1, -1, 1])))
            quaternion = tuple(bbox_data['quaternion'])
            position = tuple(np.array(bbox_data['position']) * 10)

            bbox_data = BBoxData(0, dimensions=dimensions, rotation=rotation, quaternion=quaternion, position=position)
            self.bbox_tool = BoundingBoxToolbox("Boxes", move_hook=self.move_box, resize_hook=self.resize_box, bbox_selector_hook=self.bbox_selector_hook, add_box_hook=self.add_box_hook)

        else:
            bbox_data = BBoxData(0, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            self.bbox_tool = BoundingBoxToolbox("Boxes", bbox_data, move_hook=self.move_box, resize_hook=self.resize_box, bbox_selector_hook=self.bbox_selector_hook, add_box_hook=self.add_box_hook)

    def add_box_hook(self, element):
        
        bbox_data = BBoxData(len(self.bbox_vis), [0.1, 0.1, 0.1], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

        bbox_handle = BoundingBox(data=bbox_data.get_nerfstudio_data(), cb_hook=self.display_wireframe)
        bbox_handle.install(self.bbox_tool.viser_server)

        if len(self.bbox_vis) == 0:
            self.bbox = bbox_handle

        self.bbox_vis.append(bbox_handle)
        self.bboxes["bboxes"].append(bbox_data.get_json_data())

    def move_box(self, element):
        self.bbox.update(self.bbox_tool.get_bbox_data().get_nerfstudio_data())

    def resize_box(self, element):
        self.bbox.box_handle.remove()
        self.bbox = BoundingBox(data=self.bbox_tool.get_bbox_data().get_nerfstudio_data(), cb_hook=self.display_wireframe)
        self.bbox_vis[self.bbox_tool.bbox_selector_handle.value] = self.bbox
        self.bbox.install(self.bbox_tool.viser_server)

    def display_wireframe(self, element):
        wireframe = not bool(self.bbox.wireframe)

        self.bbox.box_handle.remove()
        self.bbox = BoundingBox(data=self.bbox.data, cb_hook=self.display_wireframe, wireframe=wireframe)
        self.bbox_vis[self.bbox_tool.bbox_selector_handle.value] = self.bbox
        self.bbox.install(self.bbox_tool.viser_server)

    def bbox_selector_hook(self, element):
        self.bbox = self.bbox_vis[self.bbox_tool.bbox_selector_handle.value]
        self.bbox_tool.update(self.bbox.data)