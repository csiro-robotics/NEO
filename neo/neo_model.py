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

    def update_bbox_data(self):
        
        # for box in self.bbox_vis:
        #     position = tuple(np.array(self.bbox_tool.position_handle.value) / 10)
        #     rotation = tuple(np.deg2rad(self.bbox_tool.rotation_handle.value[::-1] * np.array([1, -1, 1])))
        #     rot = Rotation.from_euler('zyx', rotation)
        #     quaternion = tuple(rot.as_quat(True))
        #     dimensions = tuple(np.array(self.bbox_tool.dimensions_handle.value) / 10)

        #     self.bboxes.append({"data": {"position": position, "rotation": rotation, "quaternion": quaternion, "scale": dimensions}})

        # with open(self.bbox_dir, 'w') as f:
        #         bbox_data = {"boxes": self.bboxes}
        #         json.dump(bbox_data, f, indent=2)

        print("Update data!")

    def add_box_hook(self, element):
        
        bbox_data = BBoxData(len(self.bbox_vis), [0.1, 0.1, 0.1], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

        bbox_handle = BoundingBox(data=bbox_data.get_nerfstudio_data(), cb_hook=self.display_wireframe)
        bbox_handle.install(element.viser_server)

        self.bbox_vis.append(bbox_handle)
        self.bboxes["bboxes"].append(bbox_data.get_json_data())

    def move_box(self, element):
        bbox = self.bbox_vis[element.bbox_selector_handle.value]
        bbox.update(self.bbox_tool.get_bbox_data().get_nerfstudio_data())

    def resize_box(self, element):
        bbox = self.bbox_vis[self.bbox_tool.bbox_selector_handle.value]
        prev_wireframe = bool(bbox.wireframe)

        bbox.box_handle.remove()
        bbox = BoundingBox(data=element.get_bbox_data().get_nerfstudio_data(), cb_hook=self.display_wireframe, wireframe=prev_wireframe)
        bbox.install(element.viser_server)

    def display_wireframe(self, element):
        bbox = self.bbox_vis[self.bbox_tool.bbox_selector_handle.value]
        prev_wireframe = not bool(element.wireframe)

        element.box_handle.remove()
        bbox = BoundingBox(data=self.bbox_tool.get_bbox_data().get_nerfstudio_data(), cb_hook=self.display_wireframe, wireframe=prev_wireframe)
        bbox.install(self.bbox_tool.viser_server)

    def bbox_selector_hook(self, element):
        bbox = self.bbox_vis[self.bbox_tool.bbox_selector_handle.value]

        self.bbox_tool.update_data(bbox.data)