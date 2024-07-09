"""
Utils for displaying bbox in the viewer
"""
from __future__ import annotations

import time
import numpy as np
import trimesh.creation

from scipy.spatial.transform import Rotation
from dataclasses import dataclass
from typing import Tuple, List, Callable, Any

from nerfstudio.viewer.viewer_elements import ViewerElement
from viser import (
    MeshHandle,
    GuiInputHandle,
    ViserServer,
    Icon,
)


@dataclass
class BBoxData:

    idx: int
    """Index of the bounding box"""
    dimensions: List[float, float, float]
    """Dimensions of the bounding box"""
    euler: List[float, float, float]
    """Rotation of the bounding box in RPY format"""
    quaternion: List[float, float, float, float]
    """Rotation of the bounding box in wxyz format"""
    position: List[float, float, float]
    """Position of the bounding box"""

    def get_nerfstudio_data(self):

        dimensions = np.array(self.dimensions) * 10
        euler = np.round(np.rad2deg(self.euler)[::-1] * np.array([1, -1, 1]))
        R = Rotation.from_euler('xyz', euler, degrees=True)
        wxyz = R.as_quat(False)
        position = np.array(self.position) * 10

        return BBoxData(self.idx, dimensions, euler, wxyz, position)
    
    def get_json_data(self):

        return {"idx": self.idx, "data": {"dimensions": self.dimensions, "euler": self.euler, "quaternion": self.quaternion, "position": self.position}}


class BoundingBoxToolbox(ViewerElement[bool]):
    """A toolbox in the viewer

    Args:
        name: The name of the button
        move_hook: The function to call when the button is pressed
        disabled: If the button is disabled
        visible: If the button is visible
    """

    position_handle: GuiInputHandle[Tuple[float, float, float]]
    rotation_handle: GuiInputHandle[Tuple[float, float, float]]
    dimensions_handle: GuiInputHandle[Tuple[float, float, float]]

    def __init__(self, 
                 name: str, 
                 bbox_data: BBoxData,
                 move_hook: Callable[[BoundingBoxToolbox], Any], 
                 resize_hook: Callable[[BoundingBoxToolbox], Any], 
                 bbox_selector_hook: Callable[[BoundingBoxToolbox], Any], 
                 add_box_hook: Callable[[BoundingBoxToolbox], Any], 
                 disabled: bool = False, 
                 visible: bool = True, 
                 ):
        super().__init__(name, disabled=disabled, visible=visible, cb_hook=move_hook)

        self.viser_server = None

        self.resize_hook = resize_hook
        self.move_hook = move_hook
        self.bbox_selector_hook = bbox_selector_hook
        self.add_box_hook = add_box_hook

        self.bbox_data = bbox_data

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        self.position_handle = viser_server.add_gui_vector3(
            "Position", tuple(self.bbox_data.position), step=0.1, disabled=self.disabled, visible=self.visible, hint="Provide position in meters"
        )
        self.rotation_handle = viser_server.add_gui_vector3(
            "Rotation", tuple(self.bbox_data.euler), step=1.0, disabled=self.disabled, visible=self.visible, hint="Provide rotation in degrees"
        )
        self.dimensions_handle = viser_server.add_gui_vector3(
            "Size", tuple(self.bbox_data.dimensions), step=0.1, disabled=self.disabled, visible=self.visible, hint="Provide size in meters"
        )
        self.bbox_selector_handle = viser_server.add_gui_number("Selected box", 0, 0, step=1)
        self.add_box_handle = viser_server.add_gui_button("Add Bbox", color='gray', icon=Icon.BOX)

    def install(self, viser_server: ViserServer) -> None:
        self._create_gui_handle(viser_server)

        self.viser_server = viser_server

        assert self.position_handle is not None and self.rotation_handle is not None and self.dimensions_handle is not None
        self.position_handle.on_update(lambda _: self.move_hook(self))
        self.rotation_handle.on_update(lambda _: self.move_hook(self))
        self.dimensions_handle.on_update(lambda _: self.resize_hook(self))
        self.bbox_selector_handle.on_update(lambda _: self.bbox_selector_hook(self))
        self.add_box_handle.on_click(lambda _: self.add_box_hook(self))

    def get_bbox_data(self) -> BBoxData:

        dimensions = tuple(np.array(self.dimensions_handle.value) / 10)
        euler = tuple(np.deg2rad(self.rotation_handle.value[::-1] * np.array([1, -1, 1])))
        R = Rotation.from_euler('zyx', euler)
        quaternion = tuple(R.as_quat(True))
        position = tuple(np.array(self.position_handle.value) / 10)

        return BBoxData(self.bbox_selector_handle.value, dimensions, euler, quaternion, position)
    
    def update(self, data: BBoxData) -> None:

        self.bbox_data = data

        self.position_handle.value = data.position
        self.rotation_handle.value = data.euler
        time.sleep(0.1) # Not ideal solution
        # Above lines of the code trigger callback functions which run in separate threads
        # This causes issue with the line below because it should wait until the code above finished execution
        self.dimensions_handle.value = data.dimensions


class BoundingBox(ViewerElement[bool]):
    """A box in the viewer

    Args:
        name: The name of the button
        cb_hook: The function to call when the button is pressed
        disabled: If the button is disabled
        visible: If the button is visible
    """

    box_handle: MeshHandle

    def __init__(self, data: BBoxData, cb_hook: Callable[[BoundingBox], Any], disabled: bool = False, visible: bool = True, wireframe: bool = False):

        name = f"Box_{data.idx}"
        super().__init__(name, disabled=disabled, visible=visible, cb_hook=cb_hook)

        self.data = data
        self.wireframe = wireframe

    def _create_gui_handle(self, viser_server: ViserServer) -> None:

        mesh = trimesh.creation.box(tuple(self.data.dimensions))

        flat_shading = False if self.wireframe else True

        self.box_handle = viser_server.add_mesh_simple(
                          name=self.name,
                          vertices=mesh.vertices,
                          faces=mesh.faces,
                          color=(255, 0, 0),
                          wireframe=self.wireframe,
                          flat_shading=flat_shading,
                          position=tuple(self.data.position),
                          wxyz=tuple(self.data.quaternion),
                          visible=True,
                          )

    def install(self, viser_server: ViserServer) -> None:
        self._create_gui_handle(viser_server)

        assert self.box_handle is not None
        self.box_handle.on_click(lambda _: self.cb_hook(self))

    def update(self, data: BBoxData) -> None:

        self.data = data

        self.box_handle.position = data.position
        R = Rotation.from_euler('xyz', data.euler, degrees=True)
        quaternion = R.as_quat(False)
        self.box_handle.wxyz = quaternion