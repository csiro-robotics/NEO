"""
Utils for displaying bbox in the viewer
"""
from __future__ import annotations

import time
import numpy as np
import trimesh.creation

from scipy.spatial.transform import Rotation
from dataclasses import dataclass
from typing import Tuple, List, Callable, Any, Optional

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
    is_reconfigured: Optional[bool] = False
    """If content of the bbox is moved to another place"""
    offset_position: Optional[List[float, float, float]] = (0.0, 0.0, 0.0)
    """Positon offset for reconfigured bounding box"""
    offset_euler: Optional[List[float, float, float]] = (0.0, 0.0, 0.0)
    """Angle offset for reconfigured bounding box"""

    def viser_format(self) -> BBoxData:

        dimensions = np.array(self.dimensions) * 10
        euler = np.round(np.rad2deg(self.euler)[::-1] * np.array([1, -1, 1]))
        R = Rotation.from_euler('xyz', euler, degrees=True)
        quaternion = R.as_quat(False)
        position = np.array(self.position) * 10

        offset_position =  np.array(self.offset_position) * 10
        offset_euler = np.round(np.rad2deg(self.offset_euler)[::-1] * np.array([1, -1, 1]))

        return BBoxData(self.idx, dimensions, euler, quaternion, position, self.is_reconfigured, offset_position, offset_euler)
    
    def nerfstudio_format(self) -> BBoxData:

        dimensions = np.array(self.dimensions) / 10
        euler = np.deg2rad(self.euler[::-1] * np.array([1, -1, 1]))
        R = Rotation.from_euler('xyz', euler)
        quaternion = R.as_quat(True)
        position = np.array(self.position) / 10

        offset_position = np.array(self.offset_position) / 10
        offset_euler = np.deg2rad(self.offset_euler[::-1] * np.array([1, -1, 1]))

        return BBoxData(self.idx, dimensions, euler, quaternion, position, self.is_reconfigured, offset_position, offset_euler)
    
    def get_json_data(self):

        dimensions = (np.array(self.dimensions) / 10).tolist()
        euler = np.deg2rad(self.euler[::-1] * np.array([1, -1, 1]))
        R = Rotation.from_euler('xyz', euler)
        quaternion = tuple(R.as_quat(True))
        position = (np.array(self.position) / 10).tolist()

        offset_position = (np.array(self.offset_position) / 10).tolist()
        offset_euler = (np.deg2rad(self.offset_euler[::-1] * np.array([1, -1, 1]))).tolist()

        return {"idx": self.idx, 
                "data": {"dimensions": dimensions, 
                         "euler": list(euler), 
                         "quaternion": list(quaternion), 
                         "position": position,
                         "is_reconfigured": self.is_reconfigured,
                         "offset_position": offset_position,
                         "offset_euler": offset_euler
                         }
                }


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
                 offset_box_hook: Callable[[BoundingBoxToolbox], Any], 
                 resize_hook: Callable[[BoundingBoxToolbox], Any], 
                 reconfigure_hook: Callable[[BoundingBoxToolbox], Any], 
                 bbox_selector_hook: Callable[[BoundingBoxToolbox], Any], 
                 add_box_hook: Callable[[BoundingBoxToolbox], Any], 
                 remove_box_hook: Callable[[BoundingBoxToolbox], Any], 
                 apply_changes_hook: Callable[[BoundingBoxToolbox], Any], 
                 disabled: bool = False, 
                 visible: bool = True, 
                 ):
        super().__init__(name, disabled=disabled, visible=visible, cb_hook=move_hook)

        self.viser_server = None

        self.resize_hook = resize_hook
        self.move_hook = move_hook
        self.offset_box_hook = offset_box_hook
        self.reconfigure_hook = reconfigure_hook
        self.bbox_selector_hook = bbox_selector_hook
        self.add_box_hook = add_box_hook
        self.remove_box_hook = remove_box_hook
        self.apply_changes_hook = apply_changes_hook

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

        self.reconfigure_handle = viser_server.add_gui_checkbox("Reconfigure", self.bbox_data.is_reconfigured)

        self.offset_position_handle = viser_server.add_gui_vector3(
            "Offset Position", tuple(self.bbox_data.offset_position), step=0.1, disabled=self.disabled, visible=self.bbox_data.is_reconfigured, hint="Provide position in meters"
        )
        self.offset_rotation_handle = viser_server.add_gui_vector3(
            "Offset Rotation", tuple(self.bbox_data.offset_euler), step=1.0, disabled=self.disabled, visible=self.bbox_data.is_reconfigured, hint="Provide rotation in degrees"
        )

        self.bbox_selector_handle = viser_server.add_gui_number("Selected box", self.bbox_data.idx, 0, step=1)
        self.add_box_handle = viser_server.add_gui_button("Add Bbox", color='gray', icon=Icon.BOX)
        self.remove_box_handle = viser_server.add_gui_button("Remove Bbox", color='gray', icon=Icon.TRASH)
        self.apply_changes_handle = viser_server.add_gui_button("Apply changes", color='gray', icon=Icon.DEVICE_FLOPPY)

    def install(self, viser_server: ViserServer) -> None:
        self._create_gui_handle(viser_server)

        self.viser_server = viser_server

        assert self.position_handle is not None and self.rotation_handle is not None and self.dimensions_handle is not None and self.reconfigure_handle is not None and self.apply_changes_handle is not None
        self.position_handle.on_update(lambda _: self.move_hook())
        self.rotation_handle.on_update(lambda _: self.move_hook())
        self.dimensions_handle.on_update(lambda _: self.resize_hook())
        self.reconfigure_handle.on_update(lambda _:self.reconfigure_hook())
        self.offset_position_handle.on_update(lambda _: self.offset_box_hook())
        self.offset_rotation_handle.on_update(lambda _: self.offset_box_hook())
        self.bbox_selector_handle.on_update(lambda _: self.bbox_selector_hook())
        self.add_box_handle.on_click(lambda _: self.add_box_hook())
        self.remove_box_handle.on_click(lambda _: self.remove_box_hook())
        self.apply_changes_handle.on_click(lambda _: self.apply_changes_hook())

    def get_bbox_data(self) -> BBoxData:

        dimensions = tuple(np.array(self.dimensions_handle.value) / 10)
        euler = tuple(np.deg2rad(self.rotation_handle.value[::-1] * np.array([1, -1, 1])))
        offset_euler = tuple(np.deg2rad(self.offset_rotation_handle.value[::-1] * np.array([1, -1, 1])))
        R = Rotation.from_euler('xyz', euler)
        quaternion = tuple(R.as_quat(True))
        position = tuple(np.array(self.position_handle.value) / 10)
        offset_position = tuple(np.array(self.offset_position_handle.value) / 10)

        return BBoxData(self.bbox_selector_handle.value, dimensions, euler, quaternion, position, is_reconfigured=self.reconfigure_handle.value, offset_position=offset_position, offset_euler=offset_euler)
    
    def update(self, data: BBoxData) -> None:

        self.bbox_data = data

        # WARNING! Order oe execution matters!
        self.reconfigure_handle.value = data.is_reconfigured
        time.sleep(0.1) # Not ideal solution
        self.position_handle.value = data.position
        self.rotation_handle.value = data.euler
        time.sleep(0.1) # Not ideal solution
        # Above lines of the code trigger callback functions which run in separate threads
        # This causes issue with the line below because it should wait until the code above finishes execution
        self.dimensions_handle.value = data.dimensions
        self.offset_position_handle.value = data.offset_position
        self.offset_rotation_handle.value = data.offset_euler


class ViserBoundingBox(ViewerElement[bool]):
    """A box in the viewer

    Args:
        name: The name of the button
        cb_hook: The function to call when the button is pressed
        disabled: If the button is disabled
        visible: If the button is visible
    """

    box_handle: MeshHandle

    def __init__(self, data: BBoxData, name_prefix: str = "Box", cb_hook: Callable[[ViserBoundingBox], Any] = None, disabled: bool = False, visible: bool = True, wireframe: bool = False, color: Tuple[int, int, int] = (255, 0, 0), offset=False):
        
        name = f"{name_prefix}_{data.idx}_{time.time()}"
        super().__init__(name, disabled=disabled, visible=visible, cb_hook=cb_hook)

        self.data = data
        self.wireframe = wireframe
        self._color = color
        self.offset = offset

    def _create_gui_handle(self, viser_server: ViserServer) -> None:

        mesh = trimesh.creation.box(tuple(self.data.dimensions))

        flat_shading = False if self.wireframe else True

        if not self.offset:
            self.box_handle = viser_server.add_mesh_simple(
                            name=self.name,
                            vertices=mesh.vertices,
                            faces=mesh.faces,
                            color=self._color,
                            wireframe=self.wireframe,
                            flat_shading=flat_shading,
                            position=tuple(self.data.position),
                            wxyz=tuple(self.data.quaternion),
                            visible=True,
                            )
        else:
            R1 = Rotation.from_euler('xyz', self.data.euler, degrees=True)
            R2 = Rotation.from_euler('xyz', self.data.offset_euler, degrees=True)
            R = R2 * R1
            quaternion = R.as_quat(False)

            self.box_handle = viser_server.add_mesh_simple(
                            name=self.name,
                            vertices=mesh.vertices,
                            faces=mesh.faces,
                            color=self._color,
                            wireframe=self.wireframe,
                            flat_shading=flat_shading,
                            position=tuple(self.data.position + self.data.offset_position),
                            wxyz=tuple(quaternion),
                            visible=True,
                            )

    def install(self, viser_server: ViserServer) -> None:
        self._create_gui_handle(viser_server)

        if self.box_handle is not None:
            self.box_handle.on_click(lambda _: self.cb_hook(self))

    def update(self, data: BBoxData) -> None:

        self.data = data

        if not self.offset:
            self.box_handle.position = data.position
            R = Rotation.from_euler('xyz', data.euler, degrees=True)
            quaternion = R.as_quat(False)
            self.box_handle.wxyz = quaternion

        else:
            self.box_handle.position = data.position + data.offset_position
            R1 = Rotation.from_euler('xyz', data.euler, degrees=True)
            R2 = Rotation.from_euler('xyz', data.offset_euler, degrees=True)
            R = R2 * R1
            quaternion = R.as_quat(False)
            self.box_handle.wxyz = quaternion