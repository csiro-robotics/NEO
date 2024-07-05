"""
Utils for displaying bbox in the viewer
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Literal, Tuple, Type, Optional, Callable, Any

from nerfstudio.viewer.viewer_elements import ViewerElement
from viser import (
    MeshHandle,
    GuiInputHandle,
    ViserServer,
)


class BoundingBoxToolbox(ViewerElement[bool]):
    """A toolbox in the viewer

    Args:
        name: The name of the button
        cb_hook: The function to call when the button is pressed
        disabled: If the button is disabled
        visible: If the button is visible
    """

    position_handle: GuiInputHandle[Tuple[float, float, float]]
    rotation_handle: GuiInputHandle[Tuple[float, float, float]]
    dimensions_handle: GuiInputHandle[Tuple[float, float, float]]

    def __init__(self, name: str, cb_hook: Callable[[BoundingBoxToolbox], Any], resize_hook: Callable[[BoundingBoxToolbox], Any], disabled: bool = False, visible: bool = True, dimensions: Tuple[float, float, float] = (0, 0, 0), rotation: Tuple[float, float, float] = (0, 0, 0), position: Tuple[float, float, float] = (0, 0, 0)):
        super().__init__(name, disabled=disabled, visible=visible, cb_hook=cb_hook)

        self.resize_hook = resize_hook
        self.dimensions = dimensions
        self.position = position
        self.rotation = rotation

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        self.position_handle = viser_server.add_gui_vector3(
            "Position", self.position, step=0.1, disabled=self.disabled, visible=self.visible, hint="Provide position in meters"
        )
        self.rotation_handle = viser_server.add_gui_vector3(
            "Rotation", self.rotation, step=1.0, disabled=self.disabled, visible=self.visible, hint="Provide rotation in degrees"
        )
        self.dimensions_handle = viser_server.add_gui_vector3(
            "Size", self.dimensions, step=0.1, disabled=self.disabled, visible=self.visible, hint="Provide size in meters"
        )

    def install(self, viser_server: ViserServer) -> None:
        self._create_gui_handle(viser_server)

        assert self.position_handle is not None and self.rotation_handle is not None and self.dimensions_handle is not None
        self.position_handle.on_update(lambda _: self.cb_hook(self))
        self.rotation_handle.on_update(lambda _: self.cb_hook(self))
        self.dimensions_handle.on_update(lambda _: self.resize_hook(self))


class BoundingBox(ViewerElement[bool]):
    """A box in the viewer

    Args:
        name: The name of the button
        cb_hook: The function to call when the button is pressed
        disabled: If the button is disabled
        visible: If the button is visible
    """

    box_handle: MeshHandle

    def __init__(self, name: str, dimensions: Tuple[float, float, float], wxyz: Tuple[float, float, float, float], position: Tuple[float, float, float],  cb_hook: Callable[[BoundingBox], Any], disabled: bool = False, visible: bool = True, wireframe: bool = False):
        super().__init__(name, disabled=disabled, visible=visible, cb_hook=cb_hook)

        self.dimensions = dimensions
        self.wxyz = wxyz
        self.position = position
        self.viser_server = None
        self.wireframe = wireframe

    def _create_gui_handle(self, viser_server: ViserServer) -> None:

        import trimesh.creation

        mesh = trimesh.creation.box(self.dimensions)

        flat_shading = False if self.wireframe else True

        self.box_handle = viser_server.add_mesh_simple(
                          name="Box",
                          vertices=mesh.vertices,
                          faces=mesh.faces,
                          color=(255, 0, 0),
                          wireframe=self.wireframe,
                          flat_shading=flat_shading,
                          position=self.position,
                          wxyz=self.wxyz,
                          visible=True,
                          )

    def install(self, viser_server: ViserServer) -> None:
        self._create_gui_handle(viser_server)

        self.viser_server = viser_server

        assert self.box_handle is not None
        self.box_handle.on_click(lambda _: self.cb_hook(self))