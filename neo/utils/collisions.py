import numpy as np
import itertools

from typing import Dict, List
from collections import defaultdict
from nerfstudio.utils.rich_utils import CONSOLE


class OBB:
    def __init__(self, center, axes, half_sizes):
        self.center = np.array(center)
        self.axes = np.array(axes)  # should be a 3x3 matrix
        self.half_sizes = np.array(half_sizes)  # length of half-sizes along each axis

def get_vertices(obb):
    """Calculate the 8 vertices of the OBB."""
    vertices = []
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            for dz in [-1, 1]:
                vertex = obb.center + (dx * obb.half_sizes[0] * obb.axes[0] +
                                       dy * obb.half_sizes[1] * obb.axes[1] +
                                       dz * obb.half_sizes[2] * obb.axes[2])
                vertices.append(vertex)
    return np.array(vertices)

def overlap_on_axis(obb1, obb2, axis):
    """Check if two OBBs overlap on a given axis."""
    # Project OBB1
    proj1 = np.dot(get_vertices(obb1), axis)
    proj2 = np.dot(get_vertices(obb2), axis)
    
    min1, max1 = np.min(proj1), np.max(proj1)
    min2, max2 = np.min(proj2), np.max(proj2)
    
    return not (max1 < min2 or max2 < min1)

def check_obb_collision(obb1, obb2):
    """Check if two OBBs collide using the Separating Axis Theorem."""
    # Get the axes to check for separation
    axes = np.vstack([obb1.axes.T, obb2.axes.T])  # Combine the axes of both OBBs
    
    # Check axes from both OBBs
    for axis in axes:
        if not overlap_on_axis(obb1, obb2, axis):
            return False
    
    # Check cross products of the axes
    for i in range(3):
        for j in range(3):
            cross_axis = np.cross(obb1.axes[i], obb2.axes[j])
            if not np.all(cross_axis == 0):  # Avoid zero-length axes
                if not overlap_on_axis(obb1, obb2, cross_axis):
                    return False

    return True


def get_collision_groups(bboxes: Dict[str, np.ndarray]) -> List[List[int]]:

    # Add boxes to visualization
    obbs = []
    CONSOLE.print("Reading bbox data")
    for bbox in bboxes:
        bbox_data = bbox['data']
        offset = 0.0
        bbox_w, bbox_h, bbox_d  = np.array(bbox_data['dimensions']) - offset

        position = np.array(bbox_data['position'])

        alpha, beta, gamma = bbox_data['euler']

        rot_roll = np.array([[1, 0, 0],
                            [0, np.cos(alpha), -np.sin(alpha)],
                            [0, np.sin(alpha), np.cos(alpha)]])
        
        rot_pitch = np.array([[np.cos(beta), 0, np.sin(beta)],
                            [0, 1, 0],
                            [-np.sin(beta), 0, np.cos(beta)]])
        
        rot_yaw = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                            [np.sin(gamma), np.cos(gamma), 0],
                            [0, 0, 1]])
        
        R = rot_roll @ rot_pitch @ rot_yaw
        
        obb = OBB(center=position, axes=R, half_sizes=np.array([-bbox_w / 2, -bbox_h / 2, -bbox_d / 2]))
        obbs.append(obb)

    # Detect collisions between bounding boxes
    combinations = list(itertools.combinations(np.arange(len(obbs)), 2))

    collision_graph = defaultdict(set)
    for comb in combinations:
        collision = check_obb_collision(obbs[comb[0]], obbs[comb[1]])
        if collision:
            collision_graph[comb[0]].add(comb[1])
            collision_graph[comb[1]].add(comb[0])

    def find_connected_components(graph):
        visited = set()
        components = []

        def dfs(node, component):
            stack = [node]
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    component.append(current)
                    stack.extend(graph[current] - visited)

        for node in graph:
            if node not in visited:
                component = []
                dfs(node, component)
                components.append(component)

        return components

    # Get the groups of colliding bounding boxes
    collision_groups = find_connected_components(collision_graph)

    for bbox_idx in range(len(obbs)):
        skip = False
        for group in collision_groups:
            if bbox_idx in group:
                skip = True

        if not skip:
            collision_groups.append([bbox_idx])
    
    CONSOLE.print(f"Found {len(collision_groups)} groups of bounding boxes")

    return collision_groups