import numpy as np

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