from context import *

def make_rotated_box_sdf_center(half_extents, center, R, fd_h=1e-6):
    """
    Create a rotated box SDF specifying the center.

    Args:
        half_extents (array-like): Half extents of the box (3,).
        center (array-like): Center of the box (3,).
        R (array-like): Rotation matrix (3,3).
        fd_h (float, optional): Finite difference step size for gradient. Defaults to -1.0.

    Returns:
        SDFObject: The signed distance function object representing the rotated box.
    """
    half_extents = np.asarray(half_extents)
    center = np.asarray(center)
    R = np.asarray(R)
    return contouring._contouring_cpp_module._make_rotated_box_sdf_center(half_extents, center, R, fd_h)