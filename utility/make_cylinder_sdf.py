from context import *

def make_cylinder_sdf(height, radius, fd_h=1e-6):
    """
    Create a cylinder SDF centered at the origin along the z-axis.

    Args:
        height (float): Height of the cylinder.
        radius (float): Radius of the cylinder.
        fd_h (float, optional): Finite difference step size for gradient. Defaults to -1.0.

    Returns:
        SDFObject: The signed distance function object representing the cylinder.
    """
    return contouring._contouring_cpp_module._make_cylinder_sdf(height, radius, fd_h)