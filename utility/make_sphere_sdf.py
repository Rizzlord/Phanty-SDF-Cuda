from context import *
from finite_differences_gradient import finite_difference_gradient

def make_sphere_sdf(c, r, fd_h=1e-6):
    """
    Create a sphere SDF centered at the given center.

    Args:
        c (array-like): Center of the sphere (3,).
        r (float): Radius of the sphere.
        fd_h (float, optional): Finite difference step size for gradient. Defaults to -1.0.

    Returns:
        SDFObject: The signed distance function object representing the sphere.
    """
    # No C++ binding for sphere SDF
    c = np.asarray(c)
    def sphere_sdf_func(x):
        return np.linalg.norm(x - c) - r
    
    sdf_sphere = contouring._contouring_cpp_module.SDFObject()
    sdf_sphere.f = sphere_sdf_func
    sdf_sphere.grad = finite_difference_gradient(sdf_sphere.f, fd_h)
    return sdf_sphere
