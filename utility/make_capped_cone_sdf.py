from context import *
from finite_differences_gradient import finite_difference_gradient

def make_capped_cone_sdf(height, radius_top, radius_bottom, fd_h=1e-6):
    """
    Create a capped cone SDF centered at the origin along the z-axis.

    Args:
        height (float): Height of the capped cone.
        radius_top (float): Radius of the top cap.
        radius_bottom (float): Radius of the bottom cap.
        fd_h (float, optional): Finite difference step size for gradient. Defaults to 1e-6.
    """
    # See https://iquilezles.org/articles/distfunctions/

    h = height / 2.0      # Half height, to center at origin
    r1 = radius_bottom
    r2 = radius_top

    def sdf(p):
        p = np.array(p)
        
        # Project to 2D: first coord is radial distance in xz-plane, second is height (y)
        q = np.array([np.linalg.norm(p[[0, 2]]), p[1]])
        
        # Pre-calculate geometric constants
        k1 = np.array([r2, h])
        k2 = np.array([r2 - r1, 2.0 * h])

        ca = np.array([q[0] - min(q[0], (q[1] < 0.0) and r1 or r2), abs(q[1]) - h])
        cb = q - k1 + k2 * np.clip(np.dot(k1 - q, k2) / np.dot(k2, k2), 0.0, 1.0)
        s = (cb[0] < 0.0 and ca[1] < 0.0) and -1.0 or 1.0
        
        return s * np.sqrt(min(np.dot(ca, ca), np.dot(cb, cb)))

    sdf_capped_cone = contouring._contouring_cpp_module.SDFObject()
    sdf_capped_cone.f = sdf
    sdf_capped_cone.grad = finite_difference_gradient(sdf_capped_cone.f, fd_h)
    return sdf_capped_cone