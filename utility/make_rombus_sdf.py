from context import *
from finite_differences_gradient import finite_difference_gradient

def make_rombus_sdf(la, lb, h, fd_h=1e-6):
    """
    Signed distance function for a 3D Rhombus.
    
    Args:
        p (np.array): Point coordinates [x, y, z]
        la (float): Width parameter
        lb (float): Depth/Slope parameter
        h (float): Height (vertical thickness)
        
    Returns:
        float: Signed distance to the surface
    """
    # Reference: https://iquilezles.org/articles/distfunctions/
    def sdf(p):
        p = np.abs(p)
        
        numerator = la * p[0] - lb * p[2] + lb**2
        denominator = la**2 + lb**2
        f = np.clip(numerator / denominator, 0.0, 1.0)
        
        p_xz = np.array([p[0], p[2]])
        offset = np.array([la, lb]) * np.array([f, 1.0 - f])
        w = p_xz - offset
        
        len_w = np.linalg.norm(w)

        qx = len_w * np.sign(w[0])
        qy = p[1] - h
        q = np.array([qx, qy])
        
        dist_inside = min(max(q[0], q[1]), 0.0)
        dist_outside = np.linalg.norm(np.maximum(q, 0.0))    
        return dist_inside + dist_outside
    
    sdf_rombus = contouring._contouring_cpp_module.SDFObject()
    sdf_rombus.f = sdf
    sdf_rombus.grad = finite_difference_gradient(sdf_rombus.f, fd_h)
    return sdf_rombus