from context import *

def min_sdfs(sdf1, sdf2):
    """
    Create a SDF that is the minimum of two SDFs.

    Args:
        sdf1 (SDFObject): The first signed distance function object.
        sdf2 (SDFObject): The second signed distance function object.

    Returns:
        SDFObject: A new SDFObject representing the minimum of the two input SDFs.
    """
    
    min_sdf = contouring._contouring_cpp_module.SDFObject()
    min_sdf.f = lambda p: min(sdf1.f(p), sdf2.f(p))
    min_sdf.grad = lambda p: sdf1.grad(p) if sdf1.f(p) < sdf2.f(p) else sdf2.grad(p)
    return min_sdf
