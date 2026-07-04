from context import *

def max_sdfs(sdf1, sdf2):
    """
    Create a SDF that is the maximum of two SDFs.

    Args:
        sdf1 (SDFObject): The first signed distance function object.
        sdf2 (SDFObject): The second signed distance function object.

    Returns:
        SDFObject: A new SDFObject representing the maximum of the two input SDFs.
    """

    max_sdf = contouring._contouring_cpp_module.SDFObject()
    max_sdf.f = lambda p: max(sdf1.f(p), sdf2.f(p))
    max_sdf.grad = lambda p: sdf1.grad(p) if sdf1.f(p) > sdf2.f(p) else sdf2.grad(p)
    return max_sdf
