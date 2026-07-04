from context import *
from finite_differences_gradient import finite_difference_gradient

def intersection_sdfs(sdf1, sdf2, res=48, min_corner=-1.0, max_corner=1.0):
    """
    Create a SDF that is the intersection of two SDFs.

    Args:
        sdf1 (SDFObject): The first signed distance function object.
        sdf2 (SDFObject): The second signed distance function object.
        res (int, optional): Resolution used to extract the intersection mesh. Defaults to 48.
    Returns:
        SDFObject: A new SDFObject representing the intersection of the two input SDFs.
        V (ndarray): Vertices of the extracted mesh.
        F (ndarray): Faces of the extracted mesh.
    """

    intersection_sdf = contouring._contouring_cpp_module.SDFObject()

    nx, ny, nz = res, res, res

    x = np.linspace(min_corner, max_corner, nx)
    y = np.linspace(min_corner, max_corner, ny)
    z = np.linspace(min_corner, max_corner, nz)
    # Use indexing 'xy' to match libigl convention
    X, Y, Z = np.meshgrid(x, y, z, indexing='xy')

    # Flatten the grid to (N, 3) shape
    GV = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    def max_func(p):
        return max(sdf1.f(p), sdf2.f(p))
    
    # Evaluate the max function on the grid
    S = np.array([max_func(p) for p in GV])

    # Marching cubes
    V, F, _ = igl.marching_cubes(S, GV, nx, ny, nz, 0.0)

    # We do not raise an error for empty meshes here
    
    def f(p):
        # Reshape p to (1, 3)
        P = np.asarray(p).reshape(1, 3)
        # Find closest point on the mesh
        # Some ligigl bindings return 3 values, others 4
        # The first returned value is always the signed distance
        res = igl.signed_distance(P, V, F)
        return res[0][0]

    intersection_sdf.f = f
    intersection_sdf.grad = finite_difference_gradient(intersection_sdf.f, h=1e-6)

    return intersection_sdf, V, F
