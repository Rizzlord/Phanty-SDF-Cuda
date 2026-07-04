from context import *
from .mesh_sampling import sample_points_on_mesh

def directed_hausdorff(V_src, F_src, V_tgt, F_tgt, n_samples=1000, rng=None):
    if V_src.size == 0 or F_src.size == 0:
        return 1e12

    # Sample points on source mesh (delegates to igl when rng is None)
    _, _, P = sample_points_on_mesh(V_src, F_src, n_samples=n_samples, rng=rng)

    sqrD, _, _ = igl.point_mesh_squared_distance(
        P, V_tgt, F_tgt
    )

    return np.sqrt(np.max(sqrD))


def compute_hausdorff_distance(V1, F1, V2, F2, n_samples=1000, rng=None):
    """
    Compute the symmetric Hausdorff distance between two meshes.

    Args:
        V1 (ndarray): Vertices of the first mesh.
        F1 (ndarray): Faces of the first mesh.
        V2 (ndarray): Vertices of the second mesh.
        F2 (ndarray): Faces of the second mesh.
        n_samples (int, optional): Number of random samples to use for approximation. Defaults to 100000.
        rng (np.random.Generator, optional): Random number generator for reproducibility. Defaults to None.

    Returns:
        float: The symmetric Hausdorff distance between the two meshes.
    """
    if F1.shape[1] != 3 or F2.shape[1] != 3:
        raise ValueError("Input meshes must be triangular.")

    d1 = directed_hausdorff(V1, F1, V2, F2, n_samples, rng)
    d2 = directed_hausdorff(V2, F2, V1, F1, n_samples, rng)
    return max(d1, d2)