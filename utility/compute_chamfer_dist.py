from context import *
from .mesh_sampling import sample_points_on_mesh

def compute_chamfer_dist(V1, F1, V2, F2, n_samples=10000, rng=None):
    """
    Compute symmetric Chamfer distance between two triangle meshes.

    Parameters
    ----------
    V1, F1 : (n,3), (m,3)
        Vertices and faces of mesh 1
    V2, F2 : (n,3), (m,3)
        Vertices and faces of mesh 2
    n_samples : int
        Number of surface samples per mesh
    rng : np.random.Generator or None
        Optional RNG for reproducibility

    Returns
    -------
    float
        Chamfer distance
    """
    if F1.shape[1] != 3 or F2.shape[1] != 3:
        raise ValueError("Input meshes must be triangular.")

    if (V1.size == 0 or F1.size == 0 or V2.size == 0 or F2.size == 0):
        return 1e12

    # Sample points (delegates to igl when rng is None)
    B1, S1, P1 = sample_points_on_mesh(V1, F1, n_samples=n_samples, rng=rng)

    sqrD1, I1, C1 = igl.point_mesh_squared_distance(
        P1, V2, F2
    )
    mean_sqrD1 = np.mean(sqrD1)

    B2, S2, P2 = sample_points_on_mesh(V2, F2, n_samples=n_samples, rng=rng)

    # --- Distances from P2 to M1 ---
    sqrD2, I2, C2 = igl.point_mesh_squared_distance(
        P2, V1, F1
    )
    mean_sqrD2 = np.mean(sqrD2)

    error_M1_to_M2 = np.sqrt(mean_sqrD1)
    error_M2_to_M1 = np.sqrt(mean_sqrD2)

    return error_M1_to_M2 + error_M2_to_M1