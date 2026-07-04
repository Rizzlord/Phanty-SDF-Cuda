from context import *
from .mesh_sampling import sample_points_on_mesh

def evaluate_mesh_accuracy(V, F, f, n_samples=1000, rng=None):
    """
    Evaluate mesh accuracy against an SDF by maximum absolute SDF value
    over mesh vertices.

    Parameters
    ----------
    V : (n,3)
        Vertices of the mesh
    F : (m,3)
        Faces of the mesh
    f : function
        SDF function that takes (3,) points and returns float distances
    n_samples : int
        Number of surface samples
    rng : np.random.Generator or None
        Optional RNG for reproducibility   

    Returns
    -------
    float
        Maximum absolute SDF value over sampled points
    """

    if V.size == 0 or F.size == 0:
        return 1e12

    # Sample random points on the mesh surface (delegates to igl when rng is None)
    B, S, P = sample_points_on_mesh(V, F, n_samples=n_samples, rng=rng)

    max_abs_sdf = 0.0
    for k in range(P.shape[0]):
        max_abs_sdf = max(max_abs_sdf, abs(f(P[k])))

    return max_abs_sdf
