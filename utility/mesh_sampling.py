from context import *

def sample_points_on_mesh(V, F, n_samples=1000, rng=None):
    if V.size == 0 or F.size == 0:
        return np.zeros((0, 3)), np.zeros((0,), dtype=int), np.zeros((0, V.shape[1]))

    if rng is None:
        # igl.random_points_on_mesh returns (B, S, P)
        return igl.random_points_on_mesh(n_samples, V, F)

    # Compute triangle vertices
    v0 = V[F[:, 0], :]
    v1 = V[F[:, 1], :]
    v2 = V[F[:, 2], :]

    # Triangle areas
    tri_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    total_area = np.sum(tri_areas)
    if total_area == 0:
        # Degenerate mesh
        B = np.zeros((n_samples, 3), dtype=float)
        S = np.zeros((n_samples,), dtype=int)
        P = np.zeros((n_samples, V.shape[1]), dtype=float)
        return B, S, P

    probs = tri_areas / total_area
    face_indices = rng.choice(F.shape[0], size=n_samples, p=probs)

    # barycentric sampling
    r1 = np.sqrt(rng.random(n_samples))
    r2 = rng.random(n_samples)
    a = 1.0 - r1
    b = r1 * (1.0 - r2)
    c = r1 * r2

    f0 = V[F[face_indices, 0], :]
    f1 = V[F[face_indices, 1], :]
    f2 = V[F[face_indices, 2], :]

    P = (a[:, None] * f0) + (b[:, None] * f1) + (c[:, None] * f2)
    B = np.stack([a, b, c], axis=1)
    S = face_indices.astype(int)
    return B, S, P
