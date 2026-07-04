import numpy as np

def triangulate_quads(V, F):
    F_tri = []
    for face in F:
        if len(face) == 3:
            F_tri.append(face)
        elif len(face) == 4:
            v0, v1, v2, v3 = face
            F_tri.append([v0, v1, v2])
            F_tri.append([v0, v2, v3])
        else:
            raise ValueError("Face with more than 4 vertices encountered.")
    return np.array(F_tri, dtype=F.dtype)