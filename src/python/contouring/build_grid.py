import numpy as np
from . import _contouring_cpp_module as _cont_cpp

def build_grid(GV, min_corner=-1, max_corner=1):
    """Return a grid of positions from either a (nx,ny,nz) tuple or an (N,3) array.

    Args:
        GV: Either a (nx,ny,nz) tuple or an (N,3) numpy array.

    Raises:
        ValueError: If GV is not a valid grid representation.
    """
    if isinstance(GV, tuple) and len(GV) == 3 and all(isinstance(x, int) for x in GV):
        nx, ny, nz = GV
        return _cont_cpp._grid(np.array([min_corner, min_corner, min_corner]),
                              np.array([max_corner, max_corner, max_corner]),
                              nx, ny, nz)
    if isinstance(GV, np.ndarray):
        if GV.ndim != 2 or GV.shape[1] != 3:
            raise ValueError("GV numpy array must have shape (N,3)")
        return GV
    raise ValueError("GV must be either a (nx,ny,nz) tuple or an (N,3) numpy array")
