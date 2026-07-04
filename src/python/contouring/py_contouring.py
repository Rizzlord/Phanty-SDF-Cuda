from .build_grid import build_grid
from . import _contouring_cpp_module as _cont_cpp

def _build_options(options):
    if options is None:
        return _cont_cpp.ContouringOptions()
    if isinstance(options, dict):
        opts = _cont_cpp.ContouringOptions()
        # Parse the passed dict and set the corresponding fields
        for k, v in options.items():
            if hasattr(opts, k):
                setattr(opts, k, v)
            else:
                raise ValueError(f"Unknown ContouringOptions field: {k}")
        return opts
    return options


def py_contouring(
    S, GV,
    resX, resY, resZ,
    isoValue,
    options,
    true_sdf,
    true_sdf_grad
):
    """Run contouring and return vertices and faces.

    Args:
        S: 1D numpy array of scalar values sampled on a grid.

    Raises:
        ValueError: If S is provided but not a 1D array.
        ValueError: If GV is not a valid grid representation.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple (V, F) where V is an (M,3) numpy array of vertex positions and F is
            an (K,3) numpy array of triangle indices.
    """

    opts = _build_options(options)
    GV_out = build_grid(GV)

    if resX is None or resY is None or resZ is None:
        raise ValueError("resX, resY and resZ must be provided when GV is an array")

    if true_sdf is not None or true_sdf_grad is not None:
        if true_sdf is None or true_sdf_grad is None:
            raise ValueError("Both true_sdf and true_sdf_grad must be provided together")
        V, F = _cont_cpp._contouring_cpp_with_sdf(S, GV_out, int(resX), int(resY), int(resZ), float(isoValue), opts, true_sdf, true_sdf_grad)
        return V, F

    # If no true_sdf or true_sdf_grad provided, use the standard contouring
    V, F = _cont_cpp._contouring_cpp(S, GV_out, int(resX), int(resY), int(resZ), float(isoValue), opts)
    return V, F
