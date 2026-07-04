from context import *

def build_rotation_matrix(angle, axis="all"):
    """
    Build a 3D rotation matrix around the specified axis.

    Args:
        angle (float): Rotation angle in radians.
        axis (str): Axis to rotate around ('x', 'y', 'z', or 'all' for combined rotation).
    
    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    c = np.cos(angle)
    s = np.sin(angle)

    if axis == "x":
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
    elif axis == "y":
        R = np.array([[c, 0, s],
                      [0, 1, 0],
                      [-s, 0, c]])
    elif axis == "z":
        R = np.array([[c, -s, 0],
                      [s, c, 0],
                      [0, 0, 1]])
    elif axis == "all":
        Rx = np.array([[1, 0, 0],
                       [0, c, -s],
                       [0, s, c]])
        Ry = np.array([[c, 0, s],
                       [0, 1, 0],
                       [-s, 0, c]])
        Rz = np.array([[c, -s, 0],
                       [s, c, 0],
                       [0, 0, 1]])
        R = Rz @ Ry @ Rx
    else:
        raise ValueError("Axis must be 'x', 'y', 'z', or 'all'.")

    return R