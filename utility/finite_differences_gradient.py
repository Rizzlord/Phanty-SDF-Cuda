from context import *

def finite_difference_gradient(f, h):
    """Computes the finite difference gradient and normalizes it. Mirrors the C++ implementation.

    Args:
        f (callable): The signed distance function.
        h (float): The finite difference step size.

    Returns:
        callable: A function that computes the gradient at a given point.
    """
    def grad_func(x):
        # Flatten x to ensure it's a 1D array
        x_vec = np.array(x, dtype=np.float64).flatten()
        
        g = np.zeros(3)
        eps = h if h > 0.0 else 1e-6    # To match the C++ implementation

        for c in range(3):
            e = np.zeros(3)
            e[c] = eps
            
            fp = f(x_vec + e)
            fn = f(x_vec - e)
            
            g[c] = (fp - fn) / (2.0 * eps)

        nrm = np.linalg.norm(g)
        if nrm > 0.0: g /= nrm
        return g

    return grad_func