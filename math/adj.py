import numpy as np
from .skew import skew 

def adj(V):
    """
    Compute the adjoint representation of a twist vector V ∈ ℝ⁶.

    Parameters:
        V (numpy.ndarray): 6x1 twist vector (or flat 6-element array)

    Returns:
        numpy.ndarray: 6x6 adjoint matrix
    """
    w = V[:3].reshape(3)
    v = V[3:6].reshape(3)
    
    Ad_V = np.block([
        [skew(w), np.zeros((3, 3))],
        [skew(v), skew(w)]
    ])
    return Ad_V
