import numpy as np

def skew(v):
    """
    Compute the skew-symmetric matrix of a 3-element vector.
    
    Parameters:
        v (numpy.ndarray): 3-element vector
    
    Returns:
        numpy.ndarray: 3x3 skew-symmetric matrix
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
