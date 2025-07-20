import numpy as np

def unskew(R):
    """
    Extract the 3-element vector from a 3x3 skew-symmetric matrix.

    Parameters:
        R (numpy.ndarray): 3x3 skew-symmetric matrix

    Returns:
        numpy.ndarray: 3x1 vector
    """
    Rv = np.array([
        R[2, 1],
        R[0, 2],
        R[1, 0]
    ])
    return Rv
