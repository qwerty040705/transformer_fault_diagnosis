import numpy as np
from .skew import skew

def Adinv(T):
    """
    Compute the inverse adjoint representation of an SE(3) transformation matrix.
    
    Parameters:
        T (numpy.ndarray): 4x4 SE(3) matrix

    Returns:
        numpy.ndarray: 6x6 inverse adjoint matrix
    """
    R = T[:3, :3]
    p = T[:3, 3]
    Ad_Tinv = np.block([
        [R.T, np.zeros((3, 3))],
        [-R.T @ skew(p), R.T]
    ])
    return Ad_Tinv
