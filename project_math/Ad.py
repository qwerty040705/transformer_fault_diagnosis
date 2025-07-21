import numpy as np
from .skew import skew 

def Ad(T):
    """
    Compute the adjoint representation of an SE(3) transformation matrix.
    
    Parameters:
        T (numpy.ndarray): 4x4 SE(3) matrix
    
    Returns:
        numpy.ndarray: 6x6 adjoint matrix
    """
    R = T[:3, :3]
    p = T[:3, 3]
    Ad_T = np.block([
        [R, np.zeros((3, 3))],
        [skew(p) @ R, R]
    ])
    return Ad_T
