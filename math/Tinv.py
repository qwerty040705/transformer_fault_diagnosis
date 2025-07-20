import numpy as np

def Tinv(T):
    """
    Compute the inverse of an SE(3) transformation matrix.

    Parameters:
        T (numpy.ndarray): 4x4 SE(3) transformation matrix

    Returns:
        numpy.ndarray: 4x4 inverse of T
    """
    R = T[:3, :3]
    p = T[:3, 3].reshape(3, 1)
    T_inv = np.block([
        [R.T, -R.T @ p],
        [np.zeros((1, 3)), np.array([[1]])]
    ])
    return T_inv
