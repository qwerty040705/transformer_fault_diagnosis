import numpy as np

def getRaisedCosineSigmoidPosition(s_end, t_end, t_query):
    """
    Compute the position for a given time interval and total length using a raised-cosine sigmoid.
    The function ensures a smooth position profile starting at 0 and ending at s_end over t_end seconds.

    Parameters:
        s_end (ndarray): target position (n x 1)
        t_end (float): total duration
        t_query (ndarray): time query points (1 x T)

    Returns:
        ndarray: position at t_query (n x T)
    """
    s_end = np.atleast_2d(s_end)
    if s_end.shape[0] < s_end.shape[1]:
        s_end = s_end.T  # ensure (n, 1)

    t_query = np.asarray(t_query)
    profile = 0.5 * (1 - np.cos(np.pi * t_query / t_end))  # shape: (T,)
    x_query = s_end @ profile.reshape(1, -1)  # shape: (n, T)

    return x_query
