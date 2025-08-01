def _pose_from_Rp(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T

def _rand_small_rotation(max_angle=0.25):
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(-max_angle, max_angle)
    from scipy.spatial.transform import Rotation as R
    return R.from_rotvec(angle * axis).as_matrix()
