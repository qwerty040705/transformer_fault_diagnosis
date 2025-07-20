import numpy as np
import os
from scipy.spatial.transform import Rotation as R, Slerp

def create_se3_matrix(position, rotation=np.eye(3)):
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = position
    return T

def normalize_position_dynamic(pos, min_val, max_val):
    return 2 * ((pos - min_val) / (max_val - min_val + 1e-8)) - 1

def generate_random_trajectory(num_points=10, steps=100, rot_scale=np.pi, save_minmax=True):
    """
    Generates a smooth, random SE(3) trajectory (rotation + position),
    normalized position ∈ [-1, 1], smooth SLERP rotation.
    """
    # Step 1: Random waypoints in position and rotation vector
    pos_waypoints = np.random.uniform(low=-0.3, high=0.3, size=(num_points, 3))  # position
    rotvecs = np.random.uniform(low=-rot_scale, high=rot_scale, size=(num_points, 3))  # random SO(3)
    rot_waypoints = R.from_rotvec(rotvecs)

    # Step 2: Normalize position
    min_pos = pos_waypoints.min(axis=0)
    max_pos = pos_waypoints.max(axis=0)

    if save_minmax:
        os.makedirs("data", exist_ok=True)
        np.save("data/pos_min.npy", min_pos)
        np.save("data/pos_max.npy", max_pos)

    # Step 3: SE(3) trajectory construction with linear + SLERP interpolation
    total_segments = num_points - 1
    steps_per_segment = steps // total_segments
    trajectory = []

    for i in range(total_segments):
        p_start = pos_waypoints[i]
        p_end = pos_waypoints[i + 1]
        r_start = rot_waypoints[i]
        r_end = rot_waypoints[i + 1]

        # SLERP setup
        key_times = [0, 1]
        key_rots = R.from_rotvec([rotvecs[i], rotvecs[i + 1]])
        slerp = Slerp(key_times, key_rots)

        for alpha in np.linspace(0, 1, steps_per_segment, endpoint=False):
            pos = (1 - alpha) * p_start + alpha * p_end
            pos_norm = normalize_position_dynamic(pos, min_pos, max_pos)
            rot_mat = slerp([alpha])[0].as_matrix()
            T = create_se3_matrix(pos_norm, rot_mat)
            trajectory.append(T)

    # 마지막 포인트 수동 추가
    final_pos = normalize_position_dynamic(pos_waypoints[-1], min_pos, max_pos)
    final_rot = rot_waypoints[-1].as_matrix()
    trajectory.append(create_se3_matrix(final_pos, final_rot))

    return trajectory

# 예시 실행
if __name__ == "__main__":
    traj = generate_random_trajectory()
    print(f"Generated {len(traj)} SE(3) poses.")
    print("First pose:\n", traj[0])
