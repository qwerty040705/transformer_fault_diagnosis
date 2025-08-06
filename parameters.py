import numpy as np
from dataclasses import dataclass

# ---------- Utility Functions ----------
def skew(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def Ad(T):
    R = T[:3, :3]
    p = T[:3, 3]
    return np.block([
        [R, np.zeros((3, 3))],
        [skew(p) @ R, R]
    ])

def get_inertia_tensor(mass, joint_to_com, inertia_matrix):
    T_ba = np.eye(4)
    T_ba[:3, 3] = -joint_to_com
    Ad_T = Ad(T_ba)
    upper = np.hstack((inertia_matrix, np.zeros((3, 3))))
    lower = np.hstack((np.zeros((3, 3)), mass * np.eye(3)))
    M = np.vstack((upper, lower))
    return Ad_T.T @ M @ Ad_T

def calculate_anti_windup(Kp_diag, factor):
    reverse = 1.0 / np.clip(Kp_diag, 1e-8, None)
    factor = np.clip(factor, 1/3, 3.0)
    reverse *= factor
    reverse = np.clip(reverse, 0, 1)
    return np.diag(reverse)

# ---------- Data Classes ----------
@dataclass
class PIDGains:
    p: np.ndarray
    i: np.ndarray
    d: np.ndarray
    a: np.ndarray = None
    p_length_max: float = None
    i_force_max: np.ndarray = None
    i_torque_max: np.ndarray = None

@dataclass
class ODAR:
    mass: float
    length: float
    joint_to_com: np.ndarray
    inertia_matrix: np.ndarray
    position_PID: PIDGains
    orientation_PID: PIDGains
    max_thrust: float
    B: np.ndarray
    Bnsv: dict
    body_joint_screw_axes: list
    joint_inertia_tensor: list

# ---------- Main Parameter Constructor ----------
def get_parameters():
    params = {}
    nlinks = 4
    params['LASDRA'] = {
        'total_link_number': nlinks,
    }

    # Default constants
    default_mass = 1.6
    default_length = 1.0
    joint_to_com = 0.5 * default_length * np.array([1.0, 0.0, 0.0])
    inertia_diag = [0.015, 0.1, 0.1]
    inertia_matrix = np.diag(inertia_diag)
    max_thrust = 8.0
    
    B = np.array([
        [0.6797]*8,
        [0.1908, 0.1908, -0.1908, -0.1908, 0.1908, 0.1908, -0.1908, -0.1908],
        [0.7082, -0.7082]*4,
        [-0.1026]*4 + [0.1026]*4,
        [-0.0711, 0.1980, -0.1980, 0.0711, 0.0711, -0.1980, 0.1980, -0.0711],
        [0.0894, -0.0169, 0.0169, -0.0894, -0.0894, 0.0169, -0.0169, 0.0894]
    ])

    Bnsv = {
        'upper': np.array([1, -1, -1, 1, 0, 0, 0, 0]),
        'lower': np.array([0, 0, 0, 0, 1, -1, -1, 1])
    }

    # Distributed Impedance Gains
    pos_p = np.diag([5.0, 3.0, 8.0])
    pos_i = np.diag([2.5, 2.0, 4.0])
    pos_d = np.diag([4.0, 2.0, 7.0])
    pos_a = calculate_anti_windup(np.diag(pos_p), 3)
    position_PID = PIDGains(p=pos_p, i=pos_i, d=pos_d, a=pos_a, p_length_max=0.6, i_force_max=np.array([15.0, 5.0, 20.0]))

    ori_p = np.diag([0.40, 1.20, 1.20])
    ori_i = np.diag([0.20, 0.30, 0.30])
    ori_d = np.diag([0.20, 0.90, 0.90])
    ori_a = calculate_anti_windup(np.diag(ori_p), 3)
    orientation_PID = PIDGains(p=ori_p, i=ori_i, d=ori_d, a=ori_a, i_torque_max=np.array([1.5, 2.0, 2.0]))

    # Screw axes (Az, Ay, Ax)
    Ax = np.array([1,0,0,0,0,0])
    Ay = np.array([0,1,0,0,0,0])
    Az = np.array([0,0,1,0,0,0])
    G0 = np.zeros((6, 6))

    # ODAR Modules
    odars = []
    for i in range(20):
        if i == 0:
            screw_axes = [Az, Ay, Ax]
            tensors = [G0, G0, get_inertia_tensor(default_mass, joint_to_com, inertia_matrix)]
        else:
            screw_axes = [Ay, Ax]
            tensors = [G0, get_inertia_tensor(default_mass, joint_to_com, inertia_matrix)]

        odar = ODAR(
            mass=default_mass,
            length=default_length,
            joint_to_com=joint_to_com,
            inertia_matrix=inertia_matrix,
            position_PID=position_PID,
            orientation_PID=orientation_PID,
            max_thrust=max_thrust,
            B=B,
            Bnsv=Bnsv,
            body_joint_screw_axes=screw_axes,
            joint_inertia_tensor=tensors
        )
        odars.append(odar)

    params['ODAR'] = odars[:nlinks]

    # LASDRA-level screw axes and inertias
    screw_axes_all = []
    inertia_all = []
    for odar in params['ODAR']:
        screw_axes_all.extend(odar.body_joint_screw_axes)
        inertia_all.extend(odar.joint_inertia_tensor)

    dof = len(screw_axes_all)
    params['LASDRA']['body_joint_screw_axes'] = screw_axes_all
    params['LASDRA']['inertia_matrix'] = inertia_all
    params['LASDRA']['dof'] = dof

    return params
