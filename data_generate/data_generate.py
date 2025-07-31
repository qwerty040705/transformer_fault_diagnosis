import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 
import numpy as np
from dynamics.forward_kinematics_class import ForwardKinematics
from dynamics.lasdra_class import LASDRA
from planning.closed_loop_inverse_kinematics import ClosedLoopInverseKinematics
from fault_injection import inject_faults
from parameters import get_parameters
from parameters_model import parameters_model

def generate_random_trajectory(link_count, T=200, fault_time=100, epsilon_scale=0.05):
    base_param = get_parameters()
    base_param['LASDRA']['total_link_number'] = link_count
    base_param['ODAR'] = base_param['ODAR'][:link_count]

    screw_axes_all = []
    inertia_all = []
    for odar in base_param['ODAR']:
        screw_axes_all.extend(odar['body_joint_screw_axes'])
        inertia_all.extend(odar['joint_inertia_tensor'])
    base_param['LASDRA']['body_joint_screw_axes'] = screw_axes_all
    base_param['LASDRA']['inertia_matrix'] = inertia_all
    base_param['LASDRA']['dof'] = len(screw_axes_all)

    model_param = parameters_model(mode=0, params_prev=base_param)
    robot = LASDRA(model_param)
    fk_solver = ForwardKinematics(model_param)
    ik_solver = ClosedLoopInverseKinematics(model_param)

    dof = model_param['LASDRA']['dof']
    q_des = np.zeros((T, dof, 1))
    dq_des = np.zeros((T, dof, 1))
    ddq_des = np.zeros((T, dof, 1))

    q_des[0,:,0] = (2 * np.random.rand(dof) - 1) * np.pi
    dq_des[0,:,0] = 0
    ddq_des[0,:,0] = 0

    dt = 0.1
    for t in range(1, T):
        delta_q = (2 * np.random.rand(dof) - 1) * 0.1
        q_new = q_des[t-1,:,0] + delta_q
        q_new = np.mod(q_new + np.pi, 2*np.pi) - np.pi

        T_new = fk_solver.compute_end_effector_frame(q_new)
        x, y, z = T_new[0,3], T_new[1,3], T_new[2,3]
        attempts = 0
        while (z < -1 or z > 3 or np.hypot(x, y) > link_count * 1.0) and attempts < 5:
            delta_q = -0.5 * delta_q
            q_new = q_des[t-1,:,0] + delta_q
            q_new = np.mod(q_new + np.pi, 2*np.pi) - np.pi
            T_new = fk_solver.compute_end_effector_frame(q_new)
            x, y, z = T_new[0,3], T_new[1,3], T_new[2,3]
            attempts += 1
        if z < -1 or z > 3 or np.hypot(x,y) > link_count*1.0:
            q_new = q_des[t-1,:,0]
        q_des[t,:,0] = q_new
        dq_des[t,:,0] = (q_des[t,:,0] - q_des[t-1,:,0]) / dt
        ddq_des[t,:,0] = (dq_des[t,:,0] - dq_des[t-1,:,0]) / dt if t>1 else 0

    T_des_series = [fk_solver.compute_end_effector_frame(q_des[t,:,0]) for t in range(T)]

    lambda_des = np.zeros((T, 8*link_count))
    for t in range(T):
        robot.set_joint_states(q_des[t], dq_des[t])
        tau = robot.Mass @ ddq_des[t] + robot.Cori @ dq_des[t] + robot.Grav
        tau = tau.flatten()
        H = robot.D.T @ robot.B_blkdiag
        lambda_sol = np.linalg.pinv(H) @ tau
        lambda_des[t, :] = lambda_sol

    lambda_faulty, type_matrix = inject_faults(lambda_des, fault_time=fault_time, epsilon_scale=epsilon_scale)

    actual_q = np.zeros_like(q_des)
    actual_dq = np.zeros_like(dq_des)
    actual_q[0] = q_des[0]
    actual_dq[0] = dq_des[0]
    robot.set_joint_states(actual_q[0], actual_dq[0])
    T_actual_series = [fk_solver.compute_end_effector_frame(actual_q[0,:,0])]
    for t in range(1, T):
        F_total = np.zeros((6*link_count, 1))
        for link_idx in range(link_count):
            thrust_i = lambda_faulty[t, link_idx*8:(link_idx+1)*8]
            Fi = robot.B_cell[link_idx] @ thrust_i.reshape(-1,1)
            F_total[6*link_idx:6*(link_idx+1)] = Fi
        tau_fault = robot.D.T @ F_total
        next_state = robot.get_next_joint_states(dt, tau_fault)
        actual_q[t] = next_state['q']
        actual_dq[t] = next_state['dq']
        robot.set_joint_states(actual_q[t], actual_dq[t])
        T_actual_series.append(fk_solver.compute_end_effector_frame(actual_q[t,:,0]))

    return T_des_series, T_actual_series, type_matrix

if __name__ == '__main__':
    link_count = int(input("How many links?: "))

    os.makedirs("data_storage", exist_ok=True)

    NUM_SAMPLES = 1000
    all_desired = []
    all_actual = []
    all_labels = []
    for _ in range(NUM_SAMPLES):
        T_des, T_act, label = generate_random_trajectory(link_count=link_count)
        all_desired.append(np.stack(T_des))
        all_actual.append(np.stack(T_act))
        all_labels.append(label)
    all_desired = np.array(all_desired)
    all_actual = np.array(all_actual)
    all_labels = np.array(all_labels)

    print("Saving to fault_dataset.npz...")
    np.savez("data_storage/fault_dataset.npz", desired=all_desired, actual=all_actual, label=all_labels)
    print("Dataset saved successfully.")
