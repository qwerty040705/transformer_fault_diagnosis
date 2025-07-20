import numpy as np

class EndEffectorImpedanceController:
    def __init__(self, params, lasdra_model):
        self.lasdra_model = lasdra_model
        self.T_ef_buffer = {'position': [], 'orientation': []}

        self.M_d_ef_inv = np.linalg.inv(params['ef_impedance']['desired_mass'])
        self.D_d_ef = params['ef_impedance']['desired_damping']
        self.K_d_ef = params['ef_impedance']['desired_spring']

        self.M_R_d_ef_inv = np.linalg.inv(params['ef_impedance']['desired_rotational_mass'])
        self.D_R_d_ef = np.block([[params['ef_impedance']['desired_rotational_damping']]])
        self.K_R_d_ef = np.block([[params['ef_impedance']['desired_rotational_spring']]])

        self.e_position_max = 0.5
        self.K_nsp = params['ef_impedance']['K_nullspace']
        self.B_nsp = params['ef_impedance']['B_nullspace']
        self.torque_limit_nominal = params['ef_impedance']['torque_limit']
        self.dof = params['dof']
        self.torque_limit_ub = self.torque_limit_nominal * np.ones((self.dof,))
        self.torque_limit_lb = -self.torque_limit_ub

        self.T_ef_d_buffer_max = 4
        self.J_e_prev = None
        self.e_prev = None
        self.de_prev = None
        self.num_diff_damp_coeff = 0.97
        self.q_d = None
        self.dq_d = None

    def get_control_output(self, dt, T_ef, joint_state):
        x_ef = T_ef[:3, 3]
        R_ef = T_ef[:3, :3]
        q = joint_state['q']
        dq = joint_state['dq']
        M = self.lasdra_model.Mass
        C = self.lasdra_model.Cori
        G = self.lasdra_model.Grav

        J_e = self.lasdra_model.get_end_effector_analytic_jacobian()
        if self.J_e_prev is None:
            dJ = np.zeros_like(J_e)
        else:
            dJ = (J_e - self.J_e_prev) / dt
        self.J_e_prev = J_e

        R_d = self.T_ef_buffer['orientation'][-1]
        e_orientation = 0.5 * self.unskew(R_d.T @ R_ef - R_ef.T @ R_d)
        e_position = x_ef - self.T_ef_buffer['position'][-1]
        e = np.concatenate((e_orientation, e_position), axis=0)

        de = self.calculate_velocity_error(dt, e)
        ddx_d = self.get_end_effector_acceleration(dt)
        rho_square = 0.02 * np.tanh(1 / (1e3 * np.linalg.det(J_e @ J_e.T) + 1e-10))
        pinv_J_ep_rho = J_e.T @ np.linalg.inv(J_e @ J_e.T + rho_square * np.eye(6))

        e_clip = e.copy()
        e_clip[3:] = self.limit_vector_norm(e[3:], self.e_position_max)

        torque = (
            M @ pinv_J_ep_rho @ (
                np.block([
                    [self.M_R_d_ef_inv, np.zeros((3,3))],
                    [np.zeros((3,3)), self.M_d_ef_inv]
                ]) @ (-np.block([
                    [self.D_R_d_ef, np.zeros((3,3))],
                    [np.zeros((3,3)), self.D_d_ef]
                ]) @ de - np.block([
                    [self.K_R_d_ef, np.zeros((3,3))],
                    [np.zeros((3,3)), self.K_d_ef]
                ]) @ e_clip)
                + ddx_d - dJ @ dq
            )
            + C @ dq + G
        )
        return self.bound(torque, self.torque_limit_lb, self.torque_limit_ub)

    def update_desired_end_effector_pose(self, T_ef_d):
        x_ef_d = T_ef_d[:3, 3]
        R_ef_d = T_ef_d[:3, :3]
        if len(self.T_ef_buffer['position']) < self.T_ef_d_buffer_max:
            self.T_ef_buffer['position'].append(x_ef_d)
            self.T_ef_buffer['orientation'].append(R_ef_d)
        else:
            self.T_ef_buffer['position'] = self.T_ef_buffer['position'][1:] + [x_ef_d]
            self.T_ef_buffer['orientation'] = self.T_ef_buffer['orientation'][1:] + [R_ef_d]

    def calculate_velocity_error(self, dt, e):
        if self.e_prev is None:
            de = np.zeros_like(e)
        else:
            de = (e - self.e_prev) / dt
            de = self.get_damped_value(self.de_prev, de, self.num_diff_damp_coeff)
        self.e_prev = e
        self.de_prev = de
        return de

    def get_end_effector_acceleration(self, dt):
        x_buf = self.T_ef_buffer['position']
        n = len(x_buf)
        if n == 1:
            coeff = [0]
        elif n == 2:
            coeff = [0, 0]
        elif n == 3:
            coeff = [1, -2, 1]
        else:
            coeff = [-1, 4, -5, 2]
        ddx_p = np.dot(np.stack(x_buf[-len(coeff):], axis=1), coeff) / (dt ** 2)
        return np.concatenate((np.zeros(3), ddx_p))

    @staticmethod
    def unskew(R):
        return np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ])

    @staticmethod
    def get_damped_value(prev, curr, alpha):
        return (1 - alpha) * prev + alpha * curr

    @staticmethod
    def bound(v, v_min, v_max):
        return np.maximum(v_min, np.minimum(v_max, v))

    @staticmethod
    def limit_vector_norm(v, max_norm):
        norm_v = np.linalg.norm(v)
        return v * max_norm / norm_v if norm_v > max_norm else v
