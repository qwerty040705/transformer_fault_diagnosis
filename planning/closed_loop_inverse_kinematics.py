import numpy as np
from dynamics.forward_kinematics_class import ForwardKinematics

class ClosedLoopInverseKinematics:
    def __init__(self, param):
        self.dof = param['LASDRA']['dof']
        self.lambda_damping = 0.01 * np.eye(self.dof)
        self.max_iteration = 100
        self.convergence_criteria = 1e-8
        self.dt = 0.99
        self.forward_kinematics_class = ForwardKinematics(param)

    def solve(self, T_d, q_guess):
        for itr in range(self.max_iteration):
            T_itr = self.forward_kinematics_class.compute_end_effector_frame(q_guess)

            se3error = np.concatenate((
                self.get_rotation_error_vector(T_itr[:3, :3], T_d[:3, :3]),
                T_itr[:3, 3] - T_d[:3, 3]
            ))

            if np.linalg.norm(se3error) < self.convergence_criteria:
                break

            J_a = self.forward_kinematics_class.compute_end_effector_analytic_jacobian(q_guess)
            A = J_a.T @ J_a + (self.lambda_damping ** 2) * np.eye(J_a.shape[1])
            dq = -np.linalg.solve(A, J_a.T @ se3error)
            q_guess = q_guess + self.dt * dq

        return q_guess

    def get_link_SE3(self, q):
        return self.forward_kinematics_class.compute_CoM_frame(q)

    def get_link_GCSE3(self, q):
        return self.forward_kinematics_class.compute_GC_frame(q)

    def get_rotation_error_vector(self, R, R_d):
        return 0.5 * self.unskew(R_d.T @ R - R.T @ R_d)

    def unskew(self, R):
        return np.array([
            R[2, 1],
            R[0, 2],
            R[1, 0]
        ])
