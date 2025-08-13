import numpy as np
from dynamics.forward_kinematics_class import ForwardKinematics

class ClosedLoopInverseKinematics:
    def __init__(self, param,
                 alpha: float = 0.5,     # 스텝 사이즈
                 err_tol: float = 1e-10,   # SE(3) 오차 기준
                 step_tol: float = 1e-10,  # dq 크기 기준
                 stagnation_patience: int = 2):
        self.dof = param['LASDRA']['dof']
        self.lambda_damping = 0.01
        self.err_tol = err_tol
        self.step_tol = step_tol
        self.stagnation_patience = stagnation_patience
        self.alpha = alpha
        self.forward_kinematics_class = ForwardKinematics(param)

    def solve(self, T_d, q_guess):
        stagnation_cnt = 0
        q = q_guess.copy()

        while True:
            # 현재 EE 포즈와 오차
            T_now = self.forward_kinematics_class.compute_end_effector_frame(q)
            rot_err = self.get_rotation_error_vector(T_now[:3, :3], T_d[:3, :3])
            pos_err = T_now[:3, 3] - T_d[:3, 3]
            se3error = np.concatenate((rot_err, pos_err))

            # 오차 기준 종료
            if np.linalg.norm(se3error) <= self.err_tol:
                break

            # 자코비안과 damped least squares
            J_a = self.forward_kinematics_class.compute_end_effector_analytic_jacobian(q)
            lam2 = (self.lambda_damping ** 2)
            A = J_a.T @ J_a + lam2 * np.eye(J_a.shape[1])
            dq = -np.linalg.solve(A, J_a.T @ se3error)

            # 스텝 크기 기준 종료
            step_inf = np.max(np.abs(dq))
            if step_inf <= self.step_tol:
                stagnation_cnt += 1
                if stagnation_cnt >= self.stagnation_patience:
                    break
            else:
                stagnation_cnt = 0

            q = q + self.alpha * dq

        return q

    def get_link_SE3(self, q):
        return self.forward_kinematics_class.compute_CoM_frame(q)

    def get_link_GCSE3(self, q):
        return self.forward_kinematics_class.compute_GC_frame(q)

    def get_rotation_error_vector(self, R, R_d):
        return 0.5 * self.unskew(R_d.T @ R - R.T @ R_d)

    def unskew(self, M):
        return np.array([M[2,1], M[0,2], M[1,0]])
