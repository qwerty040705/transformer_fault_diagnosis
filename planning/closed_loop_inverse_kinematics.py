import numpy as np
from dynamics.forward_kinematics_class import ForwardKinematics

class ClosedLoopInverseKinematics:
    def __init__(self, param,
                 alpha: float = 0.5,          # 기본 스텝 크기
                 err_tol: float = 1e-6,       # 오차 기준 
                 step_tol: float = 1e-8,      # dq 무한소 기준
                 stagnation_patience: int = 5,
                 max_iters: int = 500,        # ★ 최대 반복
                 step_clip: float = 0.2,      # ★ joint step 클립 (rad)
                 ls_eta: float = 0.5,         # ★ 백트래킹 비율
                 ls_max: int = 8):            # ★ 백트래킹 최대 횟수
        self.dof = param['LASDRA']['dof']
        self.lambda_damping = 1e-2            # 초기 댐핑
        self.err_tol = err_tol
        self.step_tol = step_tol
        self.stagnation_patience = stagnation_patience
        self.alpha = alpha
        self.max_iters = max_iters
        self.step_clip = step_clip
        self.ls_eta = ls_eta
        self.ls_max = ls_max
        self.forward_kinematics_class = ForwardKinematics(param)

    def solve(self, T_d, q_guess):
        q = np.asarray(q_guess, dtype=float).reshape(-1)
        fk = self.forward_kinematics_class

        prev_err = np.inf
        not_improved = 0
        lam = float(self.lambda_damping)

        for _ in range(self.max_iters):
            T_now = fk.compute_end_effector_frame(q)
            rot_err = self.get_rotation_error_vector(T_now[:3, :3], T_d[:3, :3])
            pos_err = T_now[:3, 3] - T_d[:3, 3]           # (act - des)
            e = np.concatenate((rot_err, pos_err))
            err_norm = np.linalg.norm(e)

            if err_norm <= self.err_tol:
                break

            J = fk.compute_end_effector_analytic_jacobian(q)  # (6, dof)
            dof = J.shape[1]

            I = np.eye(dof)
            solved = False
            lam_try = lam
            for _damp in range(3):  
                A = J.T @ J + (lam_try**2) * I
                try:
                    dq = -np.linalg.solve(A, J.T @ e)
                    solved = True
                    break
                except np.linalg.LinAlgError:
                    lam_try *= 10.0
            if not solved:
                dq, *_ = np.linalg.lstsq(J, -e, rcond=None)

            if np.max(np.abs(dq)) <= self.step_tol:
                not_improved += 1
                if not_improved >= self.stagnation_patience:
                    break
                dq = dq * 0.5
            else:
                m = np.max(np.abs(dq))
                if m > self.step_clip:
                    dq *= (self.step_clip / (m + 1e-12))

            alpha = self.alpha
            accepted = False
            for _ls in range(self.ls_max):
                q_trial = q + alpha * dq
                T_trial = fk.compute_end_effector_frame(q_trial)
                rot_err_t = self.get_rotation_error_vector(T_trial[:3, :3], T_d[:3, :3])
                pos_err_t = T_trial[:3, 3] - T_d[:3, 3]
                e_t = np.concatenate((rot_err_t, pos_err_t))
                err_t = np.linalg.norm(e_t)

                if err_t < err_norm * (1 - 1e-4):  
                    q = q_trial
                    if (err_norm - err_t) / (err_norm + 1e-12) < 1e-3:
                        not_improved += 1
                    else:
                        not_improved = 0
                    prev_err = err_t
                    accepted = True
                    break
                else:
                    alpha *= self.ls_eta  

            if not accepted:
                lam = min(lam_try * 2.0, 1e3)
                not_improved += 1
                if not_improved >= self.stagnation_patience:
                    break

        return q

    def get_link_SE3(self, q):
        return self.forward_kinematics_class.compute_CoM_frame(q)

    def get_link_GCSE3(self, q):
        return self.forward_kinematics_class.compute_GC_frame(q)

    def get_rotation_error_vector(self, R, R_d):
        return 0.5 * self.unskew(R_d.T @ R - R.T @ R_d)

    def unskew(self, M):
        return np.array([M[2,1], M[0,2], M[1,0]])
