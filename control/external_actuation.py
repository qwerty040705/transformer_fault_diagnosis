import numpy as np
from scipy.sparse import block_diag
from scipy.optimize import linprog
from control.selective_mapping import SelectiveMapping

class ExternalActuation:
    def __init__(self, params_model, lasdra_model):
        self.lasdra = lasdra_model
        self.nlinks = len(params_model["ODAR"])
        self.smaps = [SelectiveMapping(params_model["ODAR"][i]) for i in range(self.nlinks)]
        self.B_blkdiag = self._build_B_blkdiag(params_model)
        self.lambda_set_prev = None
        self.apply_selective_mapping = False 
        self._init_lp_variables(params_model)
        self._init_qp_variables(params_model)

    # ──────────────────────────────────────────────────────
    # Builders / init
    # ──────────────────────────────────────────────────────
    def _build_B_blkdiag(self, params_model):
        eye_perm = np.block([[np.zeros((3, 3)), np.eye(3)],
                             [np.eye(3), np.zeros((3, 3))]])
        B_blocks = []
        for odar in params_model["ODAR"]:
            B_blocks.append(eye_perm @ odar.B)
        return block_diag(B_blocks).toarray()

    def _init_lp_variables(self, params_model):
        m = self.B_blkdiag.shape[1]
        self.lp = {}
        self.lp["f"] = np.concatenate([np.zeros(m), [1.0]])  # [λ; t]
        self.lp["A"] = np.vstack([
            np.hstack([ np.eye(m), -np.ones((m, 1))]),
            np.hstack([-np.eye(m), -np.ones((m, 1))]),
        ])
        self.lp["b"] = np.zeros(2 * m)
        ub = self._generate_thrust_bounds(params_model)      # (m,)
        # bounds for [λ; t]
        self.lp["ub"] = np.append(ub, np.inf)
        self.lp["lb"] = np.append(-ub, 0.0)

    def _init_qp_variables(self, params_model):
        # QP bound만 재활용 (fallback에서도 씀)
        m = self.B_blkdiag.shape[1]
        self.qp = {}
        self.qp["ub"] = self._generate_thrust_bounds(params_model)
        self.qp["lb"] = -self.qp["ub"]

    def _generate_thrust_bounds(self, params_model):
        ub = []
        for odar in params_model["ODAR"]:
            # 각 링크의 로터 개수만큼 같은 상한
            ub.extend([odar.max_thrust] * odar.B.shape[1])
        return np.array(ub, dtype=float)

    # ──────────────────────────────────────────────────────
    # Public: main solver with LP → LS fallback
    # ──────────────────────────────────────────────────────
    def distribute_torque_lp(self, torque):
        """
        torque(=τ) : (dof,) or (dof,1)
        반환: λ (m,)  — 반드시 반환 (LP 실패 시 LS 폴백)
        """
        # ── 준비: 모양/타입 정리 ───────────────────────────────
        Aeq = self.lasdra.D.T @ self.B_blkdiag   # (dof x m)
        beq = np.asarray(torque, dtype=float).reshape(-1)
        m   = self.B_blkdiag.shape[1]
        ub  = self.qp["ub"]; lb = self.qp["lb"]

        # ── 1) LP 시도: min t  s.t.  -t ≤ λ ≤ t,  그리고  |Aeq λ - beq| ≤ eps_eq ─
        #     (Equality를 작은 여유 eps로 완화하여 infeasible 줄이기)
        eps_eq = 1e-6  # 필요시 1e-5 ~ 1e-4 까지 올려도 됨
        try:
            # 기존 ∞-norm epigraph 제약
            A_ub_box = np.vstack([
                np.hstack([ np.eye(m), -np.ones((m, 1))]),
                np.hstack([-np.eye(m), -np.ones((m, 1))]),
            ])
            b_ub_box = np.zeros(2 * m)

            # |Aeq λ - beq| ≤ eps_eq  →  [ Aeq  0 ] [λ;t] ≤ beq + eps_eq
            #                           [-Aeq  0 ] [λ;t] ≤ -beq + eps_eq
            A_ub_eq = np.vstack([
                np.hstack([ Aeq, np.zeros((Aeq.shape[0], 1))]),
                np.hstack([-Aeq, np.zeros((Aeq.shape[0], 1))]),
            ])
            b_ub_eq = np.concatenate([beq + eps_eq, -beq + eps_eq])

            A_ub = np.vstack([A_ub_box, A_ub_eq])
            b_ub = np.concatenate([b_ub_box, b_ub_eq])

            # 목적함수: c = [0..0, 1]  (t만 최소화)
            c = np.concatenate([np.zeros(m), [1.0]])

            # 변수 경계: λ는 [-ub, ub], t는 [0, +inf]
            bounds = list(zip(np.append(lb, 0.0), np.append(ub, np.inf)))

            # Equality는 부등식으로 완화했으므로 A_eq=None
            res = linprog(
                c=c, A_ub=A_ub, b_ub=b_ub, A_eq=None, b_eq=None, bounds=bounds, method='highs'
            )

            if res.success:
                lam = res.x[:-1]  # drop t
            else:
                # ── 2) Fallback: 경계 포함 감쇠 LS ─────────────────
                lam = self._least_squares_fallback(beq, Aeq)

        except Exception:
            # LP에서 예외가 나도 항상 폴백으로 커버
            lam = self._least_squares_fallback(beq, Aeq)

        # ── 선택: Selective Mapping 적용 + box projection ─────────
        if self.apply_selective_mapping:
            Fodar = self.B_blkdiag @ lam
            lam = self._adjust_thrust_across_links(Fodar)
            lam = np.clip(lam, lb, ub)

        self.lambda_set_prev = lam
        return lam

    # ──────────────────────────────────────────────────────
    # Fallback: damped least-squares + box projection
    #   min ||Aeq λ − τ||² + μ||λ||²  s.t. lb ≤ λ ≤ ub
    # ──────────────────────────────────────────────────────
    def _least_squares_fallback(self, beq, Aeq, max_iter=8, mu=1e-6, tol=1e-4):
        """
        반복적으로 λ를 box(-ub..ub)로 project하면서 잔차를 보정.
        항상 수렴적인 근사해를 제공 (실패없음).
        """
        m = Aeq.shape[1]
        ub = self.qp["ub"]; lb = self.qp["lb"]
        At = Aeq.T
        # normal eq.: (A A^T + μI) y = τ,  λ0 = A^T y
        M = Aeq @ At + mu * np.eye(Aeq.shape[0])
        try:
            y = np.linalg.solve(M, beq)
        except np.linalg.LinAlgError:
            # 매우 드문 경우 — pseudo inverse fallback
            y = np.linalg.pinv(M) @ beq
        lam = At @ y

        for _ in range(max_iter):
            lam = np.clip(lam, lb, ub)
            r = beq - Aeq @ lam
            if np.linalg.norm(r, ord=np.inf) < tol:
                break
            # 다음 보정
            try:
                y = np.linalg.solve(M, r)
            except np.linalg.LinAlgError:
                y = np.linalg.pinv(M) @ r
            lam = lam + At @ y

        return np.clip(lam, lb, ub)

    # ──────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────
    def _adjust_thrust_across_links(self, Fodar):
        thrusts = []
        for i in range(self.nlinks):
            Fi = Fodar[i*6:(i+1)*6]
            wrench = self._convert_matrix_to_wrench(Fi)
            thrusts.append(self.smaps[i].get_adjusted_thrust(wrench))
        return np.concatenate(thrusts)

    @staticmethod
    def _convert_matrix_to_wrench(F):
        return {"torque": F[:3], "force": F[3:]}
