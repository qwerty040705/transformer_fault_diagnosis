# --- control/external_actuation.py ---

import numpy as np
from scipy.sparse import block_diag
from scipy.optimize import linprog
from control.selective_mapping import SelectiveMapping


class ExternalActuation:
    """
    τ(관절토크)를 각 링크의 추진기 추력 λ로 분배.
    1) LP:    min t  s.t. Aeq λ = τ,  -t ≤ λ_i ≤ t,  lb ≤ λ ≤ ub,  t ≥ 0
              (하드 equality + epigraph box)
       - infeasible 시 lb/ub를 점차 확장해 재시도
    2) 실패 시: 감쇠 LS + box projection 폴백 (확장된 bound 사용)

    선택적으로 SelectiveMapping 적용 후 최종 box-clip.
    """
    def __init__(self, params_model, lasdra_model):
        self.lasdra = lasdra_model
        self.nlinks = len(params_model["ODAR"])

        # per-link selective mapping
        self.smaps = [SelectiveMapping(params_model["ODAR"][i]) for i in range(self.nlinks)]

        # 큰 B 블록 대각행렬 (링크별 6×m_i)
        self.B_blkdiag = self._build_B_blkdiag(params_model)

        self.lambda_set_prev = None
        self.apply_selective_mapping = False

        # LP/QP용 파라미터
        self._init_lp_variables(params_model)
        self._init_qp_variables(params_model)

    # ──────────────────────────────────────────────────────
    # Builders / init
    # ──────────────────────────────────────────────────────
    def _build_B_blkdiag(self, params_model):
        """
        각 링크의 B를 '원본 그대로' 블록대각으로 구성.
        (렌치 순서 스왑 없음)
        """
        from scipy.sparse import block_diag

        B_blocks = [odar.B for odar in params_model["ODAR"]] 
        return block_diag(B_blocks).toarray()                  

    def _generate_thrust_bounds(self, params_model):
        """
        추진기 상한(크기 동일)으로부터 전역 ub, lb 생성.
        """
        ub = []
        for odar in params_model["ODAR"]:
            ub.extend([odar.max_thrust] * odar.B.shape[1])
        return np.asarray(ub, dtype=float)

    def _init_qp_variables(self, params_model):
        m = self.B_blkdiag.shape[1]
        self.qp = {}
        self.qp["ub"] = self._generate_thrust_bounds(params_model)      # (m,)
        self.qp["lb"] = -self.qp["ub"]

    def _init_lp_variables(self, params_model):
        """
        LP 변수: x = [λ; t] (길이 m+1)
        목적: min t
        제약:
          - box epigraph: -t ≤ λ_i ≤ t
          - equality: Aeq λ = τ   (Aeq = D^T B_blkdiag)
          - λ bounds: lb ≤ λ ≤ ub  (t는 [0, +inf))
        """
        m = self.B_blkdiag.shape[1]
        self.lp = {}

        # 목적함수: c = [0..0, 1]  (t만 최소화)
        self.lp["c"] = np.concatenate([np.zeros(m), [1.0]])

        # epigraph box 불등식:  [ I  -1] [λ;t] ≤ 0,  [-I  -1] [λ;t] ≤ 0
        A_ub_box = np.vstack([
            np.hstack([ np.eye(m), -np.ones((m, 1))]),
            np.hstack([-np.eye(m), -np.ones((m, 1))]),
        ])
        b_ub_box = np.zeros(2 * m)
        self.lp["A_ub_box"] = A_ub_box
        self.lp["b_ub_box"] = b_ub_box

        # 초기 bounds(정보용; 실제 LP 호출 때는 확장된 값으로 재생성)
        ub = self._generate_thrust_bounds(params_model)
        lb = -ub
        self.lp["bounds_init"] = list(zip(np.append(lb, 0.0), np.append(ub, np.inf)))

    # ──────────────────────────────────────────────────────
    # Public: LP(하드 equality + bound 확장) → 실패 시 LS 폴백
    # ──────────────────────────────────────────────────────
    def distribute_torque_lp(self, torque, max_expand_tries=3, tol_eq=1e-9):
        """
        torque τ: (dof,) or (dof, 1)
        반환: λ (m,)
        """
        # 준비
        Aeq = self.lasdra.D.T @ self.B_blkdiag   # (dof × m)
        beq = np.asarray(torque, dtype=float).reshape(-1)
        m   = self.B_blkdiag.shape[1]

        # bounds는 시도마다 확장될 수 있으므로 로컬 복사로 시작
        lb = self.qp["lb"].copy()   # (m,)
        ub = self.qp["ub"].copy()   # (m,)

        # [λ; t] bounds 생성 유틸
        def make_bounds(lb_, ub_):
            return list(zip(np.append(lb_, 0.0), np.append(ub_, np.inf)))

        # 하드 equality: A_eq_ext @ [λ;t] = beq
        A_eq_ext = np.hstack([Aeq, np.zeros((Aeq.shape[0], 1))])

        # 재시도 루프
        for attempt in range(max_expand_tries + 1):
            try:
                res = linprog(
                    c=self.lp["c"],
                    A_ub=self.lp["A_ub_box"], b_ub=self.lp["b_ub_box"],
                    A_eq=A_eq_ext, b_eq=beq,
                    bounds=make_bounds(lb, ub),
                    method="highs",
                )
            except Exception:
                res = None

            if res is not None and res.success:
                lam = res.x[:-1]  # drop t
                # equality 오차 확인 (수치적으로 꽤 빡빡하게)
                if np.linalg.norm(Aeq @ lam - beq, ord=np.inf) <= tol_eq:
                    # 선택적 매핑
                    if self.apply_selective_mapping:
                        Fodar = self.B_blkdiag @ lam  # (6N,)
                        lam = self._adjust_thrust_across_links(Fodar)
                    # 최종 clip
                    lam = np.clip(lam, lb, ub)

                    # 성공한 bound를 내부 상태에도 반영
                    self.qp["lb"], self.qp["ub"] = lb.copy(), ub.copy()
                    self.lambda_set_prev = lam.copy()
                    return lam

            # 실패/불충분 → bounds 2배 확장 (마지막 시도 전까지만)
            if attempt < max_expand_tries:
                lb *= 2.0
                ub *= 2.0

        # 여기까지 왔으면 LP가 안 맞음 → 감쇠 LS 폴백 (확장된 bound 사용!!)
        lam = self._least_squares_fallback(beq, Aeq, lb, ub)
        if self.apply_selective_mapping:
            Fodar = self.B_blkdiag @ lam
            lam = self._adjust_thrust_across_links(Fodar)
        lam = np.clip(lam, lb, ub)

        self.qp["lb"], self.qp["ub"] = lb.copy(), ub.copy()
        self.lambda_set_prev = lam.copy()
        return lam

    # ──────────────────────────────────────────────────────
    # Fallback: damped least-squares + box projection
    #   min ||Aeq λ − τ||² + μ||λ||²  s.t. lb ≤ λ ≤ ub
    # ──────────────────────────────────────────────────────
    def _least_squares_fallback(self, beq, Aeq, lb, ub, max_iter=8, mu=1e-6, tol=1e-4):
        """
        확장된 bounds(lb, ub)를 인자로 받아, 반복적으로 clip하며 잔차 보정.
        항상 수렴적인 근사해를 제공.
        """
        At = Aeq.T
        # normal eq: (A A^T + μI) y = τ
        M = Aeq @ At + mu * np.eye(Aeq.shape[0])
        try:
            y = np.linalg.solve(M, beq)
        except np.linalg.LinAlgError:
            y = np.linalg.pinv(M) @ beq
        lam = At @ y  # 초기치

        for _ in range(max_iter):
            lam = np.clip(lam, lb, ub)
            r = beq - Aeq @ lam
            if np.linalg.norm(r, ord=np.inf) < tol:
                break
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
        """
        링크별 6D 렌치 → selective mapping으로 각 링크 추진기 추력으로 보정.
        Fodar: shape (6*N,)
        반환: λ_concat: shape (Σ m_i,)
        """
        thrusts = []
        for i in range(self.nlinks):
            Fi = Fodar[i * 6:(i + 1) * 6]
            wrench = self._convert_matrix_to_wrench(Fi)
            thrusts.append(self.smaps[i].get_adjusted_thrust(wrench))
        return np.concatenate(thrusts, axis=0)

    @staticmethod
    def _convert_matrix_to_wrench(F):
        return {"torque": F[:3], "force": F[3:]}
