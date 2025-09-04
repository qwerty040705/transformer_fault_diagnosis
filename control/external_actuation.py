import numpy as np
from scipy.sparse import block_diag
from scipy.optimize import linprog
from control.selective_mapping import SelectiveMapping


class ExternalActuation:
    MAT_CLIP = 1e8   
    VEC_CLIP = 1e8   

    def __init__(self, params_model, lasdra_model):
        self.lasdra = lasdra_model
        self.nlinks = len(params_model["ODAR"])
        self.smaps = [SelectiveMapping(params_model["ODAR"][i]) for i in range(self.nlinks)]

        self.B_blkdiag = self._build_B_blkdiag(params_model)
        self.B_blkdiag = self._sanitize_mat(self.B_blkdiag, clip=self.MAT_CLIP)

        self.lambda_set_prev = None
        self.apply_selective_mapping = False

        self._init_lp_variables(params_model)
        self._init_qp_variables(params_model)

    @staticmethod
    def _sanitize_mat(M, clip=1e8):
        M = np.asarray(M, dtype=float)
        M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
        if clip is not None:
            M = np.clip(M, -clip, clip)
        return M

    @staticmethod
    def _sanitize_vec(v, clip=1e8):
        v = np.asarray(v, dtype=float).reshape(-1)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        if clip is not None:
            v = np.clip(v, -clip, clip)
        return v

    @staticmethod
    def _safe_solve(A, b):
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A) @ b

    @classmethod
    def _safe_matvec(cls, M, v, mat_clip=None, vec_clip=None):
        M = cls._sanitize_mat(M, clip=mat_clip if mat_clip is not None else cls.MAT_CLIP)
        v = cls._sanitize_vec(v, clip=vec_clip if vec_clip is not None else cls.VEC_CLIP)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            y = M @ v
        y = cls._sanitize_vec(y, clip=vec_clip if vec_clip is not None else cls.VEC_CLIP)
        return y

    # ---------- 내부 구성 ----------
    def _build_B_blkdiag(self, params_model):
        B_blocks = [np.asarray(odar.B, dtype=float) for odar in params_model["ODAR"]]
        return block_diag(B_blocks).toarray()

    def _generate_thrust_bounds(self, params_model):
        ub = []
        for odar in params_model["ODAR"]:
            ub.extend([float(odar.max_thrust)] * odar.B.shape[1])
        return np.asarray(ub, dtype=float)

    def _init_qp_variables(self, params_model):
        m = self.B_blkdiag.shape[1]
        self.qp = {}
        self.qp["ub"] = self._generate_thrust_bounds(params_model)  
        self.qp["lb"] = -self.qp["ub"]                              

    def _init_lp_variables(self, params_model):
        m = self.B_blkdiag.shape[1]
        self.lp = {}
        self.lp["c"] = np.concatenate([np.zeros(m), [1.0]])

        A_ub_box = np.vstack([
            np.hstack([ np.eye(m), -np.ones((m, 1))]), 
            np.hstack([-np.eye(m), -np.ones((m, 1))]),  
        ])
        b_ub_box = np.zeros(2 * m)
        self.lp["A_ub_box"] = A_ub_box
        self.lp["b_ub_box"] = b_ub_box

        ub = self._generate_thrust_bounds(params_model)
        lb = -ub
        self.lp["bounds_init"] = list(zip(np.append(lb, 0.0), np.append(ub, np.inf)))

    def distribute_torque_lp(self, torque, max_expand_tries=3, tol_eq=1e-8):
        """
        torque: (ndof,) 또는 (ndof,1)
        반환: λ (m,)
        """
        D = self._sanitize_mat(self.lasdra.D, clip=self.MAT_CLIP)
        B = self.B_blkdiag  

        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            Aeq = D.T @ B
        Aeq = self._sanitize_mat(Aeq, clip=self.MAT_CLIP)

        beq = self._sanitize_vec(torque, clip=self.VEC_CLIP)

        m = B.shape[1]
        lb = self._sanitize_vec(self.qp["lb"], clip=self.VEC_CLIP)
        ub = self._sanitize_vec(self.qp["ub"], clip=self.VEC_CLIP)

        def make_bounds(lb_, ub_):
            return list(zip(np.append(lb_, 0.0), np.append(ub_, np.inf)))

        A_eq_ext = np.hstack([Aeq, np.zeros((Aeq.shape[0], 1))])

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

            if (res is not None) and res.success:
                lam = self._sanitize_vec(res.x[:-1], clip=self.VEC_CLIP)

                resid = self._safe_matvec(Aeq, lam) - beq
                if np.linalg.norm(resid, ord=np.inf) <= tol_eq:
                    if self.apply_selective_mapping:
                        Fodar = self._safe_matvec(self.B_blkdiag, lam)
                        lam = self._adjust_thrust_across_links(Fodar)
                        lam = self._sanitize_vec(lam, clip=self.VEC_CLIP)

                    lam = np.clip(lam, lb, ub)
                    self.qp["lb"], self.qp["ub"] = lb.copy(), ub.copy()
                    self.lambda_set_prev = lam.copy()
                    return lam

            if attempt < max_expand_tries:
                lb *= 2.0
                ub *= 2.0
                lb = self._sanitize_vec(lb, clip=self.VEC_CLIP)
                ub = self._sanitize_vec(ub, clip=self.VEC_CLIP)

        lam = self._least_squares_fallback(beq, Aeq, lb, ub)

        if self.apply_selective_mapping:
            Fodar = self._safe_matvec(self.B_blkdiag, lam)
            lam = self._adjust_thrust_across_links(Fodar)
            lam = self._sanitize_vec(lam, clip=self.VEC_CLIP)

        lam = np.clip(lam, lb, ub)
        self.qp["lb"], self.qp["ub"] = lb.copy(), ub.copy()
        self.lambda_set_prev = lam.copy()
        return lam

    def _least_squares_fallback(self, beq, Aeq, lb, ub, max_iter=8, mu=1e-6, tol=1e-4):
        Aeq = self._sanitize_mat(Aeq, clip=self.MAT_CLIP)
        beq = self._sanitize_vec(beq, clip=self.VEC_CLIP)
        lb  = self._sanitize_vec(lb,  clip=self.VEC_CLIP)
        ub  = self._sanitize_vec(ub,  clip=self.VEC_CLIP)

        At = Aeq.T
        M = self._sanitize_mat(Aeq @ At + mu * np.eye(Aeq.shape[0]), clip=self.MAT_CLIP)
        y = self._safe_solve(M, beq)
        lam = self._sanitize_vec(At @ y, clip=self.VEC_CLIP)

        for _ in range(max_iter):
            lam = np.clip(lam, lb, ub)
            r = beq - self._safe_matvec(Aeq, lam)
            if np.linalg.norm(r, ord=np.inf) < tol:
                break
            y = self._safe_solve(M, r)
            lam = self._sanitize_vec(lam + At @ y, clip=self.VEC_CLIP)

        return np.clip(lam, lb, ub)

    def _adjust_thrust_across_links(self, Fodar):
        Fodar = self._sanitize_vec(Fodar, clip=self.VEC_CLIP)
        thrusts = []
        for i in range(self.nlinks):
            Fi = Fodar[i * 6:(i + 1) * 6]
            if Fi.shape[0] != 6:
                Fi = np.zeros(6, dtype=float)
            wrench = self._convert_matrix_to_wrench(Fi)
            thrusts.append(self.smaps[i].get_adjusted_thrust(wrench))
        lam_adj = np.concatenate(thrusts, axis=0)
        return self._sanitize_vec(lam_adj, clip=self.VEC_CLIP)

    @staticmethod
    def _convert_matrix_to_wrench(F):
        F = np.asarray(F, dtype=float).reshape(-1)
        return {"torque": F[:3], "force": F[3:]}
