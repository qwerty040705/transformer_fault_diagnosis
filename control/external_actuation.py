# external_actuation.py

import numpy as np
from scipy.sparse import block_diag
from scipy.optimize import linprog, minimize
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
    
    def _build_B_blkdiag(self, params_model):
        eye_perm = np.block([[np.zeros((3, 3)), np.eye(3)],
                             [np.eye(3), np.zeros((3, 3))]])
        B_blocks = []
        for odar in params_model["ODAR"]:
            B_blocks.append(eye_perm @ odar["B"])
        return block_diag(B_blocks).toarray()

    def _init_lp_variables(self, params_model):
        # LP: minimize max thrust
        num_rotor = self.B_blkdiag.shape[1]
        self.lp = {}
        self.lp["f"] = np.concatenate([np.zeros(num_rotor), [1]])
        self.lp["A"] = np.vstack([
            np.hstack([np.eye(num_rotor), -np.ones((num_rotor, 1))]),
            np.hstack([-np.eye(num_rotor), -np.ones((num_rotor, 1))])
        ])
        self.lp["b"] = np.zeros(2 * num_rotor)
        ub = self._generate_thrust_bounds(params_model)
        self.lp["ub"] = np.append(ub, np.inf)
        self.lp["lb"] = np.append(-ub, 0)

    def _init_qp_variables(self, params_model):
        # QP: minimize energy (sum of squares)
        num_rotor = self.B_blkdiag.shape[1]
        self.qp = {}
        self.qp["H"] = np.eye(num_rotor)
        self.qp["f"] = np.zeros(num_rotor)
        self.qp["ub"] = self._generate_thrust_bounds(params_model)
        self.qp["lb"] = -self.qp["ub"]

    def _generate_thrust_bounds(self, params_model):
        ub = []
        for odar in params_model["ODAR"]:
            ub.extend([odar["max_thrust"]] * odar["B"].shape[1])
        return np.array(ub)

    def distribute_torque_lp(self, torque):
        Aeq = self.lasdra.D.T @ self.B_blkdiag
        beq = torque
        res = linprog(
            c=self.lp["f"],
            A_ub=self.lp["A"],
            b_ub=self.lp["b"],
            A_eq=np.hstack([Aeq, np.zeros((Aeq.shape[0], 1))]),
            b_eq=beq,
            bounds=list(zip(self.lp["lb"], self.lp["ub"])),
            method='highs'
        )
        if not res.success:
            print("[LP] Solver failed.")
            return None
        lambda_full = res.x[:-1]
        if self.apply_selective_mapping:
            Fodar = self.B_blkdiag @ lambda_full
            lambda_full = self._adjust_thrust_across_links(Fodar)
        self.lambda_set_prev = lambda_full
        return lambda_full

    def _adjust_thrust_across_links(self, Fodar):
        thrusts = []
        for i in range(self.nlinks):
            Fi = Fodar[i*6:(i+1)*6]
            wrench = self._convert_matrix_to_wrench(Fi)
            thrusts.append(self.smaps[i].get_adjusted_thrust(wrench))
        return np.concatenate(thrusts)

    @staticmethod
    def _convert_matrix_to_wrench(F):
        return {
            "torque": F[:3],
            "force": F[3:]
        }
