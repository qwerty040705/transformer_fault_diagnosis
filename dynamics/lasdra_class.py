import numpy as np
from scipy.linalg import block_diag, sqrtm, solve_sylvester
from project_math.Adinv import Adinv
from project_math.adj import adj
from project_math.Tinv import Tinv
from .forward_kinematics_class import ForwardKinematics

class LASDRA:
    def __init__(self, params):
        """
        params['ODAR']: list of ODAR objects (속성 접근: odar.B, odar.length, odar.joint_to_com, odar.body_joint_screw_axes)
        params['LASDRA']['body_joint_screw_axes']: list of (6,) or ndarray (6, dof)
        params['LASDRA']['inertia_matrix']: list of 6x6, block_diag로 쌓음
        """
        self.ODAR = params['ODAR']
        self.dof = params['LASDRA']['dof']
        self.numlinks = len(self.ODAR)

        self.q = np.zeros((self.dof, 1))
        self.dq = np.zeros((self.dof, 1))
        # 중력 가속도 (필요시 축/부호 확인)
        self.dV0 = np.array([[0], [0], [0], [0], [0], [9.81]])

        self.L = np.eye(6 * self.dof)
        self.Fodar = np.zeros((6 * self.numlinks, 1))

        self.init_rotor_mapping(params)
        self.init_body_joint_screw_axes(params['LASDRA']['body_joint_screw_axes'])
        self.init_body_inertia_tensor(params['LASDRA']['inertia_matrix'])
        self.init_link_com_transforms(params)

        self.fk = ForwardKinematics(params)
        self.update_dynamics()

    # ----------------------------
    # 초기화 루틴
    # ----------------------------
    def init_rotor_mapping(self, params):
        # 퍼뮤테이션(회전/힘 ↔ 토크/선력) 매핑
        eye_perm = np.block([[np.zeros((3, 3)), np.eye(3)],
                             [np.eye(3), np.zeros((3, 3))]])
        self.B_cell = []
        for odar in params['ODAR']:
            self.B_cell.append(eye_perm @ odar.B)
        self.B_blkdiag = block_diag(*self.B_cell)

    def init_body_joint_screw_axes(self, A_cell):
        """
        S를 (6, dof)로 정규화하고,
        A를 (6*dof, dof)로 구성: 각 열 j가 블록 [6*j:6*(j+1)] 위치에 S[:, j]를 둠.
        """
        self.S = self._to_6_by_dof(A_cell)         # (6, dof)
        self.A = np.zeros((6 * self.dof, self.dof))  # (6d, d)
        for j in range(self.dof):
            self.A[6*j:6*(j+1), j] = self.S[:, j]

    @staticmethod
    def _to_6_by_dof(A_cell):
        # ndarray 케이스
        if isinstance(A_cell, np.ndarray):
            if A_cell.ndim == 2:
                if A_cell.shape[0] == 6:
                    return A_cell  # (6, dof)
                if A_cell.shape[1] == 6:
                    return A_cell.T  # (dof, 6) -> (6, dof)
            if A_cell.ndim == 1 and (A_cell.size % 6 == 0):
                dof = A_cell.size // 6
                return A_cell.reshape(dof, 6).T
            raise ValueError(f"[LASDRA] Unsupported ndarray shape for screw axes: {A_cell.shape}")

        # list/tuple 케이스
        if isinstance(A_cell, (list, tuple)):
            cols = []
            for a in A_cell:
                arr = np.asarray(a).reshape(-1)
                if arr.size != 6:
                    raise ValueError(f"[LASDRA] Each screw axis must be 6x1; got shape {arr.shape}")
                cols.append(arr)
            return np.stack(cols, axis=1)
        raise TypeError(f"[LASDRA] body_joint_screw_axes must be ndarray or list; got {type(A_cell)}")

    def init_body_inertia_tensor(self, G_cell):
        self.G = block_diag(*G_cell)  # (6d, 6d)

    def init_link_com_transforms(self, params):
        """
        1) 모든 '조인트'에 대한 6x6 변환을 대각 블록으로 구성: AdAll ∈ ℝ^{6d×6d}
           - 각 링크에서 마지막 조인트만 Adinv(Tic), 나머지는 I (조인트 프레임 ↔ 링크 CoM)
        2) 링크별 '마지막 조인트'만 추출하는 선택 행렬 S_link ∈ ℝ^{6L×6d}
        3) AdLinkCoMTransforms = S_link @ AdAll ∈ ℝ^{6L×6d}
        """
        transforms_per_joint = []  # 길이 = dof, 각 원소 6x6
        joint_counts = []

        for odar in params['ODAR']:
            nj = len(odar.body_joint_screw_axes)
            joint_counts.append(nj)
            # 링크 내 조인트들: 마지막만 CoM 변환, 나머지는 항등
            for _ in range(nj - 1):
                transforms_per_joint.append(Adinv(np.eye(4)))  # 6x6
            Tic = np.eye(4)
            Tic[:3, 3] = odar.joint_to_com
            transforms_per_joint.append(Adinv(Tic))  # 6x6

        # 1) 모든 조인트용 6x6 변환 대각 블록
        AdAll = block_diag(*transforms_per_joint)  # (6d, 6d)

        # 2) 링크별 마지막 조인트 블록만 선택하는 S_link
        d = self.dof
        L = self.numlinks
        S_link = np.zeros((6 * L, 6 * d))
        acc = 0
        for i, nj in enumerate(joint_counts):
            last_j = acc + nj - 1  # 이 링크의 마지막 조인트 인덱스
            # 행: 6*i : 6*(i+1), 열: 6*last_j : 6*(last_j+1) 에 I6 배치
            S_link[6*i:6*(i+1), 6*last_j:6*(last_j+1)] = np.eye(6)
            acc += nj

        # 3) 최종 CoM 변환 (6L, 6d)
        self.AdLinkCoMTransforms = S_link @ AdAll

        # (참고) 말단 이펙터 상대변환 (필요 시 사용)
        Tne = np.eye(4)
        Tne[:3, 3] = params['ODAR'][-1].length * np.array([1, 0, 0])
        Tn = np.eye(4)
        Tn[:3, 3] = params['ODAR'][-1].joint_to_com
        self.Tnce = Tinv(Tn) @ Tne

    # ----------------------------
    # 상태/동역학 업데이트
    # ----------------------------
    def set_joint_states(self, q, dq):
        self.q = q
        self.dq = dq
        self.update_dynamics()

    def update_dynamics(self):
        """
        V = L(q) @ A @ dq
        ad_V = block_diag(adj(V_i))
        M = Aᵀ Lᵀ G L A
        C = Aᵀ Lᵀ ( G L ad_V - ad_Vᵀ G L ) A
        g = Aᵀ Lᵀ G L dV_base
        D = AdCoM @ L @ A
        """
        Tijset = self.fk.compute_inter_joint_transition(self.q)

        # L(q) ∈ ℝ^{6d×6d}
        self.L = np.eye(6 * self.dof)
        for i in range(1, self.dof):
            T = np.eye(4)
            for j in range(i - 1, -1, -1):
                T = Tijset[j + 1] @ T
                self.L[6 * i:6 * (i + 1), 6 * j:6 * (j + 1)] = Adinv(T)

        # V ∈ ℝ^{6d×1}
        V = self.L @ self.A @ self.dq

        # ad_V ∈ ℝ^{6d×6d}
        self.ad_V = block_diag(*[adj(V[6*i:6*(i+1)]) for i in range(self.dof)])

        # 중력항 준비
        dV_base = np.zeros((6 * self.dof, 1))
        dV_base[:6] = Adinv(Tijset[0]) @ self.dV0

        # Mass, Cori, Grav
        self.Mass = self.A.T @ self.L.T @ self.G @ self.L @ self.A
        self.Cori = self.A.T @ self.L.T @ (self.G @ self.L @ self.ad_V - self.ad_V.T @ self.G @ self.L) @ self.A
        self.Grav = self.A.T @ self.L.T @ self.G @ self.L @ dV_base

        # D ∈ ℝ^{6L×d}
        self.D = self.AdLinkCoMTransforms @ self.L @ self.A

    # ----------------------------
    # 외력/추력/적분
    # ----------------------------
    def get_joint_torque_from_odars(self):
        """ τ = Dᵀ F_odar """
        return self.D.T @ self.Fodar

    def set_odar_body_wrenches(self, F):
        """ F ∈ ℝ^{6L×1} """
        self.Fodar = F

    def set_odar_body_wrench_from_thrust(self, thrust, iodar):
        """
        thrust: (m_i, 1) (예: 8x1)
        Fi = B_i @ thrust  → Fi ∈ ℝ^{6×1}
        """
        Fi = self.B_cell[iodar] @ thrust
        self.Fodar[6*iodar:6*(iodar+1)] = Fi

    def get_next_joint_states(self, dt, tau_odar, external_wrench=None):
        self.step_count = getattr(self, "step_count", -1) + 1
        """
        tau_odar: (dof, 1)
        기본 적분기: semi-implicit Euler (빠르고 안정적)
        """
        if external_wrench is None:
            tau_net = tau_odar
        else:
            R_EF = self.fk.compute_end_effector_frame(self.q)[:3, :3]
            blkR = block_diag(R_EF, R_EF)
            external_wrench_b = blkR.T @ np.vstack((external_wrench['torque'], external_wrench['force']))
            tau_net = tau_odar + self.fk.compute_end_effector_body_jacobian(self.q).T @ external_wrench_b

            if np.any(np.isnan(tau_net - self.Cori @ self.dq - self.Grav)) or np.any(np.isinf(tau_net - self.Cori @ self.dq - self.Grav)):
                print(f"[❌] RHS (tau - Cori*dq - Grav) has NaN or Inf at step {self.step_count}")
            if np.any(np.isnan(self.Mass)) or np.any(np.isinf(self.Mass)):
                print(f"[❌] Mass matrix has NaN or Inf at step {self.step_count}")



        # M(q) ddq + C(q,dq)dq + g(q) = tau_net  →  ddq = M^{-1}(tau_net - C dq - g)
        ddq = np.linalg.solve(self.Mass, tau_net - self.Cori @ self.dq - self.Grav)

        # ---- 빠른 적분기: Semi-Implicit (Symplectic) Euler ----
        dq_next = self.dq + dt * ddq
        q_next = self.q + dt * dq_next
        return {'q': q_next, 'dq': dq_next}



    # 만약 예전의 수치적 중점 적분을 쓰고 싶다면(매우 느림):
    # return self.passive_midpoint_integration(tau_net, ddq, dt)


    def passive_midpoint_integration(self, tau_net, ddq, dt):
        sMk = sqrtm(self.Mass)
        dsMk = solve_sylvester(sMk, sMk, self.Cori + self.Cori.T)
        A_bar = self.Mass / dt + 0.5 * (self.Cori - sMk @ dsMk)
        B_bar = tau_net - self.Grav + 2 * self.Mass @ self.dq / dt
        inv_A_B = np.linalg.solve(A_bar, B_bar)

        q_next = self.q + 0.5 * dt * inv_A_B
        sMkp = sqrtm(self.get_mass_matrix(q_next))
        dq_next = np.linalg.solve(sMkp, sMk @ (inv_A_B - self.dq))
        return {'q': q_next, 'dq': dq_next}

    def get_mass_matrix(self, q):
        Tijset = self.fk.compute_inter_joint_transition(q)
        L = np.eye(6 * self.dof)
        for i in range(1, self.dof):
            T = np.eye(4)
            for j in range(i - 1, -1, -1):
                T = Tijset[j + 1] @ T
                L[6 * i:6 * (i + 1), 6 * j:6 * (j + 1)] = Adinv(T)
        return self.A.T @ L.T @ self.G @ L @ self.A
