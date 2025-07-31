# /home/cdb/transformer_fault_diagnosis/dynamics/forward_kinematics_class.py

import numpy as np
from scipy.linalg import block_diag
from project_math.Adinv import Adinv

class ForwardKinematics:
    """
    Forward kinematics utilities:
      - compute_inter_joint_transition(q): 각 조인트 i의 변환 Ti = exp([ξ_i] θ_i) (4x4) 리스트 (길이 dof)
      - compute_end_effector_frame(q): 전체 체인 끝단(말단 조인트 이후 링크 끝) T_EF (4x4)
      - compute_CoM_frame(q): 각 링크의 CoM 프레임 리스트
      - compute_GC_frame(q): 각 링크의 절반 길이 지점(기하학적 중심) 프레임 리스트
      - compute_body_jacobian(q): (T_EF, J_b), Body Jacobian (6 x dof)
      - compute_end_effector_body_jacobian(q): J_b 만 반환
    """

    def __init__(self, param):
        # ODAR 목록 (속성 접근자 사용)
        self.ODAR = param['ODAR']
        self.dof = param['LASDRA']['dof']

        # Aset: screw axes (6, dof) 형식으로 보장
        Aset = param['LASDRA']['body_joint_screw_axes']
        self.Aset = self._to_6_by_dof(Aset)  # shape (6, dof)

        # 조인트 사이 고정 변환(링크 길이/CoM 등)
        self.T_joint_to_com_set = []
        self.T_joint_to_gc_set = []
        self.T_joint_to_joint_set = []
        self._init_fixed_transforms(self.ODAR)

        # 누적 조인트 프레임 저장용
        self.Tjset = [np.eye(4) for _ in range(self.dof)]

    # ----------------------------
    # 유틸: 입력 스크류를 (6, dof)로 보장
    # ----------------------------
    @staticmethod
    def _to_6_by_dof(A_cell):
        A = np.asarray(A_cell)
        if A.ndim == 2:
            if A.shape[0] == 6:
                return A
            if A.shape[1] == 6:
                return A.T
        if A.ndim == 1 and (A.size % 6 == 0):
            dof = A.size // 6
            return A.reshape(dof, 6).T
        # list/tuple 케이스
        if isinstance(A_cell, (list, tuple)):
            cols = []
            for a in A_cell:
                arr = np.asarray(a).reshape(-1)
                if arr.size != 6:
                    raise ValueError(f"[FK] Each screw axis must be 6x1; got shape {arr.shape}")
                cols.append(arr)
            return np.stack(cols, axis=1)
        raise ValueError(f"[FK] Unsupported screw axes shape: {A_cell if isinstance(A_cell, (list, tuple)) else A.shape}")

    # ----------------------------
    # 고정 변환 초기화
    # ----------------------------
    def _init_fixed_transforms(self, odar_array):
        """
        - T_joint_to_com_set[i]: 링크 i의 '마지막 조인트' 프레임에서 CoM까지의 변환
        - T_joint_to_gc_set[i]: 링크 i의 '마지막 조인트' 프레임에서 링크 길이/2 지점까지
        - T_joint_to_joint_set[i]: 링크 i의 '마지막 조인트' 프레임에서 다음 링크의 첫 조인트 프레임까지 (즉, 링크 길이만큼 x축 이동)
        """
        for odar in odar_array:
            # CoM
            Tjc = np.eye(4)
            Tjc[:3, 3] = odar.joint_to_com.reshape(3,)
            self.T_joint_to_com_set.append(Tjc)

            # 절반 길이 지점
            Tjg = np.eye(4)
            Tjg[:3, 3] = 0.5 * odar.length * np.array([1.0, 0.0, 0.0])
            self.T_joint_to_gc_set.append(Tjg)

            # 다음 링크 첫 조인트까지 (링크 길이)
            Tjj = np.eye(4)
            Tjj[:3, 3] = odar.length * np.array([1.0, 0.0, 0.0])
            self.T_joint_to_joint_set.append(Tjj)

    # ----------------------------
    # se(3) 지수맵 유틸
    # ----------------------------
    @staticmethod
    def _skew(w):
        return np.array([[0, -w[2], w[1]],
                         [w[2], 0, -w[0]],
                         [-w[1], w[0], 0]])

    @classmethod
    def _exp_se3(cls, xi, theta):
        """
        xi = [w; v] ∈ R^6, theta ∈ R
        exp([xi] theta) = [[R, p],[0,0,0,1]]
          - R = exp([w] theta)
          - p = (I*theta + (1-cosθ)[w] + (θ - sinθ)[w]^2) v  (revolute)
          - if ||w||≈0 (prismatic): R=I, p=v*theta
        """
        xi = np.asarray(xi).reshape(6,)
        w, v = xi[:3], xi[3:]
        th = float(theta)

        if np.linalg.norm(w) < 1e-12:  # prismatic
            R = np.eye(3)
            p = v * th
        else:
            w_hat = cls._skew(w)
            th2 = th * th
            R = np.eye(3) + np.sin(th) * w_hat + (1 - np.cos(th)) * (w_hat @ w_hat)
            V = np.eye(3) * th + (1 - np.cos(th)) * w_hat + (th - np.sin(th)) * (w_hat @ w_hat)
            p = V @ v

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        return T

    # ----------------------------
    # FK 본체
    # ----------------------------
    def compute_inter_joint_transition(self, q):
        """
        각 조인트 i의 4x4 변환 Ti = exp([ξ_i] θ_i) 리스트를 반환 (길이 dof).
        첫 링크가 아니고, 해당 링크의 첫 조인트라면 이전 링크 끝 → 현재 링크 첫 조인트의
        고정 변환(길이 이동)을 선곱한다.
        """
        q_flat = np.asarray(q).reshape(-1)  # (dof,)
        Tij = [None] * self.dof

        jointcnt = 0
        for iodar, odar in enumerate(self.ODAR):
            nj = len(odar.body_joint_screw_axes)
            for ijoint in range(nj):
                theta = float(q_flat[jointcnt])
                xi = self.Aset[:, jointcnt]  # (6,)
                Texp = self._exp_se3(xi, theta)

                # 이전 링크 끝에서 현재 링크 첫 조인트까지의 이동을 선곱
                if iodar != 0 and ijoint == 0:
                    Texp = self.T_joint_to_joint_set[iodar - 1] @ Texp

                Tij[jointcnt] = Texp
                jointcnt += 1

        return Tij

    def compute_CoM_frame(self, q):
        """
        각 링크의 마지막 조인트 프레임 기준 CoM 프레임을 월드 기준으로 반환 (리스트 길이 = 링크 수)
        """
        Tijset = self.compute_inter_joint_transition(q)

        # 누적
        self.Tjset[0] = Tijset[0]
        for i in range(1, self.dof):
            self.Tjset[i] = self.Tjset[i-1] @ Tijset[i]

        T_list = []
        jointcnt = 0
        for iodar in range(len(self.ODAR)):
            nj = len(self.ODAR[iodar].body_joint_screw_axes)
            last_joint_idx = jointcnt + nj - 1
            T_link_last = self.Tjset[last_joint_idx]
            T_list.append(T_link_last @ self.T_joint_to_com_set[iodar])
            jointcnt += nj
        return T_list

    def compute_GC_frame(self, q):
        """
        각 링크의 마지막 조인트 프레임 기준 '절반 길이' 지점 프레임을 월드 기준으로 반환
        """
        Tijset = self.compute_inter_joint_transition(q)

        self.Tjset[0] = Tijset[0]
        for i in range(1, self.dof):
            self.Tjset[i] = self.Tjset[i-1] @ Tijset[i]

        T_list = []
        jointcnt = 0
        for iodar in range(len(self.ODAR)):
            nj = len(self.ODAR[iodar].body_joint_screw_axes)
            last_joint_idx = jointcnt + nj - 1
            T_link_last = self.Tjset[last_joint_idx]
            T_list.append(T_link_last @ self.T_joint_to_gc_set[iodar])
            jointcnt += nj
        return T_list

    def compute_end_effector_frame(self, q):
        """
        말단 이펙터(= 마지막 링크 끝) 프레임을 반환
        """
        Tij = self.compute_inter_joint_transition(q)
        T_EF = np.eye(4)
        for i in range(self.dof):
            T_EF = T_EF @ Tij[i]
        # 마지막 링크의 끝까지 이동
        T_EF = T_EF @ self.T_joint_to_joint_set[-1]
        return T_EF

    # ----------------------------
    # Jacobian
    # ----------------------------
    def compute_body_jacobian(self, q):
        """
        Body Jacobian (6 x dof) 과 현재 EF pose를 반환.
        역전파 공식:
          T_ie ← (마지막 링크 끝)부터 시작
          for i = dof-1 .. 0:
              J_b[:, i] = Adinv(T_ie) @ ξ_i
              T_ie = exp([ξ_i] θ_i) @ T_ie
        """
        q_flat = np.asarray(q).reshape(-1)
        d = self.dof

        # 말단 링크 끝 변환(조인트-조인트 고정 변환)
        T_ie = self.T_joint_to_joint_set[-1].copy()

        J_b = np.zeros((6, d))
        for i in reversed(range(d)):
            xi = self.Aset[:, i]
            J_b[:, i] = (Adinv(T_ie) @ xi.reshape(6, 1)).reshape(6,)
            # 누적
            T_ie = self._exp_se3(xi, float(q_flat[i])) @ T_ie

        T_EF = T_ie  # base→EF 변환
        return T_EF, J_b

    def compute_end_effector_body_jacobian(self, q):
        _, J_b = self.compute_body_jacobian(q)
        return J_b
