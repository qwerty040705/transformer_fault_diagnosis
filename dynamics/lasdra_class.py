# /home/cdb/transformer_fault_diagnosis/dynamics/lasdra_class.py

import numpy as np
from scipy.linalg import block_diag, sqrtm, solve_sylvester
from project_math.Adinv import Adinv
from project_math.adj import adj
from project_math.Tinv import Tinv
from .forward_kinematics_class import ForwardKinematics

class LASDRA:
    def __init__(self, params):
        self.ODAR = params['ODAR']
        self.dof = params['LASDRA']['dof']
        self.numlinks = len(self.ODAR)

        self.q = np.zeros((self.dof, 1))
        self.dq = np.zeros((self.dof, 1))
        self.dV0 = np.array([[0], [0], [0], [0], [0], [9.81]])

        self.L = np.eye(6 * self.dof)
        self.Fodar = np.zeros((6 * self.numlinks, 1))

        self.init_rotor_mapping(params)
        self.init_body_joint_screw_axes(params['LASDRA']['body_joint_screw_axes'])
        self.init_body_inertia_tensor(params['LASDRA']['inertia_matrix'])
        self.init_link_com_transforms(params)
        self.fk = ForwardKinematics(params)

        self.update_dynamics()

    def init_rotor_mapping(self, params):
        eye_perm = np.block([[np.zeros((3, 3)), np.eye(3)], [np.eye(3), np.zeros((3, 3))]])
        self.B_cell = []
        for odar in params['ODAR']:
            self.B_cell.append(eye_perm @ odar['B'])
        self.B_blkdiag = block_diag(*self.B_cell)

    def init_body_joint_screw_axes(self, A_cell):
        self.A = block_diag(*A_cell)

    def init_body_inertia_tensor(self, G_cell):
        self.G = block_diag(*G_cell)

    def init_link_com_transforms(self, params):
        self.AdLinkCoMTransforms = []
        for iodar, odar in enumerate(params['ODAR']):
            for _ in range(len(odar['body_joint_screw_axes']) - 1):
                self.AdLinkCoMTransforms.append(Adinv(np.eye(4)))
            Tic = np.eye(4)
            Tic[:3, 3] = odar['joint_to_com']
            self.AdLinkCoMTransforms.append(Adinv(Tic))

        Tne = np.eye(4)
        Tne[:3, 3] = params['ODAR'][-1]['length'] * np.array([1, 0, 0])
        self.AdEndEffectorTransform = self.AdLinkCoMTransforms[-1]
        Tn = np.eye(4)
        Tn[:3, 3] = params['ODAR'][-1]['joint_to_com']
        self.Tnce = Tinv(Tn) @ Tne

        # 축소 (index 계산)
        joint_counts = [len(odar['body_joint_screw_axes']) for odar in self.ODAR]
        row_indices = []
        count = 0
        for joints in joint_counts:
            count += joints
            row_indices.extend(list(range(6 * (count - 1), 6 * count)))
        self.AdLinkCoMTransforms = np.vstack([self.AdLinkCoMTransforms[i] for i in row_indices])

    def set_joint_states(self, q, dq):
        self.q = q
        self.dq = dq
        self.update_dynamics()

    def update_dynamics(self):
        Tijset = self.fk.compute_inter_joint_transition(self.q)
        self.L = np.eye(6 * self.dof)
        for i in range(1, self.dof):
            T = np.eye(4)
            for j in range(i - 1, -1, -1):
                T = Tijset[j + 1] @ T
                self.L[6 * i:6 * (i + 1), 6 * j:6 * (j + 1)] = Adinv(T)

        V = self.L @ self.A @ self.dq
        self.ad_V = block_diag(*[adj(V[6*i:6*(i+1)]) for i in range(self.dof)])

        dV_base = np.zeros((6 * self.dof, 1))
        dV_base[:6] = Adinv(Tijset[0]) @ self.dV0

        self.Mass = self.A.T @ self.L.T @ self.G @ self.L @ self.A
        self.Cori = self.A.T @ self.L.T @ (self.G @ self.L @ self.ad_V - self.ad_V.T @ self.G @ self.L) @ self.A
        self.Grav = self.A.T @ self.L.T @ self.G @ self.L @ dV_base
        self.D = self.AdLinkCoMTransforms @ self.L @ self.A

    def get_joint_torque_from_odars(self):
        return self.D.T @ self.Fodar

    def set_odar_body_wrenches(self, F):
        self.Fodar = F

    def set_odar_body_wrench_from_thrust(self, thrust, iodar):
        Fi = self.B_cell[iodar] @ thrust
        self.Fodar[6*iodar:6*(iodar+1)] = Fi

    def get_next_joint_states(self, dt, tau_odar, external_wrench=None):
        if external_wrench is None:
            external_wrench_b = np.zeros((6, 1))
            tau_net = tau_odar
        else:
            R_EF = self.fk.compute_end_effector_frame(self.q)[:3, :3]
            blkR = block_diag(R_EF, R_EF)
            external_wrench_b = blkR.T @ np.vstack((external_wrench['torque'], external_wrench['force']))
            tau_net = tau_odar + self.fk.compute_end_effector_body_jacobian(self.q).T @ external_wrench_b

        ddq = np.linalg.solve(self.Mass, tau_net - self.Cori @ self.dq - self.Grav)
        return self.passive_midpoint_integration(tau_net, ddq, dt)

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
