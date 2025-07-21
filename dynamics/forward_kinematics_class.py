import numpy as np
from scipy.linalg import block_diag
from project_math.Adinv import Adinv

class ForwardKinematics:
    def __init__(self, param):
        self.ODAR = param['ODAR']
        self.dof = param['LASDRA']['dof']
        self.Tjset = [None] * self.dof
        self.Aset = param['LASDRA']['body_joint_screw_axes']
        self.T_joint_to_com_set = []
        self.T_joint_to_gc_set = []
        self.T_joint_to_joint_set = []
        self.initialize_rotation_axis_functions(self.ODAR)
        self.initialize_link_forward_kinematics(self.ODAR)

    def initialize_rotation_axis_functions(self, odar_array):
        self.RotationAxisFunctions = []
        for odar in odar_array:
            joint_funcs = []
            for S in odar['body_joint_screw_axes']:
                if S[0] == 1:
                    joint_funcs.append(self.rotx)
                elif S[1] == 1:
                    joint_funcs.append(self.roty)
                elif S[2] == 1:
                    joint_funcs.append(self.rotz)
                else:
                    joint_funcs.append(lambda theta: np.eye(3))
            self.RotationAxisFunctions.append(joint_funcs)

    def initialize_link_forward_kinematics(self, odar_array):
        for odar in odar_array:
            self.T_joint_to_com_set.append(np.vstack((np.hstack((np.eye(3), odar['joint_to_com'].reshape(3,1))), [0,0,0,1])))
            self.T_joint_to_gc_set.append(np.vstack((np.hstack((np.eye(3), 0.5 * odar['length'] * np.array([[1],[0],[0]]))), [0,0,0,1])))
            self.T_joint_to_joint_set.append(np.vstack((np.hstack((np.eye(3), odar['length'] * np.array([[1],[0],[0]]))), [0,0,0,1])))

    def compute_inter_joint_transition(self, q):
        Tij = [None] * self.dof
        jointcnt = 0
        for iodar, odar in enumerate(self.ODAR):
            for ijoint, S in enumerate(odar['body_joint_screw_axes']):
                theta = q[jointcnt]
                R = self.RotationAxisFunctions[iodar][ijoint](theta)
                Ti = np.eye(4)
                Ti[:3, :3] = R
                if iodar != 0 and ijoint == 0:
                    Ti = self.T_joint_to_joint_set[iodar - 1] @ Ti
                Tij[jointcnt] = Ti
                jointcnt += 1
        return Tij

    def compute_CoM_frame(self, q):
        Tijset = self.compute_inter_joint_transition(q)
        self.Tjset[0] = Tijset[0]
        for i in range(1, self.dof):
            self.Tjset[i] = self.Tjset[i-1] @ Tijset[i]
        T = []
        jointcnt = 0
        for iodar in range(len(self.ODAR)):
            jointcnt += len(self.ODAR[iodar]['body_joint_screw_axes'])
            T.append(self.Tjset[jointcnt - 1] @ self.T_joint_to_com_set[iodar])
        return T

    def compute_GC_frame(self, q):
        Tijset = self.compute_inter_joint_transition(q)
        self.Tjset[0] = Tijset[0]
        for i in range(1, self.dof):
            self.Tjset[i] = self.Tjset[i-1] @ Tijset[i]
        T = []
        jointcnt = 0
        for iodar in range(len(self.ODAR)):
            jointcnt += len(self.ODAR[iodar]['body_joint_screw_axes'])
            T.append(self.Tjset[jointcnt - 1] @ self.T_joint_to_gc_set[iodar])
        return T

    def compute_end_effector_frame(self, q):
        Tij = self.compute_inter_joint_transition(q)
        T_EF = np.eye(4)
        for i in range(self.dof):
            T_EF = T_EF @ Tij[i]
        T_EF = T_EF @ self.T_joint_to_joint_set[-1]
        return T_EF

    def compute_body_jacobian(self, q):
        Tij = self.compute_inter_joint_transition(q)
        T_ie = self.T_joint_to_joint_set[-1].copy()
        J_b = np.zeros((6, self.dof))
        for i in reversed(range(self.dof)):
            J_b[:, i] = Adinv(T_ie) @ self.Aset[i]
            T_ie = Tij[i] @ T_ie
        T_EF = T_ie
        return T_EF, J_b

    def compute_end_effector_body_jacobian(self, q):
        _, J_b = self.compute_body_jacobian(q)
        return J_b

    def compute_end_effector_analytic_jacobian(self, q):
        T_EF, J_b = self.compute_body_jacobian(q)
        R = T_EF[:3, :3]
        J_a = block_diag(R, R) @ J_b
        return J_a

    def rotx(self, theta):
        return np.array([[1, 0, 0],
                         [0, np.cos(theta), -np.sin(theta)],
                         [0, np.sin(theta), np.cos(theta)]])

    def roty(self, theta):
        return np.array([[np.cos(theta), 0, np.sin(theta)],
                         [0, 1, 0],
                         [-np.sin(theta), 0, np.cos(theta)]])

    def rotz(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta),  np.cos(theta), 0],
                         [0, 0, 1]])
