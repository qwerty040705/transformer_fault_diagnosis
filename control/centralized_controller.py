import numpy as np

class CentralizedController:
    def __init__(self, param, lasdra_model):
        self.lasdra_model = lasdra_model

        pid = param["LASDRA"]["multijoint_PID"]
        self.Kp = pid["p"]
        self.Ki = pid["i"]
        self.Kd = pid["d"]
        self.Kp_s = pid["simplified"]["p"]
        self.Ki_s = pid["simplified"]["i"]
        self.Kd_s = pid["simplified"]["d"]
        self.q_e_i = np.zeros(param["LASDRA"]["dof"])
        self.q_e_i_max = pid["i_max"]
        self.ddq_d = np.zeros(param["LASDRA"]["dof"])
        self.max_acc_feedforward_ratio = pid["max_acc_feedforward_ratio"]

        nominal = pid["torque_limit"]
        self.torque_limit = {
            "nominal": nominal,
            "ub": nominal * np.ones(param["LASDRA"]["dof"]),
            "lb": -nominal * np.ones(param["LASDRA"]["dof"]),
        }

        self.q_prev = None
        self.dq_prev = None
        self.num_diff_damp_coeff = 0.97

    def set_desired_joint_angle(self, q_d):
        self.q_d = q_d

    def set_desired_joint_velocity(self, dq_d):
        self.dq_d = dq_d

    def set_desired_state(self, joint_state):
        self.set_desired_joint_angle(joint_state["q"])
        self.set_desired_joint_velocity(joint_state["dq"])

    def set_desired_acceleration(self, ddq_d):
        self.ddq_d = ddq_d

    def calculate_velocity(self, dt, q):
        if self.q_prev is None:
            dq = np.zeros_like(q)
        else:
            dq = (q - self.q_prev) / dt
            dq = self.get_damped_value(self.dq_prev, dq, self.num_diff_damp_coeff)
        self.q_prev = q
        self.dq_prev = dq
        return dq

    def get_control_output(self, dt, q, dq=None):
        if dq is None:
            dq = self.calculate_velocity(dt, q)

        q_e = q - self.q_d
        dq_e = dq - self.dq_d

        self.q_e_i += q_e * dt
        self.q_e_i = self.bound(self.q_e_i, -self.q_e_i_max, self.q_e_i_max)

        M = self.lasdra_model.Mass
        C = self.lasdra_model.Cori
        G = self.lasdra_model.Grav

        acc_ff = M @ self.ddq_d
        acc_ff_max = np.max(np.abs(acc_ff))
        if acc_ff_max / self.torque_limit["nominal"] > self.max_acc_feedforward_ratio:
            acc_ff *= self.max_acc_feedforward_ratio / acc_ff_max

        torque = acc_ff + M @ (-self.Kp * q_e - self.Ki * self.q_e_i - self.Kd * dq_e) + C @ dq + G
        torque = self.bound(torque, self.torque_limit["lb"], self.torque_limit["ub"])
        return torque

    def get_simplified_control_output(self, dt, q, dq=None):
        if dq is None:
            dq = self.calculate_velocity(dt, q)

        q_e = q - self.q_d
        dq_e = dq - self.dq_d
        self.q_e_i += q_e * dt
        self.q_e_i = self.bound(self.q_e_i, -self.q_e_i_max, self.q_e_i_max)

        G = self.lasdra_model.Grav
        torque = -self.Kp_s * q_e - self.Ki_s * self.q_e_i - self.Kd_s * dq_e + G
        torque = self.bound(torque, self.torque_limit["lb"], self.torque_limit["ub"])
        return torque

    def get_integral_control_output(self, dt, q, dq=None):
        if dq is None:
            dq = self.calculate_velocity(dt, q)

        q_e = q - self.q_d
        self.q_e_i += q_e * dt
        self.q_e_i = self.bound(self.q_e_i, -self.q_e_i_max, self.q_e_i_max)

        M = self.lasdra_model.Mass
        torque = M @ (-self.Kp * q_e - self.Ki * self.q_e_i)
        torque = self.bound(torque, self.torque_limit["lb"], self.torque_limit["ub"])
        return torque

    def get_damped_value(self, v_prev, v, damping):
        return (1 - damping) * v_prev + damping * v

    def bound(self, x, x_min, x_max):
        return np.maximum(x_min, np.minimum(x_max, x))

    def set_initial_state(self, joint_state):
        self.q_prev = joint_state["q"]
        self.dq_prev = joint_state["dq"]
