import numpy as np

class SelectiveMapping:
    def __init__(self, odar):
        self.B = odar.B  # shape (6, 8)
        self.Bnsv = odar.Bnsv  # dict with 'upper' and 'lower'
        self.eps0 = None  # for future use
        self.eps1 = None  # for future use

    def get_adjusted_thrust(self, wrench_d):
        # wrench_d: dict with 'force' and 'torque' as 3D numpy arrays
        F = self.convert_wrench_to_matrix(wrench_d)
        lambda_ = np.linalg.pinv(self.B) @ F

        lambda_alpha = self.optimize_in_null_space(lambda_)
        lambda_beta = self.avoid_dead_zones(lambda_alpha)
        lambda_gamma = self.smooth_set_points(lambda_beta)

        return lambda_gamma

    def optimize_in_null_space(self, lambda_):
        # upper half
        upper_prod = lambda_[:4] * self.Bnsv['upper'][:4]
        adjust_factor = 0.5 * (np.max(upper_prod) + np.min(upper_prod))
        lambda_ = lambda_ - adjust_factor * self.Bnsv['upper']

        # lower half
        lower_prod = lambda_[4:] * self.Bnsv['upper'][4:]
        k_lower = 0.5 * (np.max(lower_prod) + np.min(lower_prod))
        lambda_ = lambda_ - k_lower * self.Bnsv['lower']

        return lambda_

    def avoid_dead_zones(self, lambda_):
        # TODO: implement algorithm based on self.eps0
        return lambda_

    def smooth_set_points(self, lambda_):
        # TODO: implement algorithm based on self.eps1
        return lambda_

    @staticmethod
    def convert_wrench_to_matrix(wrench):
        return np.concatenate((wrench['force'], wrench['torque']), axis=0)
