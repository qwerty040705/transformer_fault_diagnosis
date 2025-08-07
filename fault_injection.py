import numpy as np
from typing import Tuple, Union

def _sample_k(rng: np.random.Generator, lam: float, max_faults: int) -> int:
    if max_faults <= 0:
        return 0
    for _ in range(100):
        k = rng.poisson(lam=lam)
        if 1 <= k <= max_faults:
            return k
    k = int(round(lam))
    return min(max_faults, max(1, k))

def inject_faults(
    lambda_arr: np.ndarray,
    fault_time: Union[int, float, None] = None,
    epsilon_scale: float = 0.01,
    fault_lambda: float = 1.0,
    seed: Union[int, None] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(lambda_arr)
    eps = 1e-9  # 작은 수

    if arr.ndim == 1:
        M = arr.shape[0]
        if M % 8 != 0:
            raise ValueError(f"[inject_faults] length {M} is not a multiple of 8.")
        N = M // 8
        type_matrix = np.ones((N, 8), dtype=int)

        k = _sample_k(rng, fault_lambda, max_faults=N)
        fault_indices = rng.choice(M, size=k, replace=False)

        faulty = arr.copy()
        for j in range(M):
            ref_mag = abs(arr[j])
            if ref_mag < eps:
                ref_mag = 1.0
            noise = rng.normal(loc=0.0, scale=epsilon_scale * ref_mag)

            if j in fault_indices:
                faulty[j] = 0.0 + noise
                link_idx, motor_idx = divmod(j, 8)
                type_matrix[link_idx, motor_idx] = 0
            else:
                faulty[j] = arr[j] + noise

        return faulty, type_matrix

    elif arr.ndim == 2:
        T, M = arr.shape
        if M % 8 != 0:
            raise ValueError(f"[inject_faults] width {M} is not a multiple of 8.")
        N = M // 8
        type_matrix = np.ones((N, 8), dtype=int)

        if fault_time is None:
            t0 = rng.integers(low=int(T * 0.2), high=int(T * 0.8))
        elif isinstance(fault_time, float):
            t0 = int(T * fault_time) if 0.0 < fault_time < 1.0 else int(fault_time)
        else:
            t0 = int(fault_time)
        t0 = max(0, min(T, t0))

        k = _sample_k(rng, fault_lambda, max_faults=N)
        fault_indices = rng.choice(M, size=k, replace=False)

        faulty = arr.copy()
        for j in range(M):
            ref_mag = float(np.median(np.abs(arr[:, j]))) if T > 0 else 1.0
            if ref_mag < eps:
                ref_mag = 1.0
            noise = rng.normal(loc=0.0, scale=epsilon_scale * ref_mag, size=T)

            if j in fault_indices:
                faulty[t0:, j] = 0.0 + noise[t0:]
                link_idx, motor_idx = divmod(j, 8)
                type_matrix[link_idx, motor_idx] = 0
            else:
                faulty[:, j] = arr[:, j] + noise

        return faulty, type_matrix

    else:
        raise ValueError("[inject_faults] lambda_arr must be 1D (M,) or 2D (T, M).")
