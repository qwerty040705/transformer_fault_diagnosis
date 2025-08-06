# fault_injection_fast.py
import numpy as np
from typing import Tuple, Union

def inject_faults_fast(lambda_arr: np.ndarray,
                       fault_time: Union[int, float, None] = None,
                       epsilon_scale: float = 0.01,
                       fault_lambda: float = 1.0,
                       seed: Union[int, None] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    고속(벡터화) fault 주입:
    - 1D (M,) : 일부 모터 값을 0+noise, 나머지는 value+noise
    - 2D (T,M): t0 이후 선택된 모터는 0+noise, 나머지는 전체구간 value+noise
    - 링크당 8개 모터 가정(M%8==0), type_matrix: (N,8)
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(lambda_arr)
    eps = 1e-9

    if arr.ndim == 1:
        M = arr.shape[0]
        if M % 8 != 0:
            raise ValueError("width must be multiple of 8")
        N = M // 8
        k = min(N, max(1, int(rng.poisson(fault_lambda))))
        idx = rng.choice(M, size=k, replace=False)

        type_matrix = np.ones((N, 8), dtype=int)
        link_idx, motor_idx = np.divmod(idx, 8)
        type_matrix[link_idx, motor_idx] = 0

        ref = np.abs(arr).astype(float)
        ref[ref < eps] = 1.0
        noise = rng.normal(0.0, epsilon_scale, size=M) * ref
        out = arr + noise
        out[idx] = noise[idx]  # 고장: 0 + noise
        return out, type_matrix

    elif arr.ndim == 2:
        T, M = arr.shape
        if M % 8 != 0:
            raise ValueError("width must be multiple of 8")
        N = M // 8

        # t0 설정
        if fault_time is None:
            t0 = T // 2
        elif isinstance(fault_time, float):
            t0 = int(T * fault_time) if 0.0 < fault_time < 1.0 else int(fault_time)
        else:
            t0 = int(fault_time)
        t0 = max(0, min(T, t0))

        k = min(N, max(1, int(rng.poisson(fault_lambda))))
        idx = rng.choice(M, size=k, replace=False)

        type_matrix = np.ones((N, 8), dtype=int)
        link_idx, motor_idx = np.divmod(idx, 8)
        type_matrix[link_idx, motor_idx] = 0

        # 기준 크기(모터별 전체구간 median|.|)
        ref = np.median(np.abs(arr), axis=0).astype(float)
        ref[ref < eps] = 1.0
        noise = rng.normal(0.0, epsilon_scale, size=(T, M)) * ref[None, :]

        out = arr + noise
        if t0 < T:
            out[t0:, idx] = noise[t0:, idx]  # 고장: t0 이후 0 + noise
        return out, type_matrix

    else:
        raise ValueError("lambda_arr must be 1D or 2D")
