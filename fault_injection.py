import numpy as np
from typing import Tuple, Union

def _sample_k(rng: np.random.Generator, lam: float, max_faults: int) -> int:
    """Poisson(lam)에서 1~max_faults 사이가 나올 때까지 샘플링 (fallback 포함)."""
    if max_faults <= 0:
        return 0
    for _ in range(100):
        k = rng.poisson(lam=lam)
        if 1 <= k <= max_faults:
            return k
    # fallback: lam 반올림, 1~max_faults로 클램프
    k = int(round(lam))
    return min(max_faults, max(1, k))

def inject_faults(
    lambda_arr: np.ndarray,
    fault_time: Union[int, float, None] = None,
    epsilon_scale: float = 0.05,
    fault_lambda: float = 1.0,
    seed: Union[int, None] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    시계열(2D) 또는 단일 시점(1D) thrust 벡터에 모터 고장을 주입합니다.

    Parameters
    ----------
    lambda_arr : np.ndarray
        - (T, M) 또는 (M,) 배열. M = 8 * N (N = 링크 수).
        - 2D인 경우 fault_time 이후 구간에만 고장 적용.
    fault_time : int | float | None
        - 2D 입력일 때만 사용.
        - None: T//2 에서 시작
        - int: 해당 시점에서 시작 (0 ≤ fault_time ≤ T)
        - float∈(0,1): 전체 길이의 비율로 해석 (예: 0.5 → T//2)
    epsilon_scale : float
        - 노이즈 표준편차를 정하는 스케일링 계수 (기준 크기 × epsilon_scale).
    fault_lambda : float
        - 포아송 분포의 λ. 기대 고장 개수.
    seed : int | None
        - 랜덤 시드.
    **kwargs :
        - 향후 호환성을 위한 잉여 파라미터 (무시).

    Returns
    -------
    lambda_faulty : np.ndarray
        - 입력과 동일한 shape. 고장 주입 후 thrust.
    type_matrix : np.ndarray
        - shape = (N, 8). 각 (링크, 모터)에 대해 1=정상, 0=고장.
          (시간에 따라 바뀌지 않는 모터별 고장 여부를 반환)
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(lambda_arr)
    if arr.ndim == 1:
        # ---- (M,) 케이스 ----
        M = arr.shape[0]
        if M % 8 != 0:
            raise ValueError(f"[inject_faults] length {M} is not a multiple of 8.")
        N = M // 8
        type_matrix = np.ones((N, 8), dtype=int)

        k = _sample_k(rng, fault_lambda, max_faults=N)  # 링크 당 최대 1개 고장 가정
        fault_indices = rng.choice(M, size=k, replace=False)

        faulty = arr.copy()
        eps = 1e-9
        for idx in fault_indices:
            link_idx = idx // 8
            motor_idx = idx % 8
            ref_mag = abs(arr[idx])
            if ref_mag < eps:
                ref_mag = 1.0
            local_std = epsilon_scale * ref_mag
            # 완전 고장에 가까운 노이즈(0 중심, 작은 분산)로 치환
            faulty[idx] = rng.normal(loc=0.0, scale=local_std)
            type_matrix[link_idx, motor_idx] = 0

        return faulty, type_matrix

    elif arr.ndim == 2:
        # ---- (T, M) 케이스 ----
        T, M = arr.shape
        if M % 8 != 0:
            raise ValueError(f"[inject_faults] width {M} is not a multiple of 8.")
        N = M // 8
        type_matrix = np.ones((N, 8), dtype=int)

        # fault 시작 시점 결정
        if fault_time is None:
            t0 = T // 2
        elif isinstance(fault_time, float):
            t0 = int(T * fault_time) if 0.0 < fault_time < 1.0 else int(fault_time)
        else:
            t0 = int(fault_time)
        t0 = max(0, min(T, t0))  # [0, T]로 클램프

        k = _sample_k(rng, fault_lambda, max_faults=N)
        fault_indices = rng.choice(M, size=k, replace=False)

        faulty = arr.copy()
        eps = 1e-9
        for j in fault_indices:
            link_idx = j // 8
            motor_idx = j % 8

            # 기준 크기: 전체 혹은 t0 이후 구간의 절대값 중앙값(robust)
            ref_mag = float(np.median(np.abs(arr[:, j]))) if T > 0 else 0.0
            if ref_mag < eps:
                ref_mag = 1.0
            local_std = epsilon_scale * ref_mag

            if t0 < T:
                # t0 이후를 0 중심 가우시안 노이즈로 치환 (완전/부분 고장 모델)
                noise = rng.normal(loc=0.0, scale=local_std, size=T - t0)
                faulty[t0:, j] = noise

            type_matrix[link_idx, motor_idx] = 0

        return faulty, type_matrix

    else:
        raise ValueError("[inject_faults] lambda_arr must be 1D (M,) or 2D (T, M).")
