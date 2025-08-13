import numpy as np
from typing import Tuple, Union, Optional, Union as _Union  # alias to avoid clash

def _sample_k(rng: np.random.Generator, lam: float, max_faults: int) -> int:
    """포아송으로 샘플링하되 [1, max_faults] 범위에서 유효값 얻도록 리트라이."""
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
    fault_time: _Union[int, float, None] = None,
    epsilon_scale: float = 0,
    fault_lambda: float = 1.0,
    seed: _Union[int, None] = None,
    return_labels: bool = False,  # 2D에서 시간축 라벨/메타 반환 여부
    **kwargs
) -> _Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]
    ]:
    """
    고장 주입기:
      - 입력 1D (M,):  한 시점의 M채널 → (faulty, type_matrix)
      - 입력 2D (T,M): 시계열          → 
           return_labels=False → (faulty, type_matrix)
           return_labels=True  → (faulty, type_matrix, label_TxM, t0, fault_indices, which_fault_mask, onset_idx)

    정의:
      * type_matrix: (N,8) (N=M//8) 링크×모터 라벨(시간축 없음, 0=고장, 1=정상)
      * label_TxM : (T,M) 시간축 라벨(1=정상, 0=고장), t>=t0 & m∈fault_indices 에서 0
      * t0: 고장 시작 프레임 인덱스(int). t0==T이면 실질적 고장 없음
      * fault_indices: 고장 모터의 컬럼 인덱스 배열(shape (k,))
      * which_fault_mask: (M,) 모터별 고장 존재 여부(0/1)
      * onset_idx: (M,) 모터별 고장 시작 프레임(없으면 -1)
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(lambda_arr)
    eps = 1e-9

    if arr.ndim == 1:
        M = arr.shape[0]
        if M % 8 != 0:
            raise ValueError(f"[inject_faults] length {M} is not a multiple of 8.")
        N = M // 8

        # 고장 모터 선택
        k = _sample_k(rng, fault_lambda, max_faults=N)
        fault_indices = rng.choice(M, size=k, replace=False)
        fault_mask = np.zeros(M, dtype=bool)
        fault_mask[fault_indices] = True

        # (N,8) 타입 라벨
        type_matrix = np.ones((N, 8), dtype=int)
        link_idx, motor_idx = np.divmod(fault_indices, 8)
        type_matrix[link_idx, motor_idx] = 0

        # 신호 생성
        ref = np.abs(arr).astype(float)
        ref[ref < eps] = 1.0
        noise = rng.normal(loc=0.0, scale=epsilon_scale, size=M) * ref
        faulty = arr + noise
        faulty[fault_mask] = noise[fault_mask]  # 고장: 베이스 제거, 노이즈만

        return faulty, type_matrix

    elif arr.ndim == 2:
        T, M = arr.shape
        if M % 8 != 0:
            raise ValueError(f"[inject_faults] width {M} is not a multiple of 8.")
        N = M // 8

        # 고장 시작 프레임 t0
        if fault_time is None:
            lo = int(T * 0.2)
            hi = max(int(T * 0.8), lo + 1)  # 상한 보정(최소 한 칸)
            t0 = int(rng.integers(low=lo, high=hi))
        elif isinstance(fault_time, float) and 0.0 < fault_time < 1.0:
            t0 = int(T * fault_time)
        else:
            t0 = int(fault_time)
        t0 = max(0, min(T, t0))  # 0 ≤ t0 ≤ T

        # 고장 모터 선택
        k = _sample_k(rng, fault_lambda, max_faults=N)
        fault_indices = rng.choice(M, size=k, replace=False)
        fault_indices = np.sort(fault_indices)  # 보기 좋게 정렬(선택)
        fault_mask = np.zeros(M, dtype=bool)
        fault_mask[fault_indices] = True

        # (N,8) 타입 라벨
        type_matrix = np.ones((N, 8), dtype=int)
        link_idx, motor_idx = np.divmod(fault_indices, 8)
        type_matrix[link_idx, motor_idx] = 0

        # 시간축 라벨 (T,M): 기본 1(정상)
        label_TxM = np.ones((T, M), dtype=int)
        if t0 < T:
            label_TxM[t0:, fault_mask] = 0

        # 메타: 모터별 고장 여부/온셋
        which_fault_mask = fault_mask.astype(np.int32)          # (M,)
        onset_idx = np.full(M, -1, dtype=np.int32)              # (M,)
        if t0 < T:
            onset_idx[fault_mask] = t0

        # 신호 생성
        ref = (np.median(np.abs(arr), axis=0).astype(float) if T > 0 else np.ones(M, float))
        ref[ref < eps] = 1.0
        noise = rng.normal(loc=0.0, scale=epsilon_scale, size=(T, M)) * ref[None, :]

        faulty = arr + noise
        if t0 < T:
            faulty[t0:, fault_mask] = noise[t0:, fault_mask]  # 고장: 베이스 제거, 노이즈만

        if return_labels:
            return (
                faulty,            # (T,M)
                type_matrix,       # (N,8)
                label_TxM,         # (T,M)
                int(t0),           # scalar
                fault_indices,     # (k,)
                which_fault_mask,  # (M,) 0/1
                onset_idx          # (M,) -1 or t0
            )
        else:
            return faulty, type_matrix

    else:
        raise ValueError("[inject_faults] lambda_arr must be 1D (M,) or 2D (T, M).")
