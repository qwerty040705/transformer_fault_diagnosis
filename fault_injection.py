import numpy as np
from typing import Tuple, Union, Optional, Union as _Union

def inject_faults(
    lambda_arr: np.ndarray,
    fault_time: _Union[int, float, None] = None,
    epsilon_scale: float = 0.0,
    fault_lambda: float = 1.0,
    seed: _Union[int, None] = None,
    return_labels: bool = False,
    **kwargs
) -> _Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]
]:
    """
    고장 주입기:
      - 입력 1D (M,): 한 시점의 M채널 → (faulty, type_matrix)
      - 입력 2D (T,M): 시계열 →
           return_labels=False → (faulty, type_matrix)
           return_labels=True  → (faulty, type_matrix, label_TxM, t0, fault_indices, which_fault_mask, onset_idx)

    수정 원칙:
      - 고장 이전(t < t0)에는 노이즈 미적용.
      - 건강한 모터에는 전 구간 노이즈 미적용.
      - 고장난 모터는 t0 이후에만 '노이즈만' 남기고 베이스 제거.
      - Poisson 샘플에서 k=0도 허용(완전 정상 샘플).
      - k==0이면 t0=T로 표기(실질적 고장 없음).
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(lambda_arr)
    eps = 1e-9

    if arr.ndim == 1:
        # --------------- 1D: 한 시점 ---------------
        M = arr.shape[0]
        if M % 8 != 0:
            raise ValueError("[inject_faults] length must be a multiple of 8.")
        N = M // 8

        # 고장 모터 수 k (0 허용)
        k_raw = rng.poisson(lam=fault_lambda)
        k = int(np.clip(k_raw, 0, N))

        if k > 0:
            fault_indices = np.sort(rng.choice(M, size=k, replace=False))
        else:
            fault_indices = np.array([], dtype=int)

        fault_mask = np.zeros(M, dtype=bool)
        fault_mask[fault_indices] = True

        # (N,8) 타입 라벨 (모터별 고장 여부: 0 or 1)
        type_matrix = np.ones((N, 8), dtype=int)
        if k > 0:
            link_idx, motor_idx = np.divmod(fault_indices, 8)
            type_matrix[link_idx, motor_idx] = 0

        # 출력 신호 생성: 기본적으로 원신호 유지
        faulty = arr.copy()
        if k > 0:
            ref = np.abs(arr).astype(float)
            ref[ref < eps] = 1.0
            noise = rng.normal(0.0, epsilon_scale, size=M) * ref
            faulty[fault_mask] = noise[fault_mask]  # 고장 모터만 노이즈로 대체

        return faulty, type_matrix

    elif arr.ndim == 2:
        # --------------- 2D: 시계열 ---------------
        T, M = arr.shape
        if M % 8 != 0:
            raise ValueError("[inject_faults] width must be a multiple of 8.")
        N = M // 8

        # 고장 시작 프레임 결정
        if fault_time is None:
            lo = int(T * 0.2)
            hi = max(int(T * 0.8), lo + 1)  # hi는 제외 상한
            t0 = int(rng.integers(low=lo, high=hi))
        elif isinstance(fault_time, float) and 0.0 < fault_time < 1.0:
            t0 = int(T * fault_time)
        else:
            t0 = int(fault_time)
        t0 = max(0, min(T, t0))  # 0 ≤ t0 ≤ T

        # 고장 모터 수 k (0 허용)
        k_raw = rng.poisson(lam=fault_lambda)
        k = int(np.clip(k_raw, 0, N))
        if k > 0:
            fault_indices = np.sort(rng.choice(M, size=k, replace=False))
        else:
            fault_indices = np.array([], dtype=int)

        fault_mask = np.zeros(M, dtype=bool)
        fault_mask[fault_indices] = True

        # (N,8) 타입 라벨 생성 (모터별 0/1)
        type_matrix = np.ones((N, 8), dtype=int)
        if k > 0:
            link_idx, motor_idx = np.divmod(fault_indices, 8)
            type_matrix[link_idx, motor_idx] = 0

        # 시간축 레이블 – 초기 1, 고장 모터는 t0부터 0
        label_TxM = np.ones((T, M), dtype=int)
        if k > 0 and t0 < T:
            label_TxM[t0:, fault_mask] = 0

        # 메타: which_fault_mask/onset_idx
        which_fault_mask = fault_mask.astype(np.int32)
        onset_idx = np.full(M, -1, dtype=np.int32)
        if k > 0 and t0 < T:
            onset_idx[fault_mask] = t0
        if k == 0:
            t0 = T  # 고장 없음 표시

        # 출력 신호 생성: 기본적으로 원신호 유지
        faulty = arr.copy()
        if k > 0 and t0 < T:
            # t0 이후 고장 모터는 베이스 제거하고 노이즈만 남김
            ref = np.median(np.abs(arr), axis=0).astype(float) if T > 0 else np.ones(M, float)
            ref[ref < eps] = 1.0
            noise = rng.normal(0.0, epsilon_scale, size=(T, M)) * ref[None, :]
            faulty[t0:, fault_mask] = noise[t0:, fault_mask]

        if return_labels:
            return (
                faulty,                 # (T, M)
                type_matrix,            # (N, 8)
                label_TxM,              # (T, M)
                int(t0),                # t0 (k==0이면 T)
                fault_indices,          # (k,)
                which_fault_mask,       # (M,)
                onset_idx               # (M,)
            )
        else:
            return faulty, type_matrix

    else:
        raise ValueError("[inject_faults] lambda_arr must be 1D or 2D.")
