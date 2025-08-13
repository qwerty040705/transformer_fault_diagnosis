import numpy as np
from typing import Tuple, Union, Optional, Union as _Union

def inject_faults_fast(lambda_arr: np.ndarray,
                       fault_time: _Union[int, float, None] = None,
                       epsilon_scale: float = 0,
                       fault_lambda: float = 1.0,
                       seed: Optional[int] = None,
                       return_labels: bool = False
                       ) -> _Union[
                            Tuple[np.ndarray, np.ndarray],
                            Tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]
                       ]:
    """
    고장 주입기 (fast):
      - 입력 1D (M,): (out, type_matrix)
      - 입력 2D (T,M):
          return_labels=False →
              (out, type_matrix)
          return_labels=True  →
              (out, type_matrix, label_TxM, t0, idx, which_fault_mask, onset_idx)

    정의:
      * type_matrix: (N,8) (N=M//8) 링크×모터 라벨(시간축 없음, 0=고장, 1=정상)
      * label_TxM : (T,M) 시간축 라벨(1=정상, 0=고장), t>=t0 & m∈idx → 0
      * t0: 고장 시작 프레임 인덱스(int). t0==T면 실질적 고장 없음
      * idx: 고장 모터의 컬럼 인덱스 배열(shape (k,))
      * which_fault_mask: (M,) 모터별 고장 존재 여부(0/1)
      * onset_idx: (M,) 모터별 고장 시작 프레임(없으면 -1)
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(lambda_arr)
    eps = 1e-9

    if arr.ndim == 1:
        # ---------------- 1D: 한 시점 ----------------
        M = arr.shape[0]
        if M % 8 != 0:
            raise ValueError("width must be multiple of 8")
        N = M // 8

        # 고장 모터 수 k (링크 수 N에 클램프) — 최소 1
        k = min(N, max(1, int(rng.poisson(fault_lambda))))
        idx = rng.choice(M, size=k, replace=False)
        fault_mask = np.zeros(M, dtype=bool); fault_mask[idx] = True

        # (N,8) 타입 라벨
        type_matrix = np.ones((N, 8), dtype=int)
        link_idx, motor_idx = np.divmod(idx, 8)
        type_matrix[link_idx, motor_idx] = 0

        # 값 생성
        ref = np.abs(arr).astype(float)
        ref[ref < eps] = 1.0
        noise = rng.normal(0.0, epsilon_scale, size=M) * ref
        out = arr + noise
        out[fault_mask] = noise[fault_mask]  # 고장: 베이스 제거, 노이즈만

        return out, type_matrix

    elif arr.ndim == 2:
        # ---------------- 2D: 시계열 ----------------
        T, M = arr.shape
        if M % 8 != 0:
            raise ValueError("width must be multiple of 8")
        N = M // 8

        # 고장 시작 프레임 t0
        if fault_time is None:
            lo = int(T * 0.2)
            hi = max(int(T * 0.8), lo + 1)  # hi는 제외 상한이므로 최소 한 칸 확보
            t0 = int(rng.integers(low=lo, high=hi))
        elif isinstance(fault_time, float) and 0.0 < fault_time < 1.0:
            t0 = int(T * fault_time)
        else:
            t0 = int(fault_time)
        t0 = max(0, min(T, t0))  # 0 ≤ t0 ≤ T

        # 고장 모터 선택
        k = min(N, max(1, int(rng.poisson(fault_lambda))))
        idx = rng.choice(M, size=k, replace=False)
        idx = np.sort(idx)  # 보기 좋게 정렬(선택)
        fault_mask = np.zeros(M, dtype=bool); fault_mask[idx] = True

        # (N,8) 타입 라벨(시간축 없음)
        type_matrix = np.ones((N, 8), dtype=int)
        link_idx, motor_idx = np.divmod(idx, 8)
        type_matrix[link_idx, motor_idx] = 0

        # 시간축 라벨 (T,M): 기본 1(정상), t>=t0 & m in idx → 0(고장)
        label_TxM = np.ones((T, M), dtype=int)
        if t0 < T:
            label_TxM[t0:, fault_mask] = 0

        # 메타: 어느 모터가 고장 / 온셋 프레임
        which_fault_mask = fault_mask.astype(np.int32)      # (M,) 0/1
        onset_idx = np.full(M, -1, dtype=np.int32)          # (M,)
        if t0 < T:
            onset_idx[fault_mask] = t0

        # 값 생성
        ref = np.median(np.abs(arr), axis=0).astype(float)
        ref[ref < eps] = 1.0
        noise = rng.normal(0.0, epsilon_scale, size=(T, M)) * ref[None, :]

        out = arr + noise
        if t0 < T:
            # t0 이후 고장 모터는 원신호 대신 노이즈만(베이스 제거)
            out[t0:, fault_mask] = noise[t0:, fault_mask]

        if return_labels:
            return out, type_matrix, label_TxM, int(t0), idx, which_fault_mask, onset_idx
        else:
            return out, type_matrix

    else:
        raise ValueError("lambda_arr must be 1D or 2D")
