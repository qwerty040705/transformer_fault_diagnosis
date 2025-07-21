import numpy as np

def inject_faults(lambda_clean: np.ndarray, epsilon_scale: float = 0.05, fault_lambda: float = 1):
    """
    Inject faults into lambda_clean using Poisson-distributed number of faults (at least one).
    
    Args:
        lambda_clean (np.ndarray): (8N,) vector of clean thrust values.
        epsilon_scale (float): Noise level as a fraction of original |lambda_clean[i]|.
        fault_lambda (float): λ parameter of Poisson distribution (expected number of faults).
    
    Returns:
        lambda_faulty (np.ndarray): Modified lambda with faults.
        type_matrix (np.ndarray): (N, 8) matrix with 1 (normal) or 0 (faulty).
    """
    lambda_faulty = lambda_clean.copy()
    N_total = lambda_clean.shape[0]
    N = N_total // 8
    max_faults = N  # 최대 고장 개수는 링크 수

    type_matrix = np.ones((N, 8), dtype=int)  # 1 = 정상, 0 = 고장

    # 고장 개수 결정 (1 이상 N 이하)
    while True:
        k = np.random.poisson(lam=fault_lambda)
        if 1 <= k <= max_faults:
            break

    # 고장 motor 선택
    fault_indices = np.random.choice(N_total, size=k, replace=False)

    for idx in fault_indices:
        link_idx = idx // 8
        motor_idx = idx % 8

        # 해당 모터의 원래 thrust 크기에 비례한 노이즈 표준편차
        local_std = epsilon_scale * abs(lambda_clean[idx])
        lambda_faulty[idx] = np.random.normal(loc=0.0, scale=local_std)
        type_matrix[link_idx, motor_idx] = 0  # 0 = 고장

    return lambda_faulty, type_matrix
