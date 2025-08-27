# visualize_stream_casual_tcn.py
# -*- coding: utf-8 -*-

import sys, os, time, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from casual_tcn.stream_casual_tcn_demo import (
    FaultDiagnosisTCN, receptive_field, KofN, Streamer, load_series_from_npz
)

# ---------------------- utils ----------------------
def set_seed(seed: int):
    import random
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pick_device():
    # CUDA > MPS > CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_ok = getattr(torch.backends, "mps", None)
    if mps_ok is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_poses_from_npz(npz_path: str, seq_idx: int):
    d = np.load(npz_path, allow_pickle=True)
    keys = set(d.files)
    if {"actual_link_cum", "desired_link_cum"}.issubset(keys):
        A = d["actual_link_cum"]   # [S,T,L,4,4]  (누적변환)
        D = d["desired_link_cum"]  # [S,T,L,4,4]
    else:
        raise KeyError(f"NPZ에 cum pose가 없습니다. keys={sorted(keys)}")
    assert 0 <= seq_idx < A.shape[0]
    return D[seq_idx], A[seq_idx]  # [T,L,4,4], [T,L,4,4]

def prepend_base_identity(cum):
    """데이터에 기저 프레임이 빠져 있으면 단위행렬을 앞에 붙여 링크수를 +1."""
    T, L, _, _ = cum.shape
    I = np.tile(np.eye(4), (T, 1, 1))  # [T,4,4]
    I = I[:, None, :, :]               # [T,1,4,4]
    return np.concatenate([I, cum], axis=1)  # [T,L+1,4,4]

# ---------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--seq_idx", type=int, default=0)

    # 모델 / 추론
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--kofn", type=str, default="3,5")       # e.g., "3,5", "0,0" to disable
    ap.add_argument("--override_lookback", type=int, default=0)

    # 시각화/재생
    ap.add_argument("--fps", type=float, default=30.0)       # 기본 프레임속도 ↑
    ap.add_argument("--speed", type=float, default=3.0,      # 재생 가속도(>1 더 빠름) ↑
                    help="실시간 대비 배속. interval=1/(fps*speed)")
    ap.add_argument("--data_hz", type=float, default=20.0,   # 데이터 샘플레이트(초 환산용)
                    help="t_real 계산용 샘플링 주파수(Hz)")
    ap.add_argument("--prepend_base", type=int, default=1,   # base 프레임 자동 추가
                    help="1이면 base 단위행렬을 프레임 앞에 추가")

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device()
    print("📥 device:", device)

    # ---- 체크포인트 로드 ----
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    D_in = int(ckpt["input_dim"])
    M_out = int(ckpt["M"])
    cfg = ckpt["cfg"]; hidden=int(cfg.get("hidden",128))
    layers = int(cfg.get("layers",6))
    ksize  = int(cfg.get("k",3))
    dropout= float(cfg.get("dropout",0.1))
    mu = ckpt["train_mean"]; std = ckpt["train_std"]

    model = FaultDiagnosisTCN(D_in, M_out, hidden=hidden, layers=layers, k=ksize, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    R = receptive_field(layers, ksize)
    lookback = args.override_lookback if args.override_lookback > 0 else R
    print(f"🧮 receptive_field R≈{R} | lookback={lookback} | threshold={args.threshold}")

    # ---- 스트림 피처/레이블 ----
    X, Y = load_series_from_npz(args.npz, seq_idx=args.seq_idx)  # X:[T,D], Y:[T,M]
    T = X.shape[0]
    assert X.shape[1] == D_in, f"NPZ feature dim {X.shape[1]} != ckpt {D_in}"
    assert Y.shape[1] == M_out, f"NPZ label dim {Y.shape[1]} != ckpt {M_out}"

    # ---- 포즈(누적변환) ----
    Dcum, Acum = load_poses_from_npz(args.npz, seq_idx=args.seq_idx)  # [T,L,4,4]
    if args.prepend_base:
        Dcum = prepend_base_identity(Dcum)
        Acum = prepend_base_identity(Acum)

    T_pose, L = Acum.shape[0], Acum.shape[1]
    assert T_pose == T, "pose 시퀀스 길이와 피처 길이가 다릅니다."
    print(f"🦾 links={L-1} | motors={M_out} | timesteps={T}")

    # ---- 스트리머 구성 ----
    kofn = None
    if args.kofn != "0,0":
        K, N = map(int, args.kofn.split(","))
        kofn = KofN(M=M_out, K=K, N=N)
        print(f"🛡️  K-of-N smoothing: K={K}, N={N}")

    streamer = Streamer(model, mu, std, device, lookback=lookback, threshold=args.threshold, kofn=kofn)

    # ---- Matplotlib 세팅 ----
    plt.close("all")
    fig = plt.figure(figsize=(12,6))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])

    ax3d = fig.add_subplot(gs[0,0], projection="3d")
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.set_title("Robot Arm (desired vs actual)")

    axbar = fig.add_subplot(gs[0,1])
    axbar.set_ylim(0,1.0)
    axbar.set_ylabel("Prob(fault)")
    axbar.set_title("Motor-wise probability (K-of-N after thresholding)")

    # 월드 범위 자동 산정(처음 200프레임 샘플)
    sample = Acum[:min(200, T)]
    p = sample[..., :3, 3].reshape(-1, 3)
    pmin, pmax = p.min(axis=0), p.max(axis=0)
    span = (pmax - pmin).max()
    center = (pmax + pmin)/2
    lim = span*0.8 if span > 0 else 0.5
    ax3d.set_xlim(center[0]-lim, center[0]+lim)
    ax3d.set_ylim(center[1]-lim, center[1]+lim)
    ax3d.set_zlim(center[2]-lim, center[2]+lim)

    # 링크(세그먼트) 선들 준비: desired(녹색 점선), actual(파랑→빨강)
    desired_lines = []
    actual_lines  = []
    for _ in range(L-1):
        d_ln, = ax3d.plot([], [], [], linestyle="--", color="g", lw=2, alpha=0.8)
        a_ln, = ax3d.plot([], [], [], linestyle="-",  color="C0", lw=4, alpha=0.9)
        desired_lines.append(d_ln)
        actual_lines.append(a_ln)

    # 노드(조인트) 점 – 필요하면 사용
    desired_nodes = ax3d.scatter([], [], [], s=25, c="g", alpha=0.6)
    actual_nodes  = ax3d.scatter([], [], [], s=30, c="C0", alpha=0.9)

    # 모터 막대
    bar_x = np.arange(M_out)
    bars = axbar.bar(bar_x, np.zeros(M_out))
    axbar.set_xticks(bar_x)
    axbar.set_xticklabels([f"M{i}" for i in range(M_out)], rotation=0)

    # 범례/텍스트
    legend_lines = [
        plt.Line2D([0],[0], color="C0", lw=4, label="actual"),
        plt.Line2D([0],[0], color="g",  lw=2, linestyle="--", label="desired")
    ]
    ax3d.legend(handles=legend_lines, loc="upper left")

    txt = ax3d.text2D(0.02, 0.92, "", transform=ax3d.transAxes)

    # 상태
    t_idx = [0]
    prob_last = np.zeros(M_out, dtype=float)
    predk_last = np.zeros(M_out, dtype=np.uint8)
    fault_seen = [False]  # 한번 고장 나면 이후 계속 빨강 유지

    # 재생 속도(간격 ms)
    # interval = 1000 / (fps * speed)
    interval_ms = max(1, int(1000.0 / max(1e-6, args.fps * args.speed)))

    def update(_):
        t = t_idx[0]
        if t >= T:
            return []

        # ---- 추론 1스텝 ----
        probs, pred_raw, pred_k, Lwin = streamer.step(X[t])
        m = M_out
        prob_last[:m] = probs[:m]
        predk_last[:m] = pred_k[:m]

        # 고장 감지(한 번 켜지면 유지)
        if predk_last.any():
            fault_seen[0] = True

        # ---- 포즈 갱신 ----
        Td = Dcum[t]           # [L,4,4]
        Ta = Acum[t]           # [L,4,4]
        P_d = Td[:, :3, 3]     # [L,3]
        P_a = Ta[:, :3, 3]     # [L,3]

        # 링크 선 갱신
        for i in range(L-1):
            # desired
            xd = [P_d[i,0], P_d[i+1,0]]
            yd = [P_d[i,1], P_d[i+1,1]]
            zd = [P_d[i,2], P_d[i+1,2]]
            desired_lines[i].set_data(xd, yd)
            desired_lines[i].set_3d_properties(zd)

            # actual
            xa = [P_a[i,0], P_a[i+1,0]]
            ya = [P_a[i,1], P_a[i+1,1]]
            za = [P_a[i,2], P_a[i+1,2]]
            actual_lines[i].set_data(xa, ya)
            actual_lines[i].set_3d_properties(za)
            actual_lines[i].set_color("r" if fault_seen[0] else "C0")

        # 노드 점 갱신 (scatter는 _offsets3d로)
        desired_nodes._offsets3d = (P_d[:,0], P_d[:,1], P_d[:,2])
        actual_nodes._offsets3d  = (P_a[:,0], P_a[:,1], P_a[:,2])
        actual_nodes.set_color("r" if fault_seen[0] else "C0")

        # ---- 막대 갱신(모터별) ----
        for i, b in enumerate(bars):
            b.set_height(prob_last[i])
            is_alarm = bool(predk_last[i])
            b.set_edgecolor("r" if is_alarm else "black")
            b.set_linewidth(2.5 if is_alarm else 0.5)

        t_real = t / max(1e-6, args.data_hz)
        txt.set_text(f"t={t:04d} | t_real={t_real:6.3f}s | lookback={Lwin} | alarms(motors)={int(predk_last.sum())}/{M_out}")

        t_idx[0] += 1
        return desired_lines + actual_lines + [desired_nodes, actual_nodes] + list(bars) + [txt]

    ani = FuncAnimation(fig, update, interval=interval_ms, blit=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
