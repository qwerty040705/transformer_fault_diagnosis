import sys, os, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d  # ← 3D→2D 투영(항상 수평 라벨용)

from causal_tcn.stream_causal_tcn_demo import (
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
        A = d["actual_link_cum"]   # [S,T,L,4,4]
        D = d["desired_link_cum"]  # [S,T,L,4,4]
    else:
        raise KeyError(f"NPZ에 cum pose가 없습니다. keys={sorted(keys)}")
    assert 0 <= seq_idx < A.shape[0]
    return D[seq_idx], A[seq_idx]

def prepend_base_identity(cum):
    T, L, _, _ = cum.shape
    I = np.tile(np.eye(4), (T, 1, 1))  # [T,4,4]
    I = I[:, None, :, :]               # [T,1,4,4]
    return np.concatenate([I, cum], axis=1)  # [T,L+1,4,4]

def positions_from_cum(cum):
    return cum[..., :3, 3]

def normalize_by_base(P):
    if P.ndim == 3:      # [T,L,3]
        return P - P[:, :1, :]
    elif P.ndim == 2:    # [L,3]
        return P - P[:1, :]
    return P

# ---------------------- anchor & blade helpers ----------------------
def link_anchor_points(P0, P1, ratio_front: float):
    """
    링크 시작 P0 → 끝 P1.
    ratio_front 지점에 '앞' 앵커, (1-ratio_front) 지점에 '뒤' 앵커를 둔다.
    """
    ratio_back = 1.0 - ratio_front
    p_front = (1.0 - ratio_front) * P0 + ratio_front * P1
    p_back  = (1.0 - ratio_back)  * P0 + ratio_back  * P1
    return p_front, p_back

def cross_four_positions(p_anchor, R_local, arm_len):
    """
    앵커 기준 로컬 Y/Z축 ±arm_len 위치에 4개 모터(십자).
    (두 지점에서 4개씩, 총 8개)
    """
    y = R_local[:, 1]
    z = R_local[:, 2]
    return np.array([
        p_anchor + arm_len * y,
        p_anchor - arm_len * y,
        p_anchor + arm_len * z,
        p_anchor - arm_len * z,
    ], dtype=float)  # (4,3)

def _norm(v, eps=1e-9):
    n = np.linalg.norm(v)
    return v / (n + eps)

def orthonormal_blade_axes(n_hat: np.ndarray, R_ref: np.ndarray):
    """
    stem 방향 n_hat에 수직인 2-직교축 {u, v} 생성.
    R_ref의 (Y,Z)축을 n_hat에 직교화해서 일관성 유지.
    """
    y_ref = R_ref[:, 1]
    z_ref = R_ref[:, 2]
    u = y_ref - np.dot(y_ref, n_hat) * n_hat
    if np.linalg.norm(u) < 1e-6:
        u = z_ref - np.dot(z_ref, n_hat) * n_hat
        if np.linalg.norm(u) < 1e-6:
            u = np.array([1.0, 0.0, 0.0]) - np.dot([1.0,0.0,0.0], n_hat) * n_hat
    u = _norm(u)
    v = _norm(np.cross(n_hat, u))
    return u, v

def blade_quad(center, dir_u, dir_v, radius, chord, theta):
    """
    사각 블레이드 하나의 꼭짓점 4개 생성.
    center: (3,) 모터 위치
    dir_u, dir_v: stem에 수직인 직교 기저
    radius: 팁 반지름
    chord : 현폭
    theta : u-v 평면에서의 회전각(라디안)
    """
    c = np.cos(theta); s = np.sin(theta)
    axis =  c * dir_u + s * dir_v       # 블레이드 길이 방향
    perp = -s * dir_u + c * dir_v       # 블레이드 폭 방향

    r_root = 0.25 * radius
    r_tip  = radius
    half_c = 0.5  * chord

    root = center + r_root * axis
    tip  = center + r_tip  * axis

    p1 = root + half_c * perp
    p2 = tip  + half_c * perp
    p3 = tip  - half_c * perp
    p4 = root - half_c * perp
    return np.stack([p1, p2, p3, p4], axis=0)

# ---------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--seq_idx", type=int, default=0)

    # 모델 / 추론
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--kofn", type=str, default="3,5")     # "3,5", "0,0" to disable
    ap.add_argument("--override_lookback", type=int, default=0)

    # 시각화/재생
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--data_hz", type=float, default=20.0)
    ap.add_argument("--prepend_base", type=int, default=1)
    ap.add_argument("--fix_origin", type=int, default=1)

    # 라벨 스킴
    ap.add_argument("--label_fault_is_one", type=int, default=1)

    # 모터/레이아웃
    ap.add_argument("--motors_per_link", type=int, default=8)
    ap.add_argument("--anchor_ratio", type=float, default=0.85,
                    help="앞 anchor 비율(뒤는 1-ratio). 그림처럼 링크에 수직이 되도록 값을 조정.")
    ap.add_argument("--arm_len", type=float, default=0.22,
                    help="anchor→모터 거리(십자 팔 길이)")

    # 프로펠러(블레이드)
    ap.add_argument("--prop_blades", type=int, default=4, help="블레이드 개수(권장 4)")
    ap.add_argument("--prop_radius", type=float, default=0.10, help="블레이드 반지름")
    ap.add_argument("--prop_chord",  type=float, default=0.035, help="블레이드 현폭")
    ap.add_argument("--prop_alpha",  type=float, default=0.85,  help="블레이드 투명도")
    ap.add_argument("--stem_alpha",  type=float, default=0.95,  help="링크-모터 연결선 투명도")

    # 회전 애니메이션(정상 모터만 회전)
    ap.add_argument("--prop_rps", type=float, default=15.0,
                    help="정상 모터의 회전속도(초당 회전수, rev/s). 고장 모터는 0.")
    ap.add_argument("--spin_dir_alt", type=int, default=1,
                    help="1이면 모터마다 회전방향 번갈아가며(+1/-1)")

    # 비디오 저장
    ap.add_argument("--save_video", type=int, default=0)
    ap.add_argument("--out", type=str, default="output.mp4")
    ap.add_argument("--video_fps", type=int, default=30)
    ap.add_argument("--codec", type=str, default="libx264")
    ap.add_argument("--bitrate", type=str, default="4000k")
    ap.add_argument("--dpi", type=int, default=150)

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    def _parse_bitrate_to_kbps(b):
        s = str(b).strip().lower()
        if s.endswith("k"):
            s = s[:-1]
        return int(float(s))

    bitrate_kbps = _parse_bitrate_to_kbps(args.bitrate)

    set_seed(args.seed)
    device = pick_device()
    print("📥 device:", device)

    # ---- 체크포인트 로드 ----
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    # state dict에서 시각화 모델에 없는 키 제거
    state = dict(ckpt["model_state"])
    state.pop("logit_scale", None)
    state.pop("logit_shift", None)

    D_in  = int(ckpt["input_dim"])
    M_out = int(ckpt["M"])
    cfg   = ckpt["cfg"]; hidden=int(cfg.get("hidden",128))
    layers= int(cfg.get("layers",6))
    ksize = int(cfg.get("k",3))
    dropout=float(cfg.get("dropout",0.1))
    mu = ckpt["train_mean"]; std = ckpt["train_std"]

    model = FaultDiagnosisTCN(D_in, M_out, hidden=hidden, layers=layers, k=ksize, dropout=dropout).to(device)
    model.load_state_dict(state, strict=True)   # ✅ 불필요한 키 제거 후 로드 → 에러 해결
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
    link_count = L - 1
    assert link_count >= 1, "링크가 최소 1개 이상이어야 합니다."
    motors_per_link = args.motors_per_link
    assert motors_per_link * link_count == M_out, \
        f"모터 수 불일치: {motors_per_link}*{link_count} != {M_out}"

    print(f"🦾 links={link_count} | motors={M_out} | timesteps={T} | fix_origin={args.fix_origin}")

    # ---- 스트리머 구성 ----
    kofn = None
    if args.kofn != "0,0":
        K, N = map(int, args.kofn.split(","))
        kofn = KofN(M=M_out, K=K, N=N)
        print(f"🛡️  K-of-N smoothing: K={K}, N={N}")

    streamer = Streamer(model, mu, std, device, lookback=lookback, threshold=args.threshold, kofn=kofn)

    # ===== Matplotlib 레이아웃 =====
    rows = link_count
    fig_h = 6 + 1.8 * max(0, rows - 1)
    plt.close("all")
    fig = plt.figure(figsize=(12, fig_h))
    gs = fig.add_gridspec(rows, 2, width_ratios=[3, 2], height_ratios=[1]*rows, wspace=0.35, hspace=0.45)

    # 3D 축
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.set_title("LASDRA (desired vs actual)", fontsize = 18)

    # 오른쪽 막대축
    axbars = [fig.add_subplot(gs[r, 1]) for r in range(rows)]

    # 월드 범위 산정
    sample = Acum[:min(200, T)]
    p_all = positions_from_cum(sample)
    if args.fix_origin:
        p_all = normalize_by_base(p_all)
    p = p_all.reshape(-1, 3)
    pmin, pmax = p.min(axis=0), p.max(axis=0)
    span = (pmax - pmin).max()
    center = (pmax + pmin)/2
    lim = span*0.8 if span > 0 else 0.5
    ax3d.set_xlim(center[0]-lim, center[0]+lim)
    ax3d.set_ylim(center[1]-lim, center[1]+lim)
    ax3d.set_zlim(center[2]-lim, center[2]+lim)

    # 링크 선 (Desired=초록 점선, Actual=검은 실선(살짝 투명))
    desired_lines, actual_lines = [], []
    for _ in range(link_count):
        d_ln, = ax3d.plot([], [], [], linestyle="--", color="g", lw=2.0, alpha=1.0)
        a_ln, = ax3d.plot([], [], [], linestyle="-",  color="k", lw=2.5, alpha=0.45)
        desired_lines.append(d_ln)
        actual_lines.append(a_ln)

    # 노드
    desired_nodes = ax3d.scatter([], [], [], s=15, c="g", alpha=1.0)
    actual_nodes  = ax3d.scatter([], [], [], s=18, c="k", alpha=0.45)

    # BASE
    base_marker = ax3d.scatter([0], [0], [0], s=120, marker='o',
                               facecolor='k', edgecolor='y', linewidth=2.0, alpha=1.0, zorder=5)
    base_text   = ax3d.text(0.05, 0.05, 0.05, "BASE", color="y", fontsize=10, ha="left", va="bottom")

    # ===== 모터 아티스트: stem(선) + blades(폴리곤 여러개) + fault 라벨 =====
    assert motors_per_link == 8, "이 시각화는 링크당 8모터 기준입니다."
    stems_lines    = [[None]*motors_per_link for _ in range(link_count)]
    blade_patches  = [[[]  for _ in range(motors_per_link)] for _ in range(link_count)]
    fault_texts    = [[None]*motors_per_link for _ in range(link_count)]  # ← 항상 수평 라벨

    for li in range(link_count):
        for mj in range(motors_per_link):
            ln_stem,  = ax3d.plot([], [], [], color="k", lw=1.2, alpha=args.stem_alpha)
            stems_lines[li][mj] = ln_stem
            # 블레이드 폴리곤 생성 (개수 prop_blades)
            patches = []
            for _ in range(args.prop_blades):
                poly = Poly3DCollection([np.zeros((4,3))], closed=True,
                                        facecolor="k", edgecolor="none",
                                        alpha=args.prop_alpha)
                ax3d.add_collection3d(poly)
                patches.append(poly)
            blade_patches[li][mj] = patches
            # 고장 라벨(초기에는 빈 텍스트) — 항상 수평 유지용(2D)
            txt = ax3d.text2D(0, 0, "", color="r", fontsize=8,
                              ha="left", va="center", transform=ax3d.transData)
            fault_texts[li][mj] = txt

    # 인덱스 헬퍼
    def link_motor_slice(link_idx):
        m0 = link_idx * motors_per_link
        m1 = m0 + motors_per_link
        return m0, m1

    # 오른쪽 막대
    bars_gt_rows, bars_pred_rows, gt_texts_rows = [], [], []
    width = 0.35
    for r, ax in enumerate(axbars):
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Link {r+1} (M1–M{motors_per_link})")
        idxs  = np.arange(motors_per_link)
        x_gt   = idxs - width/2
        x_pred = idxs + width/2

        bars_gt = ax.bar(x_gt, np.zeros(motors_per_link), width=width, alpha=0.35,
                         linewidth=1.0, edgecolor="gray", hatch="//", label="REAL VALUE (fault=1)")
        bars_pd = ax.bar(x_pred, np.zeros(motors_per_link), width=width,
                         label="Pred prob(fault)")

        ax.set_xticks(idxs)
        ax.set_xticklabels([f"M{i+1}" for i in range(motors_per_link)], rotation=0)

        if r == 0:
            ax.legend(loc="upper right")
        else:
            if hasattr(ax, "legend_") and ax.legend_:
                ax.legend_.remove()

        gt_txts = [ax.text(i, 1.02, "", ha="center", va="bottom", fontsize=8) for i in idxs]

        bars_gt_rows.append(bars_gt)
        bars_pred_rows.append(bars_pd)
        gt_texts_rows.append(gt_txts)

    # 범례
    legend_lines = [
        plt.Line2D([0],[0], color="k", lw=2.5, label="Actual Link", alpha=0.45),
        plt.Line2D([0],[0], color="g", lw=2.0, linestyle="--", label="Desired Link"),
        plt.Line2D([0],[0], marker="$x$", color="k", lw=0, markersize=8, label="Motor"),
        plt.Line2D([0],[0], marker="$x$", color="r", lw=0, markersize=8, label="Faulty Motor"),
    ]
    ax3d.legend(handles=legend_lines, loc="upper left")

    # 상태 텍스트
    txt = ax3d.text2D(0.58, 0.92, "", transform=ax3d.transAxes, fontsize=14)

    # 상태 버퍼
    t_idx = [0]
    prob_last  = np.zeros(M_out, dtype=float)
    predk_last = np.zeros(M_out, dtype=np.uint8)

    # 회전 위상 (정상 모터만 진행)
    prop_phase = 0.0
    dt = 1.0 / max(1e-6, args.data_hz)
    omega = 2.0 * np.pi * args.prop_rps  # rad/s
    # 모터별 회전방향(+1/-1) 교대
    if args.spin_dir_alt:
        spin_sign = np.array([1 if (i%2==0) else -1 for i in range(motors_per_link)], dtype=float)
    else:
        spin_sign = np.ones(motors_per_link, dtype=float)

    # 재생 간격
    interval_ms = max(1, int(1000.0 / max(1e-6, args.data_hz * args.speed)))

    # 업데이트
    def update(_):
        nonlocal prop_phase
        t = t_idx[0]
        if t >= T:
            return []

        # 추론
        logits, probs, pred_raw, pred_k, Lwin = streamer.step(X[t])
        prob_last[:M_out] = probs[:M_out]
        predk_last[:M_out] = pred_k[:M_out]

        # GT
        if args.label_fault_is_one:
            gt_fault = (Y[t] > 0.5).astype(float)
        else:
            gt_fault = (Y[t] < 0.5).astype(float)

        # 포즈
        Td = Dcum[t]; Ta = Acum[t]
        P_d = Td[:, :3, 3]
        P_a = Ta[:, :3, 3]
        if args.fix_origin:
            P_d = normalize_by_base(P_d)
            P_a = normalize_by_base(P_a)

        # 링크 선
        for i in range(link_count):
            xd, yd, zd = [P_d[i,0], P_d[i+1,0]], [P_d[i,1], P_d[i+1,1]], [P_d[i,2], P_d[i+1,2]]
            desired_lines[i].set_data(xd, yd); desired_lines[i].set_3d_properties(zd)
            xa, ya, za = [P_a[i,0], P_a[i+1,0]], [P_a[i,1], P_a[i+1,1]], [P_a[i,2], P_a[i+1,2]]
            actual_lines[i].set_data(xa, ya);   actual_lines[i].set_3d_properties(za)

        # 노드
        desired_nodes._offsets3d = (P_d[:,0], P_d[:,1], P_d[:,2])
        actual_nodes._offsets3d  = (P_a[:,0], P_a[:,1], P_a[:,2])

        # 정상 모터 회전 위상 업데이트 (고장 모터는 정지)
        prop_phase = (prop_phase + omega * dt) % (2.0*np.pi)

        # 모터(앞/뒤 앵커 4개씩) — 블레이드 평면 ⟂ stem, 라벨은 항상 수평
        for i in range(link_count):
            R_start = Ta[i,   :3, :3]
            R_end   = Ta[i+1, :3, :3]
            p_front, p_back = link_anchor_points(P_a[i], P_a[i+1], args.anchor_ratio)

            four_front = cross_four_positions(p_front, R_end,   args.arm_len)  # (4,3)
            four_back  = cross_four_positions(p_back,  R_start, args.arm_len)  # (4,3)
            motor_pos  = np.vstack([four_front, four_back])                    # (8,3)

            m0, m1 = link_motor_slice(i)
            R_for_blade = [R_end]*4 + [R_start]*4
            anchors     = [p_front]*4 + [p_back]*4

            for j in range(motors_per_link):
                mi = m0 + j
                is_fault = bool(predk_last[mi])
                color_face = "r" if is_fault else "k"

                pj     = motor_pos[j]
                p_anc  = anchors[j]
                R_ref  = R_for_blade[j]

                # stem
                stems_lines[i][j].set_data([p_anc[0], pj[0]], [p_anc[1], pj[1]])
                stems_lines[i][j].set_3d_properties([p_anc[2], pj[2]])
                stems_lines[i][j].set_color(color_face)

                # 블레이드 평면(⊥ stem)
                n_hat = _norm(pj - p_anc)
                u, v  = orthonormal_blade_axes(n_hat, R_ref)

                # 고장 모터는 회전 멈춤(phase 0), 정상은 진행(교대 회전방향 적용)
                base_phase = 0.0 if is_fault else (prop_phase * spin_sign[j])

                for k, poly in enumerate(blade_patches[i][j]):
                    theta = base_phase + 2.0*np.pi * (k / max(1, args.prop_blades))
                    quad  = blade_quad(pj, u, v, args.prop_radius, args.prop_chord, theta)
                    poly.set_verts([quad])
                    poly.set_facecolor(color_face)
                    poly.set_edgecolor("none")
                    poly.set_alpha(args.prop_alpha)

                # ===== 항상 수평 라벨 (고장 모터만 표시) =====
                label = fault_texts[i][j]
                if is_fault:
                    # 3D→2D 투영 좌표
                    x2, y2, _ = proj3d.proj_transform(pj[0], pj[1], pj[2], ax3d.get_proj())
                    # 약간 오른쪽 위로 오프셋 (화면 좌표상)
                    label.set_text(f"Link{i+1} Motor{j+1} Stopped")
                    label.set_position((x2 + 0.005, y2 + 0.005))  # 화면(axes data)에서 작은 오프셋
                    label.set_color("r")
                    label.set_alpha(1.0)
                    label.set_visible(True)
                    label.set_transform(ax3d.transData)  # 화면 기준으로 항상 수평
                else:
                    label.set_visible(False)

        # 우측 막대
        for r in range(rows):
            m0 = r * motors_per_link
            bars_gt  = bars_gt_rows[r]
            bars_pred= bars_pred_rows[r]
            gt_txts  = gt_texts_rows[r]
            for i_m in range(motors_per_link):
                mi = m0 + i_m
                bars_gt[i_m].set_height(float(gt_fault[mi]))
                bars_pred[i_m].set_height(float(prob_last[mi]))
                is_alarm = bool(predk_last[mi])
                bars_pred[i_m].set_edgecolor("r" if is_alarm else "black")
                bars_pred[i_m].set_linewidth(2.5 if is_alarm else 0.5)
                gt_txts[i_m].set_text("GT:F" if gt_fault[mi] >= 0.5 else "")

        t_real = t / max(1e-6, args.data_hz)
        alarms_total = int(sum(predk_last))
        txt.set_text(
            f"          t = {t_real:4.2f}s"
        )

        t_idx[0] += 1

        # 반환 아티스트
        artists = desired_lines + actual_lines + [desired_nodes, actual_nodes, base_marker, base_text, txt]
        for i in range(link_count):
            artists.extend(stems_lines[i])
            for patches in blade_patches[i]:
                artists.extend(patches)
            # fault_texts는 아티스트 리스트에 굳이 넣지 않아도 화면에 유지됨
        for r in range(rows):
            artists.extend(list(bars_gt_rows[r]))
            artists.extend(list(bars_pred_rows[r]))
            artists.extend(gt_texts_rows[r])
        return artists

    # ===== 애니메이션 =====
    ani = FuncAnimation(
        fig, update,
        interval=interval_ms,
        blit=False,
        save_count=T,
        cache_frame_data=False
    )
    plt.tight_layout()

    if args.save_video:
        ext = os.path.splitext(args.out)[1].lower()
        try:
            if ext in [".mp4", ".m4v", ".mov"]:
                writer = FFMpegWriter(fps=args.video_fps, codec=args.codec, bitrate=bitrate_kbps)
            elif ext in [".gif"]:
                writer = PillowWriter(fps=args.video_fps)
            else:
                raise ValueError(f"지원하지 않는 확장자: {ext} (mp4/gif 사용)")
            print(f"💾 Saving video to: {args.out}  (fps={args.video_fps}, dpi={args.dpi})")
            ani.save(args.out, writer=writer, dpi=args.dpi)
            print("✅ Done.")
        except Exception as e:
            print(f"❌ Video save failed: {e}")
        finally:
            plt.close(fig)
    else:
        plt.show()

if __name__ == "__main__":
    main()






"""
디버깅용 터미널 입력
python3 causal_tcn/visualize_stream_causal_tcn.py \
  --ckpt TCN/TCN_link_3_RELonly_CAUSAL_best.pth \
  --npz  data_storage/link_3/fault_dataset.npz \
  --seq_idx 100 \
  --threshold 0.5 \
  --kofn 3,5 \
  --data_hz 100 --speed 1.0 \
  --prepend_base 1 --fix_origin 1 \
  --label_fault_is_one 1 \
  --motors_per_link 8 \
  --anchor_ratio 0.15 \
  --arm_len 0.15 \
  --prop_blades 4 --prop_radius 0.08 --prop_chord 0.028 --prop_alpha 0.85
"""


""" MP4저장
python3 causal_tcn/visualize_stream_causal_tcn.py \
  --ckpt TCN/TCN_link_1_RELonly_CAUSAL.pth \
  --npz  data_storage/link_1/fault_dataset.npz \
  --seq_idx 100 \
  --threshold 0.5 --kofn 3,5 \
  --data_hz 100 --speed 3.33 \
  --prepend_base 1 --fix_origin 1 \
  --label_fault_is_one 1 \
  --motors_per_link 8 \
  --anchor_ratio 0.15 \
  --arm_len 0.15 \
  --prop_blades 4 --prop_radius 0.08 --prop_chord 0.028 --prop_alpha 0.85 \
  --save_video 1 \
  --out data_storage/link_1/vis.mp4 \
  --video_fps 30 --codec libx264 --bitrate 6000k --dpi 150
"""