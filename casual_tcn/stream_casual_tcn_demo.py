# stream_causal_tcn_demo.py
import os, time, argparse, csv, random
import numpy as np
import torch
import torch.nn as nn

# ====================== (A) TCN 모델 정의 (학습과 동일) ======================
class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        padding = (kernel_size - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.remove = padding
    def forward(self, x):  # x: [B,C,T]
        y = super().forward(x)
        if self.remove > 0:
            y = y[:, :, :-self.remove]
        return y

class TemporalBlock(nn.Module):
    def __init__(self, ch_in, ch_out, k=3, d=1, drop=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(ch_in,  ch_out, k, dilation=d)
        self.conv2 = CausalConv1d(ch_out, ch_out, k, dilation=d)
        self.act   = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout(drop)
        self.down  = nn.Conv1d(ch_in, ch_out, 1) if ch_in != ch_out else nn.Identity()
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity="relu")
        if isinstance(self.down, nn.Conv1d):
            nn.init.kaiming_uniform_(self.down.weight, nonlinearity="linear")
    def forward(self, x):  # [B,C,T]
        y = self.act(self.conv1(x)); y = self.drop(y)
        y = self.act(self.conv2(y)); y = self.drop(y)
        return self.act(y + self.down(x))

class FaultDiagnosisTCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=128, layers=6, k=3, dropout=0.1):
        super().__init__()
        chans = [input_dim] + [hidden]*layers
        dilations = [2**i for i in range(layers)]
        blocks = []
        for i in range(layers):
            blocks.append(TemporalBlock(chans[i], chans[i+1], k=k, d=dilations[i], drop=dropout))
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Conv1d(chans[-1], chans[-1], 1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(chans[-1], output_dim, 1)
        )
    def forward(self, x):           # x: [B,T,D]
        x = x.transpose(1,2)        # -> [B,D,T]
        h = self.tcn(x)             # -> [B,H,T]
        y = self.head(h)            # -> [B,M,T]
        return y.transpose(1,2)     # -> [B,T,M]

def receptive_field(layers:int, k:int) -> int:
    # R ≈ 1 + (k-1)*(2^layers - 1)  (dilations: 1,2,4,...)
    return 1 + (k-1)*(2**layers - 1)

# ====================== (B) 스트리머: 한 프레임씩 추론 ======================
from collections import deque

class KofN:
    """ 최근 N프레임 중 K프레임 이상 양성일 때만 True (오경보 억제) """
    def __init__(self, M:int, K:int=3, N:int=5):
        self.K, self.N, self.M = K, N, M
        self.buf = deque(maxlen=N)  # 각 step에서 (M,) bool
    def step(self, pred_bool_m):    # pred_bool_m: (M,) bool/uint8
        self.buf.append(pred_bool_m.astype(np.uint8))
        arr = np.stack(self.buf, axis=0)          # [n,M]
        vote = (arr.sum(axis=0) >= self.K)        # [M]
        return vote

class Streamer:
    def __init__(self, model, mu, std, device, lookback=None, threshold=0.5, kofn=None):
        self.model = model.eval()
        self.device = device
        self.mu = torch.as_tensor(mu, dtype=torch.float32, device=device)
        self.std= torch.as_tensor(std, dtype=torch.float32, device=device)
        self.D = self.mu.numel()
        self.M = model.head[-1].out_channels  # last conv out ch
        self.threshold = threshold
        self.lookback = lookback
        self.buf = deque()  # 최신 X 프레임(raw, torch (D,)) 보관
        self.kofn = kofn

    def step(self, x_t_np: np.ndarray):
        """ x_t_np: (D,) numpy vector (raw feature at time t) """
        assert x_t_np.shape[-1] == self.D, f"expected D={self.D}, got {x_t_np.shape}"
        x_t = torch.from_numpy(x_t_np.astype(np.float32)).to(self.device)
        self.buf.append(x_t)
        if (self.lookback is not None) and (len(self.buf) > self.lookback):
            self.buf.popleft()

        # make window tensor [1,L,D]
        x_list = list(self.buf)
        X = torch.stack(x_list, dim=0)                        # [L,D]
        Xn = (X - self.mu) / self.std
        Xn = Xn.unsqueeze(0)                                  # [1,L,D]

        with torch.no_grad():
            logits = self.model(Xn)                           # [1,L,M]
            prob = torch.sigmoid(logits[0, -1])               # [M] only current frame
        pred = (prob >= self.threshold).detach().cpu().numpy().astype(np.uint8)  # [M]
        if self.kofn is not None:
            pred_k = self.kofn.step(pred)
        else:
            pred_k = pred
        return prob.detach().cpu().numpy(), pred, pred_k, len(self.buf)  # probs, raw, K-of-N, Lwin

# ====================== (C) 데이터 로딩 (임의/NPY/CSV/NPZ) ======================
def load_series_from_npy(npy_path:str):
    X = np.load(npy_path)
    assert X.ndim == 2, "Expected shape [T,D]"
    return X

def load_series_from_csv(csv_path:str):
    rows = []
    with open(csv_path, "r", newline="") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row: continue
            rows.append([float(x) for x in row])
    X = np.array(rows, dtype=np.float32)
    assert X.ndim == 2, "Expected shape [T,D]"
    return X

# ── 피처 빌더 (LEGACY/REL-only 대응) ──
def _vee_skew(A):
    return np.stack([A[...,2,1]-A[...,1,2],
                     A[...,0,2]-A[...,2,0],
                     A[...,1,0]-A[...,0,1]], axis=-1)/2.0
def _so3_log(Rm):
    tr = np.clip((np.einsum('...ii', Rm) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(tr)
    A = Rm - np.swapaxes(Rm, -1, -2)
    v = _vee_skew(A)
    sin_th = np.sin(theta); eps = 1e-9
    scale = np.where(np.abs(sin_th)[...,None] > eps, (theta/(sin_th+eps))[...,None], 1.0)
    w = v * scale
    return np.where((theta < 1e-6)[...,None], v, w)
def _rel_log_increment(R):
    Tdim = R.shape[-3]
    out = np.zeros(R.shape[:-2] + (3,), dtype=R.dtype)
    if Tdim > 1:
        R_prev = R[..., :-1, :, :]
        R_next = R[..., 1:, :, :]
        R_rel  = np.matmul(np.swapaxes(R_prev, -1, -2), R_next)
        out[..., 1:, :] = _so3_log(R_rel)
    return out
def _time_diff(x):
    d = np.zeros_like(x)
    if x.shape[-3] > 1:
        d[..., 1:, :] = x[..., 1:, :] - x[..., :-1, :]
    return d
def _flatten_3x4(T):
    return T[..., :3, :4].reshape(*T.shape[:-2], 12)
def build_features_rel_only(d_rel, a_rel):
    S, T, L = d_rel.shape[:3]
    des_12 = _flatten_3x4(d_rel)
    act_12 = _flatten_3x4(a_rel)
    p_des, R_des = d_rel[..., :3, 3], d_rel[..., :3, :3]
    p_act, R_act = a_rel[..., :3, 3], a_rel[..., :3, :3]
    p_err  = p_act - p_des
    def _rot_err_vec(R_des, R_act):
        R_rel = np.matmul(np.swapaxes(R_des, -1, -2), R_act)
        return _so3_log(R_rel)
    r_err  = _rot_err_vec(R_des, R_act)
    dp_des = _time_diff(p_des); dp_act = _time_diff(p_act)
    R_des_SK = np.swapaxes(R_des, 1, 2); R_act_SK = np.swapaxes(R_act, 1, 2)
    dr_des_SK = _rel_log_increment(R_des_SK); dr_act_SK = _rel_log_increment(R_act_SK)
    dr_des = np.swapaxes(dr_des_SK, 1, 2);  dr_act = np.swapaxes(dr_act_SK, 1, 2)
    feats = np.concatenate([des_12, act_12, p_err, r_err, dp_des, dp_act, dr_des, dr_act], axis=-1)
    S_, T_, L_, _ = feats.shape
    return feats.reshape(S_, T_, L_*42).astype(np.float32)
def build_features_legacy(desired, actual):
    S, T = desired.shape[:2]
    des_12 = desired[:, :, :3, :4].reshape(S, T, 12)
    act_12 = actual[:,  :, :3, :4].reshape(S, T, 12)
    p_des, R_des = desired[..., :3, 3], desired[..., :3, :3]
    p_act, R_act = actual[...,  :3, 3], actual[...,  :3, :3]
    def _rot_err_vec(R_des, R_act):
        R_rel = np.matmul(np.swapaxes(R_des, -1, -2), R_act)
        return _so3_log(R_rel)
    p_err  = p_act - p_des
    r_err  = _rot_err_vec(R_des, R_act)
    dp_des = _time_diff(p_des); dp_act = _time_diff(p_act)
    dr_des = _rel_log_increment(R_des); dr_act = _rel_log_increment(R_act)
    X = np.concatenate([des_12, act_12, p_err, r_err, dp_des, dp_act, dr_des, dr_act], axis=2).astype(np.float32)
    return X
def load_series_from_npz(npz_path:str, seq_idx:int=0):
    d = np.load(npz_path, allow_pickle=True)
    keys = set(d.files)
    if {"desired_link_rel","actual_link_rel","desired_link_cum","actual_link_cum","label"}.issubset(keys):
        d_rel = d["desired_link_rel"]; a_rel = d["actual_link_rel"]
        labels = d["label"]
        X = build_features_rel_only(d_rel, a_rel)  # (S,T,D)
        Y = (1.0 - labels).astype(np.float32)      # (S,T,M) 1=fault
    elif {"desired","actual","label"}.issubset(keys):
        desired = d["desired"]; actual = d["actual"]; labels = d["label"]
        X = build_features_legacy(desired, actual) # (S,T,D)
        Y = (1.0 - labels).astype(np.float32)
    else:
        raise KeyError(f"Unsupported .npz schema. keys={sorted(keys)}")
    S = X.shape[0]; assert 0 <= seq_idx < S
    return X[seq_idx], Y[seq_idx]  # (T,D), (T,M)

# ====================== (D) 메인: 실시간 데모 루프 ======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="학습 저장파일(.pth)")
    ap.add_argument("--mode", choices=["rand","npy","csv","npz"], default="rand")
    ap.add_argument("--path", type=str, help="mode가 npy/csv/npz일 때 파일 경로")
    ap.add_argument("--seq_idx", type=int, default=0, help="npz에서 사용할 시퀀스 인덱스")
    ap.add_argument("--T", type=int, default=1000, help="rand 모드 길이")
    ap.add_argument("--sleep_ms", type=int, default=0, help="각 프레임 사이 대기(ms) (실시간 흉내)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--kofn", type=str, default="0,0", help="K,N (예: 3,5). 0,0이면 미사용")
    ap.add_argument("--override_lookback", type=int, default=0, help="0이면 자동(R), 아니면 수동")
    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42, help="재현성 보장을 위한 난수 시드")
    args = ap.parse_args()

    # ── 시드 고정 ──
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ── 장치 선택 ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("📥 device:", device)

    # ── 체크포인트 로드 ──
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    D = int(ckpt["input_dim"]); M = int(ckpt["M"])
    cfg = ckpt["cfg"]
    hidden = int(cfg.get("hidden", 128))
    layers = int(cfg.get("layers", 6))
    ksize  = int(cfg.get("k", 3))
    dropout= float(cfg.get("dropout", 0.1))

    model = FaultDiagnosisTCN(input_dim=D, output_dim=M, hidden=hidden, layers=layers, k=ksize, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    mu = ckpt["train_mean"]; std = ckpt["train_std"]
    R_auto = receptive_field(layers, ksize)
    lookback = args.override_lookback if args.override_lookback > 0 else R_auto
    print(f"🧮 receptive_field R≈{R_auto} | use lookback={lookback} | threshold={args.threshold}")

    # ── 데이터 준비 ──
    if args.mode == "rand":
        T = args.T
        X = np.random.randn(T, D).astype(np.float32) * 1.0
        Y = None
    elif args.mode == "npy":
        X = load_series_from_npy(args.path)
        assert X.shape[1] == D, f"npy feature dim {X.shape[1]} != ckpt D {D}"
        Y = None
    elif args.mode == "csv":
        X = load_series_from_csv(args.path)
        assert X.shape[1] == D, f"csv feature dim {X.shape[1]} != ckpt D {D}"
        Y = None
    elif args.mode == "npz":
        X, Y = load_series_from_npz(args.path, seq_idx=args.seq_idx)
        assert X.shape[1] == D, f"npz feature dim {X.shape[1]} != ckpt D {D}"
        if Y.shape[1] != M:
            raise ValueError(f"npz label dim {Y.shape[1]} != ckpt M {M}")
        print(f"🎯 using NPZ sequence idx={args.seq_idx} | T={X.shape[0]}")

    # ── 스트리머 구성 ──
    if args.kofn != "0,0":
        K,N = map(int, args.kofn.split(","))
        kofn = KofN(M=M, K=K, N=N)
        print(f"🛡️  K-of-N smoothing: K={K}, N={N}")
    else:
        kofn = None

    streamer = Streamer(model, mu, std, device, lookback=lookback, threshold=args.threshold, kofn=kofn)

    # ── 실시간 루프 ──
    T = X.shape[0]; sleep_s = args.sleep_ms/1000.0
    hit_exact = 0
    for t in range(T):
        probs, pred_raw, pred_k, Lwin = streamer.step(X[t])

        # 현재 프레임의 고장 예측 개수
        n_pos_raw = int(pred_raw.sum())
        n_pos_k   = int(pred_k.sum())
        line = f"[t={t:04d} | Lwin={Lwin:3d}] pos_raw={n_pos_raw:3d} pos_k={n_pos_k:3d}"

        # 레이블 있으면 Exact-All 일치도 출력
        if Y is not None:
            gt = Y[t].astype(np.uint8)
            exact_raw = int((pred_raw == gt).all())
            exact_k   = int((pred_k   == gt).all())
            line += f" | exact_raw={exact_raw} exact_k={exact_k}"
            hit_exact += exact_k

        print(line)
        if sleep_s > 0: time.sleep(sleep_s)

    if Y is not None:
        print(f"✅ stream Exact-All (K-of-N 기준) = {hit_exact / T:.4f}")

if __name__ == "__main__":
    main()
