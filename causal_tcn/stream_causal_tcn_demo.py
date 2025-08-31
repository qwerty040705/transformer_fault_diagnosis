# stream_causal_tcn_demo.py
import os, time, argparse, csv, random, warnings
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
    return 1 + (k-1)*(2**layers - 1)

# ====================== (B) 스트리머 ======================
from collections import deque

class KofN:
    """최근 N프레임 중 K프레임 이상 양성일 때만 True"""
    def __init__(self, M:int, K:int=3, N:int=5):
        self.K, self.N, self.M = K, N, M
        self.buf = deque(maxlen=N)
    def step(self, pred_bool_m):
        self.buf.append(pred_bool_m.astype(np.uint8))
        arr = np.stack(self.buf, axis=0)
        return (arr.sum(axis=0) >= self.K)

class Streamer:
    def __init__(self, model, mu, std, device, lookback=None, threshold=0.5, kofn=None):
        self.model = model.eval()
        self.device = device
        self.mu = torch.as_tensor(mu, dtype=torch.float32, device=device)
        self.std= torch.as_tensor(std, dtype=torch.float32, device=device)
        self.D = self.mu.numel()
        self.M = model.head[-1].out_channels
        self.threshold = threshold
        self.lookback = lookback
        self.buf = deque()
        self.kofn = kofn

    def step(self, x_t_np: np.ndarray):
        assert x_t_np.shape[-1] == self.D, f"expected D={self.D}, got {x_t_np.shape}"
        x_t = torch.from_numpy(x_t_np.astype(np.float32)).to(self.device)
        self.buf.append(x_t)
        if (self.lookback is not None) and (len(self.buf) > self.lookback):
            self.buf.popleft()
        X = torch.stack(list(self.buf), dim=0)      # [L,D]
        Xn = ((X - self.mu) / self.std).unsqueeze(0)  # [1,L,D]
        with torch.no_grad():
            logits = self.model(Xn)                 # [1,L,M]
            prob = torch.sigmoid(logits[0, -1])     # [M]
        pred = (prob >= self.threshold).detach().cpu().numpy().astype(np.uint8)
        pred_k = self.kofn.step(pred) if self.kofn is not None else pred
        return prob.detach().cpu().numpy(), pred, pred_k, len(self.buf)

# ====================== (C) 데이터 로딩 ======================
def load_series_from_npy(npy_path:str):
    X = np.load(npy_path); assert X.ndim == 2
    return X

def load_series_from_csv(csv_path:str):
    rows = []
    with open(csv_path, "r", newline="") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row: continue
            rows.append([float(x) for x in row])
    X = np.array(rows, dtype=np.float32); assert X.ndim == 2
    return X

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

def load_all_from_npz(npz_path:str):
    d = np.load(npz_path, allow_pickle=True)
    keys = set(d.files)
    if {"desired_link_rel","actual_link_rel","desired_link_cum","actual_link_cum","label"}.issubset(keys):
        d_rel = d["desired_link_rel"]; a_rel = d["actual_link_rel"]
        labels = d["label"]
        X = build_features_rel_only(d_rel, a_rel)
        Y = (1.0 - labels).astype(np.float32)
    elif {"desired","actual","label"}.issubset(keys):
        desired = d["desired"]; actual = d["actual"]; labels = d["label"]
        X = build_features_legacy(desired, actual)
        Y = (1.0 - labels).astype(np.float32)
    else:
        raise KeyError(f"Unsupported .npz schema. keys={sorted(keys)}")
    return X, Y

def load_series_from_npz(npz_path:str, seq_idx:int=0):
    X, Y = load_all_from_npz(npz_path)
    S = X.shape[0]; assert 0 <= seq_idx < S
    return X[seq_idx], Y[seq_idx]

# ====================== (D) 메트릭 & ETA 유틸 ======================
def segments_from_binary(b):
    T = len(b); segs = []; in_run=False; s=0
    for t in range(T):
        if b[t] and not in_run: in_run=True; s=t
        elif (not b[t]) and in_run: in_run=False; segs.append((s,t))
    if in_run: segs.append((s,T))
    return segs

def pr_auc_average_precision(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    tp = labels_sorted.cumsum()
    fp = np.cumsum(1 - labels_sorted)
    prec = tp / np.maximum(tp + fp, 1)
    total_pos = max(labels.sum(), 1)
    ap = (prec * labels_sorted).sum() / total_pos
    return ap

def safe_div(a, b): return (a / b) if b > 0 else 0.0

def _fmt_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    m, s = divmod(int(sec + 0.5), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def eval_sequence(model, mu, std, device, X, Y, threshold=0.5, kofn=None, lookback=None, dt_ms=10,
                  print_stream=False, show_eta=False, print_every=10, seq_name=None):
    streamer = Streamer(model, mu, std, device, lookback=lookback, threshold=threshold, kofn=kofn)
    T, D = X.shape; M = Y.shape[1]
    hit_exact = 0
    y_true_any = []; y_pred_any = []; score_any = []
    t0 = time.time()
    for t in range(T):
        probs, pred_raw, pred_k, Lwin = streamer.step(X[t])
        gt = Y[t].astype(np.uint8)
        exact_k = int((pred_k == gt).all()); hit_exact += exact_k
        any_true = int(gt.any()); any_pred = int(pred_k.any())
        y_true_any.append(any_true); y_pred_any.append(any_pred)
        score_any.append(float(probs.max()))
        want_line = print_stream
        want_eta  = show_eta and ((t+1) % max(1, print_every) == 0 or (t+1)==T)
        if want_line:
            n_pos_raw = int(pred_raw.sum()); n_pos_k = int(pred_k.sum())
            line = f"[t={t:04d} | Lwin={Lwin:3d}] pos_raw={n_pos_raw:3d} pos_k={n_pos_k:3d} | exact_k={exact_k}"
            if want_eta:
                elapsed = time.time() - t0; done = t + 1
                fps = done / max(elapsed, 1e-9); eta = (T - done) / max(fps, 1e-9); p = 100.0 * done / T
                tag = f" | {seq_name}" if seq_name else ""
                line += f"{tag} | {p:5.1f}% | elapsed={_fmt_time(elapsed)} | ETA={_fmt_time(eta)} | FPS={fps:.1f}"
            print(line)
        elif want_eta:
            elapsed = time.time() - t0; done = t + 1
            fps = done / max(elapsed, 1e-9); eta = (T - done) / max(fps, 1e-9); p = 100.0 * done / T
            tag = f"[{seq_name}] " if seq_name else ""
            print(f"{tag}progress {p:5.1f}% | elapsed={_fmt_time(elapsed)} | ETA={_fmt_time(eta)} | FPS={fps:.1f}")
    y_true_any = np.array(y_true_any, dtype=np.uint8)
    y_pred_any = np.array(y_pred_any, dtype=np.uint8)
    score_any  = np.array(score_any, dtype=np.float32)
    exact_all = hit_exact / T
    tp = int(((y_true_any == 1) & (y_pred_any == 1)).sum())
    fp = int(((y_true_any == 0) & (y_pred_any == 1)).sum())
    fn = int(((y_true_any == 1) & (y_pred_any == 0)).sum())
    prec = safe_div(tp, tp + fp); rec  = safe_div(tp, tp + fn)
    f1   = safe_div(2*prec*rec, prec+rec) if (prec+rec)>0 else 0.0
    auprc = pr_auc_average_precision(score_any, y_true_any)
    gt_segs = segments_from_binary(y_true_any); pr_segs = segments_from_binary(y_pred_any)
    tp_e=fp_e=fn_e=0; latencies_ms=[]; pr_used=set()
    for (gs, ge) in gt_segs:
        det_idx=None
        for t in range(gs, ge):
            if y_pred_any[t]==1: det_idx=t; break
        if det_idx is not None:
            tp_e += 1; latencies_ms.append((det_idx-gs)*dt_ms)
            for i,(ps,pe) in enumerate(pr_segs):
                if ps<=det_idx<pe: pr_used.add(i); break
        else:
            fn_e += 1
    for i in range(len(pr_segs)):
        if i not in pr_used: fp_e += 1
    ev_prec = safe_div(tp_e, tp_e + fp_e); ev_rec = safe_div(tp_e, tp_e + fn_e)
    lat_mean = float(np.mean(latencies_ms)) if len(latencies_ms)>0 else 0.0
    lat_std  = float(np.std(latencies_ms))  if len(latencies_ms)>0 else 0.0
    return {
        "frame_exact_all": float(exact_all),
        "frame_any_f1": float(f1),
        "frame_auprc": float(auprc),
        "event_recall": float(ev_rec),
        "event_precision": float(ev_prec),
        "event_latency_mean_ms": lat_mean,
        "event_latency_std_ms": lat_std,
        "n_events": len(gt_segs),
        "n_pred_events": len(pr_segs),
    }

# ====================== (E) 메인 ======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--mode", choices=["rand","npy","csv","npz"], default="rand")
    ap.add_argument("--path", type=str)
    ap.add_argument("--seq_idx", type=int, default=0, help="-1=전체, 0~S-1=단일")
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--sleep_ms", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--kofn", type=str, default="0,0")  # "K,N"
    ap.add_argument("--override_lookback", type=int, default=0)
    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dt_ms", type=float, default=10.0)
    ap.add_argument("--quiet_stream", action="store_true")
    ap.add_argument("--show_eta", action="store_true")
    ap.add_argument("--sample_n", type=int, default=0, help="전체(-1) 모드에서 샘플 개수(중복 없이)")
    args = ap.parse_args()

    # 시드
    np.random.seed(args.seed); torch.manual_seed(args.seed); random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

    # 장치
    if torch.cuda.is_available(): device=torch.device("cuda")
    elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available(): device=torch.device("mps")
    else: device=torch.device("cpu")
    print("📥 device:", device)

    # 체크포인트 로드 (weights_only 안전 로드 → 실패 시 fallback)
    import torch.serialization as ts
    try:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    except Exception:
        try:
            if hasattr(ts, "add_safe_globals"):
                import numpy as _np
                ts.add_safe_globals([_np._core.multiarray._reconstruct])
        except Exception:
            pass
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    D = int(ckpt["input_dim"]); M = int(ckpt["M"])
    cfg = ckpt["cfg"]; hidden=int(cfg.get("hidden",128))
    layers=int(cfg.get("layers",6)); ksize=int(cfg.get("k",3)); dropout=float(cfg.get("dropout",0.1))
    model = FaultDiagnosisTCN(D, M, hidden=hidden, layers=layers, k=ksize, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True); model.eval()
    mu = ckpt["train_mean"]; std = ckpt["train_std"]
    R_auto = receptive_field(layers, ksize)
    lookback = args.override_lookback if args.override_lookback>0 else R_auto
    print(f"🧮 receptive_field R≈{R_auto} | use lookback={lookback} | threshold={args.threshold}")

    # K-of-N
    if args.kofn != "0,0":
        K,N = map(int, args.kofn.split(",")); kofn = KofN(M=M, K=K, N=N)
        print(f"🛡️  K-of-N smoothing: K={K}, N={N}")
    else:
        kofn = None

    # 데이터 & 실행
    if args.mode == "rand":
        T = args.T
        X = np.random.randn(T, D).astype(np.float32)
        Y = np.zeros((T, M), dtype=np.float32)
        res = eval_sequence(model, mu, std, device, X, Y, args.threshold, kofn, lookback,
                            dt_ms=args.dt_ms, print_stream=not args.quiet_stream,
                            show_eta=args.show_eta, print_every=args.print_every, seq_name="rand")
        print("\n================ STREAM METRICS ================")
        print(f"Frame  - Exact-All: {res['frame_exact_all']:.4f} | AnyFault-F1: {res['frame_any_f1']:.4f} | AUPRC: {res['frame_auprc']:.4f}")
        print(f"Event  - Recall: {res['event_recall']:.4f} | Precision: {res['event_precision']:.4f} | Latency: {res['event_latency_mean_ms']:.1f}±{res['event_latency_std_ms']:.1f} ms")
        print("Note   - Metrics computed with K-of-N predictions.")
        print("===============================================\n")
        return

    elif args.mode in ["npy","csv"]:
        X = load_series_from_npy(args.path) if args.mode=="npy" else load_series_from_csv(args.path)
        assert X.shape[1] == D
        _ = eval_sequence(model, mu, std, device, X, np.zeros((X.shape[0], M), dtype=np.float32),
                          args.threshold, kofn, lookback, dt_ms=args.dt_ms,
                          print_stream=not args.quiet_stream, show_eta=args.show_eta,
                          print_every=args.print_every, seq_name=os.path.basename(args.path))
        return

    elif args.mode == "npz":
        if args.seq_idx == -1:
            Xall, Yall = load_all_from_npz(args.path)
            assert Xall.shape[2] == D and Yall.shape[2] == M
            S = Xall.shape[0]

            # === 샘플 평가 ===
            if args.sample_n > 0:
                n = min(args.sample_n, S)
                rng = np.random.RandomState(args.seed)  # 시드 42로 고정 가능
                idxs = rng.choice(S, size=n, replace=False)
                print(f"🎯 NPZ 샘플 평가: S={S} 중 {n}개 샘플 (seed={args.seed})")
            else:
                idxs = np.arange(S)
                print(f"🎯 NPZ 전체 평가: S={S} sequences")

            accs=[]; f1s=[]; aups=[]; er=[]; ep=[]; lats=[]; latstd=[]; nevs=[]; npreds=[]
            t0 = time.time()
            for i, s in enumerate(idxs):
                name = f"seq{s:03d}"
                res = eval_sequence(model, mu, std, device, Xall[s], Yall[s],
                                    args.threshold, kofn, lookback, dt_ms=args.dt_ms,
                                    print_stream=False, show_eta=False,
                                    print_every=args.print_every, seq_name=name)
                accs.append(res["frame_exact_all"]); f1s.append(res["frame_any_f1"]); aups.append(res["frame_auprc"])
                er.append(res["event_recall"]); ep.append(res["event_precision"])
                lats.append(res["event_latency_mean_ms"]); latstd.append(res["event_latency_std_ms"])
                nevs.append(res["n_events"]); npreds.append(res["n_pred_events"])

                if args.show_eta:
                    elapsed = time.time() - t0
                    done = i + 1; total = len(idxs)
                    seqps = done / max(elapsed, 1e-9); eta = (total - done) / max(seqps, 1e-9); p = 100.0 * done / total
                    print(f"[dataset] {p:5.1f}% | elapsed={_fmt_time(elapsed)} | ETA={_fmt_time(eta)} | seq/s={seqps:.2f}")

            def mean(x): return float(np.mean(x)) if len(x)>0 else 0.0
            picked = f"(picked idx: {list(map(int, idxs))[:10]}{'...' if len(idxs)>10 else ''})" if args.sample_n>0 else ""
            print("\n================ DATASET AVERAGE METRICS ================")
            print(f"Frame  - Exact-All: {mean(accs):.4f} | AnyFault-F1: {mean(f1s):.4f} | AUPRC: {mean(aups):.4f}")
            print(f"Event  - Recall: {mean(er):.4f} | Precision: {mean(ep):.4f} | Latency: {mean(lats):.1f}±{mean(latstd):.1f} ms")
            print(f"Counts - #GT events/seq: {mean(nevs):.2f} | #Pred events/seq: {mean(npreds):.2f}")
            if args.sample_n>0: print(f"Note   - {args.sample_n} sequences sampled {picked}")
            print("Note   - Metrics computed with K-of-N predictions.")
            print("=========================================================\n")
            return
        else:
            X, Y = load_series_from_npz(args.path, seq_idx=args.seq_idx)
            assert X.shape[1] == D and Y.shape[1] == M
            print(f"🎯 using NPZ sequence idx={args.seq_idx} | T={X.shape[0]}")
            res = eval_sequence(model, mu, std, device, X, Y, args.threshold, kofn, lookback,
                                dt_ms=args.dt_ms, print_stream=not args.quiet_stream,
                                show_eta=args.show_eta, print_every=args.print_every,
                                seq_name=f"seq{args.seq_idx:03d}")
            print("\n================ STREAM METRICS ================")
            print(f"Frame  - Exact-All: {res['frame_exact_all']:.4f} | AnyFault-F1: {res['frame_any_f1']:.4f} | AUPRC: {res['frame_auprc']:.4f}")
            print(f"Event  - Recall: {res['event_recall']:.4f} | Precision: {res['event_precision']:.4f} | Latency: {res['event_latency_mean_ms']:.1f}±{res['event_latency_std_ms']:.1f} ms")
            print("Note   - Metrics computed with K-of-N predictions.")
            print("===============================================\n")
            return

if __name__ == "__main__":
    main()








"""
python3 causal_tcn/stream_causal_tcn_demo.py \
  --ckpt TCN/TCN_link_2_RELonly_CAUSAL.pth \
  --mode npz --path data_storage/link_2/fault_dataset.npz \
  --seq_idx -1 --kofn 3,5 --threshold 0.5 \
  --show_eta --print_every 25
  """


"""
python3 causal_tcn/stream_causal_tcn_demo.py \
  --ckpt TCN/TCN_link_2_RELonly_CAUSAL.pth \
  --mode npz --path data_storage/link_2/fault_dataset.npz \
  --seq_idx -1 --sample_n 100 \
  --kofn 3,5 --threshold 0.5 --dt_ms 10 \
  --show_eta
"""