# fault_detect/infer.py
from __future__ import annotations
import os, glob, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import LASDRAFaultDataset
from .model import LinkTemporalModel
from .postprocess import ewma, hysteresis, first_onset
from .utils import compute_timewise_metrics, onset_mae

@torch.no_grad()
def run_infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = LASDRAFaultDataset(args.data_dir, split="val", win_size=999999)  # 전체 시퀀스
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = LinkTemporalModel(in_dim=36, hidden=args.hidden, nheads=8, nlayers=2, dropout=0.0).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    os.makedirs(args.save_dir, exist_ok=True)

    all_true_flat = []
    all_prob_flat = []

    shard_idx = 0
    for i, batch in enumerate(tqdm(dl, desc="Infer")):
        x = batch["x"].to(device)   # (1,T,F) with full seq (val dataset 구성 상)
        y = batch["y"].numpy()[0]   # (T,8) fault target
        logits = model(x)
        prob = torch.sigmoid(logits).cpu().numpy()[0]  # (T,8)

        # postprocess
        prob_s = ewma(prob, alpha=args.ewma_alpha)
        dec = hysteresis(prob_s, th_on=args.th_on, th_off=args.th_off)
        onset_pred = first_onset(dec)

        # 저장
        out = {
            "prob": prob,
            "prob_s": prob_s,
            "decision": dec.astype(np.uint8),
            "onset_pred": onset_pred.astype(np.int32),
        }
        np.savez_compressed(os.path.join(args.save_dir, f"sample_{i:05d}.npz"), **out)

        all_true_flat.append(y)
        all_prob_flat.append(prob)

    Y = np.concatenate(all_true_flat, axis=0)  # (N*T,8) after reshape below
    P = np.concatenate(all_prob_flat, axis=0)
    metrics = compute_timewise_metrics(Y, P, threshold=args.eval_threshold)
    print(f"[VAL] bacc={metrics['bacc']:.4f} f1={metrics['f1']:.4f} auroc={metrics['auroc']:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--save_dir", type=str, default="predictions")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--ewma_alpha", type=float, default=0.2)
    ap.add_argument("--th_on", type=float, default=0.6)
    ap.add_argument("--th_off", type=float, default=0.4)
    ap.add_argument("--eval_threshold", type=float, default=0.5)
    args = ap.parse_args()
    run_infer(args)
