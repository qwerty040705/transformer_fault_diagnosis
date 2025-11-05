# fault_detect/train.py
from __future__ import annotations
import os, argparse, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import LASDRAFaultDataset
from .model import LinkTemporalModel
from .loss import WeightedBCELoss, onset_weighted_bce, total_variation_loss
from .utils import compute_timewise_metrics

def estimate_pos_weight(loader, max_batches=50, device="cuda"):
    pos = 0
    neg = 0
    n_batches = 0
    for batch in loader:
        y = batch["y"].to(device)  # (B,T,8) fault target
        pos += y.sum().item()
        neg += (1.0 - y).sum().item()
        n_batches += 1
        if n_batches >= max_batches:
            break
    pos = max(pos, 1.0)
    neg = max(neg, 1.0)
    return neg / pos  # pos_weight

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    ds_train = LASDRAFaultDataset(args.data_dir, split="train", win_size=args.win_size)
    ds_val   = LASDRAFaultDataset(args.data_dir, split="val",   win_size=args.win_size)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=args.workers>0)
    dl_val   = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, pin_memory=True, drop_last=False, persistent_workers=args.workers>0)

    model = LinkTemporalModel(in_dim=36, hidden=args.hidden, nheads=8, nlayers=2, dropout=args.dropout).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    ema = EMA(model, decay=0.999)

    # 불균형 가중 추정
    print("Estimating class pos_weight ...")
    pos_weight = estimate_pos_weight(dl_train, max_batches=50, device=device)
    print(f"  pos_weight ≈ {pos_weight:.2f}")
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    best_bacc = -1.0
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_path = os.path.join(args.ckpt_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{args.epochs}")
        total_loss = 0.0

        for batch in pbar:
            x = batch["x"].to(device)             # (B,W,F)
            y = batch["y"].to(device)             # (B,W,8)
            om = batch["onset_mask"].to(device)   # (B,W,8)

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                logits = model(x)                 # (B,W,8)
                loss_bce = bce_loss(logits, y)
                loss_on  = onset_weighted_bce(logits, y, om, weight=args.onset_weight)
                probs = torch.sigmoid(logits)
                loss_tv = total_variation_loss(probs, lam=args.tv_lambda)
                loss = loss_bce + loss_on + loss_tv

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            ema.update(model)

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{(total_loss/(pbar.n+1e-8)):.4f}", "lr": f"{sched.get_last_lr()[0]:.2e}"})

        sched.step()

        # ---- validation (EMA weights) ----
        model.eval()
        ema.apply_shadow(model)
        with torch.no_grad():
            all_true, all_prob = [], []
            for batch in dl_val:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                logits = model(x)
                prob = torch.sigmoid(logits)
                all_true.append(y.cpu().numpy())
                all_prob.append(prob.cpu().numpy())
            Y = np.concatenate(all_true, axis=0)  # (N,W,8)
            P = np.concatenate(all_prob, axis=0)
            metrics = compute_timewise_metrics(Y.reshape(-1,8), P.reshape(-1,8), threshold=args.eval_threshold)
            bacc = metrics["bacc"]
            print(f"[Val] bacc={bacc:.4f} f1={metrics['f1']:.4f} auroc={metrics['auroc']:.4f}")
            if bacc > best_bacc:
                best_bacc = bacc
                torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
                print(f"  ✅ Saved best to {best_path}")
        ema.restore(model)

    print(f"Training done. Best balanced accuracy={best_bacc:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--win_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--onset_weight", type=float, default=3.0)
    ap.add_argument("--tv_lambda", type=float, default=0.05)
    ap.add_argument("--eval_threshold", type=float, default=0.5)
    args = ap.parse_args()
    train(args)
"""
python3 -m fault_detect.train \
  --data_dir data_storage/link_3 \
  --epochs 25 \
  --batch_size 24 \
  --win_size 256 \
  --workers 8
"""