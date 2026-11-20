"""Train a ResNet classifier on cached normalised patches.

Data flow
─────────
  CACHE_DIR/{method}/X_{split}.npy   uint8 (N,96,96,3) — written by merge_shards.py
  CACHE_DIR/{method}/y_{split}.npy   int32 (N,)
  CACHE_DIR/{method}/hospital_{split}.npy  int32 (N,)

The full arrays are loaded once into CPU RAM as uint8 tensors, then moved to
GPU in mini-batches.  All augmentation runs on GPU.  No DataLoader workers.

Outputs
───────
  RESULTS_DIR/{method}_seed{seed}.json  — best-epoch metrics + args
  CACHE_DIR/{method}/ckpt_seed{seed}.pt — checkpoint (resumed automatically)

Usage
─────
    python scripts/train.py --method ot_bal --seed 0
    python scripts/train.py --method none --epochs 20 --seed 1 --color_aug
    python scripts/train.py --method macenko --subsample 5000 --seed 0  # pilot
    python scripts/train.py --method ot_bal --backbone resnet18 --lr_sched cosine
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ── repo-root bootstrap ──────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import config as cfg

# ── constants ────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # torch imported lazily — seed after import
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_split(method_dir: Path, split: str,
               subsample: Optional[int], seed: int):
    """Return (X uint8 cpu tensor, y long cpu tensor, hospital int32 np array)."""
    import torch
    X = np.load(method_dir / f"X_{split}.npy")
    y = np.load(method_dir / f"y_{split}.npy")
    h = np.load(method_dir / f"hospital_{split}.npy")

    if subsample is not None and subsample < len(X):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=subsample, replace=False)
        X, y, h = X[idx], y[idx], h[idx]

    # (N,H,W,C) uint8 → (N,C,H,W) uint8 tensor on CPU
    X_t = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()   # uint8
    y_t = torch.from_numpy(y.astype(np.int64))
    return X_t, y_t, h


# ─────────────────────────────────────────────────────────────────────────────
# GPU augmentation + normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _to_float_normalised(x: "torch.Tensor") -> "torch.Tensor":
    """uint8 (B,C,H,W) → float32 ImageNet-normalised on the same device."""
    import torch
    x = x.float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=torch.float32)[None, :, None, None]
    std  = torch.tensor(IMAGENET_STD,  device=x.device, dtype=torch.float32)[None, :, None, None]
    return (x - mean) / std


def _augment_gpu(x: "torch.Tensor", color_aug: bool) -> "torch.Tensor":
    """In-place GPU augmentation on a uint8 (B,C,H,W) batch."""
    import torch
    import torchvision.transforms.functional as TF

    # random horizontal flip
    if torch.rand(1).item() > 0.5:
        x = torch.flip(x, dims=[3])
    # random vertical flip
    if torch.rand(1).item() > 0.5:
        x = torch.flip(x, dims=[2])
    # random rot90 (0/1/2/3 × 90°)
    k = int(torch.randint(4, (1,)).item())
    if k:
        x = torch.rot90(x, k=k, dims=[2, 3])
    # optional colour jitter (H&E-safe: brightness/contrast only, no saturation)
    if color_aug:
        x = x.float()
        # brightness ±20 %
        b = 1.0 + (torch.rand(1, device=x.device).item() - 0.5) * 0.4
        x = torch.clamp(x * b, 0, 255)
        # contrast ±20 %
        c = 1.0 + (torch.rand(1, device=x.device).item() - 0.5) * 0.4
        mean_px = x.mean(dim=[2, 3], keepdim=True)
        x = torch.clamp((x - mean_px) * c + mean_px, 0, 255)
        x = x.to(torch.uint8)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(backbone: str, num_classes: int = 2) -> "torch.nn.Module":
    """ResNet trained from scratch (no pretrained weights) with 2-class head."""
    import torch.nn as nn
    from torchvision import models

    if backbone == "resnet18":
        model = models.resnet18(weights=None)
    elif backbone == "resnet34":
        model = models.resnet34(weights=None)
    elif backbone == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported backbone: {backbone!r}. Choose resnet18/34/50.")

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# AUC (no sklearn)
# ─────────────────────────────────────────────────────────────────────────────

def roc_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    order         = np.argsort(-probs)
    labels_sorted = labels[order]
    n_pos = int(labels_sorted.sum())
    n_neg = len(labels_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp  = np.cumsum(labels_sorted)
    fp  = np.cumsum(1 - labels_sorted)
    tpr = np.concatenate([[0.0], tp / n_pos])
    fpr = np.concatenate([[0.0], fp / n_neg])
    fn  = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(fn(tpr, fpr))


# ─────────────────────────────────────────────────────────────────────────────
# Training / evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch_train(model, X_cpu, y_cpu, batch_size, optimizer, criterion,
                    scaler, device, color_aug: bool):
    import torch
    model.train()
    n = len(X_cpu)
    perm = torch.randperm(n)
    total_loss = correct = 0

    for start in range(0, n, batch_size):
        idx  = perm[start:start + batch_size]
        xb   = X_cpu[idx].to(device, non_blocking=True)   # uint8
        yb   = y_cpu[idx].to(device, non_blocking=True)
        xb   = _augment_gpu(xb, color_aug)
        xb   = _to_float_normalised(xb)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            logits = model(xb)
            loss   = criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(yb)
        correct    += (logits.detach().argmax(1) == yb).sum().item()

    return {"loss": total_loss / n, "acc": correct / n}


@torch.no_grad() if False else lambda f: f   # decorator applied below
def run_epoch_eval(model, X_cpu, y_cpu, batch_size, criterion, device):
    import torch
    model.eval()
    n = len(X_cpu)
    total_loss = correct = 0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb  = X_cpu[start:start + batch_size].to(device, non_blocking=True)
            yb  = y_cpu[start:start + batch_size].to(device, non_blocking=True)
            xb  = _to_float_normalised(xb)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                logits = model(xb)
                loss   = criterion(logits, yb)
            probs = torch.softmax(logits.float(), dim=1)[:, 1]
            total_loss += loss.item() * len(yb)
            correct    += (logits.argmax(1) == yb).sum().item()
            all_labels.append(yb.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return {
        "loss":   total_loss / n,
        "acc":    correct / n,
        "labels": np.concatenate(all_labels),
        "probs":  np.concatenate(all_probs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ckpt_path(method_dir: Path, seed: int) -> Path:
    return method_dir / f"ckpt_seed{seed}.pt"


def save_checkpoint(path: Path, epoch: int, model, optimizer, scaler,
                    scheduler, history: list) -> None:
    import torch
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler":    scaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "history":   history,
    }, path)


def load_checkpoint(path: Path, model, optimizer, scaler, scheduler):
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["history"]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a ResNet on cached normalised patches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--method",     required=True,
                   help="Normalisation method key (must match CACHE_DIR sub-dir).")
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--epochs",     type=int, default=10)
    p.add_argument("--backbone",   default="resnet18",
                   choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lr_sched",   default="cosine",
                   choices=["cosine", "step"],
                   help="LR schedule: cosine annealing or step (×0.1 at 50%% and 75%%).")
    p.add_argument("--color_aug",  action="store_true",
                   help="Add GPU colour jitter (brightness + contrast) for the aug baseline.")
    p.add_argument("--subsample",  type=int, default=None,
                   help="Use only this many training patches (pilot runs).")
    p.add_argument("--cache_dir",  type=Path, default=cfg.CACHE_DIR,
                   help="Default directory for data and checkpoints.")
    p.add_argument("--data_dir",   type=Path, default=None,
                   help="Directory containing {method}/X_{split}.npy (defaults to --cache_dir).")
    p.add_argument("--ckpt_dir",   type=Path, default=None,
                   help="Directory to save/load checkpoints (defaults to --cache_dir).")
    p.add_argument("--no_checkpoint", action="store_true",
                   help="Disable saving/loading checkpoints.")
    p.add_argument("--results_dir",type=Path, default=cfg.RESULTS_DIR)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import torch
    import torch.nn as nn

    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device     : {device}")
    if device.type == "cuda":
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"method     : {args.method}")
    print(f"seed       : {args.seed}")
    print(f"backbone   : {args.backbone}")
    print(f"epochs     : {args.epochs}")
    print(f"batch_size : {args.batch_size}")
    print(f"color_aug  : {args.color_aug}")
    print(f"lr_sched   : {args.lr_sched}")
    if args.subsample:
        print(f"subsample  : {args.subsample}")

    data_dir = args.data_dir if args.data_dir is not None else args.cache_dir
    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else args.cache_dir

    method_data_dir = data_dir / args.method
    method_ckpt_dir = ckpt_dir / args.method

    if not args.no_checkpoint:
        method_ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ────────────────────────────────────────────────────────────
    print("\nLoading arrays …")
    X_tr, y_tr, _ = load_split(method_data_dir, "train", args.subsample, args.seed)
    X_va, y_va, _ = load_split(method_data_dir, "val",   None,            args.seed)
    X_te, y_te, _ = load_split(method_data_dir, "test",  None,            args.seed)
    print(f"  train {len(X_tr):,}   val {len(X_va):,}   test {len(X_te):,}")

    # ── model / optimiser ────────────────────────────────────────────────────
    model     = build_model(args.backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_sched == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)
    else:  # step
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[args.epochs // 2, 3 * args.epochs // 4],
            gamma=0.1)
    scaler = (torch.amp.GradScaler("cuda") if device.type == "cuda"
              else torch.amp.GradScaler("cpu", enabled=False))

    # ── resume from checkpoint if available ──────────────────────────────────
    ckpt_path   = _ckpt_path(method_ckpt_dir, args.seed)
    start_epoch = 1
    history     = []        # list of dicts, one per epoch
    if not args.no_checkpoint and ckpt_path.exists():
        print(f"\nResuming from {ckpt_path}")
        start_epoch, history = load_checkpoint(
            ckpt_path, model, optimizer, scaler, scheduler)
        start_epoch += 1
        print(f"  starting from epoch {start_epoch}")

    # ── training loop ────────────────────────────────────────────────────────
    t_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()

        tr = run_epoch_train(model, X_tr, y_tr, args.batch_size,
                             optimizer, criterion, scaler, device, args.color_aug)
        scheduler.step()

        va = run_epoch_eval(model, X_va, y_va, args.batch_size, criterion, device)
        te = run_epoch_eval(model, X_te, y_te, args.batch_size, criterion, device)

        val_auc  = roc_auc(va["labels"], va["probs"])
        test_auc = roc_auc(te["labels"], te["probs"])

        elapsed = time.perf_counter() - t0
        print(
            f"epoch {epoch:>3}/{args.epochs}  "
            f"tr_loss={tr['loss']:.4f}  tr_acc={tr['acc']:.4f}  "
            f"val_acc={va['acc']:.4f}  val_auc={val_auc:.4f}  "
            f"te_acc={te['acc']:.4f}  te_auc={test_auc:.4f}  "
            f"({elapsed:.0f}s)",
            flush=True,
        )

        history.append({
            "epoch":    epoch,
            "tr_loss":  round(tr["loss"],    5),
            "tr_acc":   round(tr["acc"],     5),
            "val_loss": round(va["loss"],    5),
            "val_acc":  round(va["acc"],     5),
            "val_auc":  round(val_auc,       5),
            "te_loss":  round(te["loss"],    5),
            "te_acc":   round(te["acc"],     5),
            "te_auc":   round(test_auc,      5),
        })

        if not args.no_checkpoint:
            save_checkpoint(ckpt_path, epoch, model, optimizer, scaler, scheduler, history)

    # ── select best epoch by VAL AUC (never by test) ─────────────────────────
    best = max(history, key=lambda r: r["val_auc"])
    runtime = time.time() - t_start
    print(
        f"\nBest epoch: {best['epoch']}  "
        f"val_auc={best['val_auc']:.4f}  "
        f"test_acc={best['te_acc']:.4f}  test_auc={best['te_auc']:.4f}  "
        f"total {runtime/60:.1f} min"
    )

    # ── write JSON result ─────────────────────────────────────────────────────
    result = {
        "method":      args.method,
        "seed":        args.seed,
        "best_epoch":  best["epoch"],
        "val_acc":     best["val_acc"],
        "val_auc":     best["val_auc"],
        "test_acc":    best["te_acc"],
        "test_auc":    best["te_auc"],
        "runtime_s":   round(runtime, 1),
        "history":     history,
        "args": {
            "backbone":     args.backbone,
            "epochs":       args.epochs,
            "batch_size":   args.batch_size,
            "lr":           args.lr,
            "weight_decay": args.weight_decay,
            "lr_sched":     args.lr_sched,
            "color_aug":    args.color_aug,
            "subsample":    args.subsample,
        },
    }
    out_path = args.results_dir / f"{args.method}_seed{args.seed}.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
