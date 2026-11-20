"""Classifier utilities for Stage B: dataset, model, training, evaluation.

Nothing here depends on src/stainops or WILDS — it only reads .npz shards
produced by Stage A.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ── reproducibility helper ──────────────────────────────────────────────────

def seed_everything(seed: int = 0) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── dataset ─────────────────────────────────────────────────────────────────

class ShardDataset(Dataset):
    """Reads .npz shards written by Stage A.

    Shards are loaded lazily: on first access to a shard its contents are
    cached in memory for the rest of the epoch.  This keeps RAM proportional
    to the number of concurrently accessed shards rather than the whole split.

    Parameters
    ----------
    shard_paths : list of Path
        Sorted list of .npz shard files for one (method, split) pair.
    augment : bool
        If True, apply random horizontal + vertical flips (train only).
    """

    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, shard_paths: List[Path], augment: bool = False):
        self.shard_paths = sorted(shard_paths)
        self.augment = augment

        # Build an index: list of (shard_idx, within_shard_idx) pairs.
        self._index: List[tuple[int, int]] = []
        self._sizes: List[int] = []
        for si, p in enumerate(self.shard_paths):
            # Peek at the shape without loading pixel data
            with np.load(p, allow_pickle=False) as d:
                n = d["labels"].shape[0]
            self._sizes.append(n)
            self._index.extend((si, j) for j in range(n))

        self._cache: dict[int, dict] = {}

        tf_list = [transforms.ToTensor(),
                   transforms.Normalize(self._IMAGENET_MEAN, self._IMAGENET_STD)]
        if augment:
            tf_list = [transforms.RandomHorizontalFlip(),
                       transforms.RandomVerticalFlip()] + tf_list
        self.transform = transforms.Compose(tf_list)

    def __len__(self) -> int:
        return len(self._index)

    def _load_shard(self, si: int) -> dict:
        if si not in self._cache:
            with np.load(self.shard_paths[si], allow_pickle=False) as d:
                self._cache[si] = {
                    "patches":  d["patches"],    # uint8 (B,96,96,3)
                    "labels":   d["labels"],     # int32 (B,)
                    "hospital": d["hospital"],   # int32 (B,)
                }
        return self._cache[si]

    def __getitem__(self, idx: int):
        si, j = self._index[idx]
        shard = self._load_shard(si)
        img   = shard["patches"][j]            # uint8 HxWx3
        label = int(shard["labels"][j])
        hosp  = int(shard["hospital"][j])
        # PIL-free path: numpy uint8 -> float tensor via transforms
        img_pil = transforms.functional.to_pil_image(img)
        x = self.transform(img_pil)
        return x, label, hosp

    def clear_cache(self) -> None:
        self._cache.clear()


def make_loader(shard_paths: List[Path], augment: bool, batch_size: int,
                num_workers: int = 2, shuffle: bool = True) -> DataLoader:
    ds = ShardDataset(shard_paths, augment=augment)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())


# ── model ────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """ResNet-18 with the final FC replaced for `num_classes` outputs."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ── training loop ────────────────────────────────────────────────────────────

def train_epoch(model: nn.Module, loader: DataLoader, optimizer,
                criterion, device: torch.device) -> dict:
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return {"loss": total_loss / n, "acc": correct / n}


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader,
               criterion, device: torch.device) -> dict:
    """Returns loss, accuracy, and per-sample (label, prob_pos, hospital) arrays."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_labels, all_probs, all_hosps = [], [], []
    for x, y, hosp in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        probs = torch.softmax(logits, dim=1)[:, 1]
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
        all_labels.append(y.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_hosps.append(hosp.numpy())
    return {
        "loss":    total_loss / n,
        "acc":     correct / n,
        "labels":  np.concatenate(all_labels),
        "probs":   np.concatenate(all_probs),
        "hosps":   np.concatenate(all_hosps),
    }


# ── AUC helper (no sklearn dependency) ──────────────────────────────────────

def roc_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Compute ROC-AUC via the trapezoidal rule (binary classification)."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    order = np.argsort(-probs)
    labels_sorted = labels[order]
    n_pos = labels_sorted.sum()
    n_neg = len(labels_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    tpr = tp / n_pos
    fpr = fp / n_neg
    # prepend origin
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    return float(np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr))


# ── full training run ────────────────────────────────────────────────────────

def run_training(
    method_key: str,
    train_shards: List[Path],
    val_shards:   List[Path],
    test_shards:  List[Path],
    *,
    epochs:       int   = 10,
    batch_size:   int   = 64,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers:  int   = 2,
    seed:         int   = 0,
    device:       Optional[torch.device] = None,
) -> pd.DataFrame:
    """Train ResNet-18 on one normalisation method's cached shards.

    Returns a DataFrame with one row per (epoch, split) with columns:
      method, seed, epoch, split, loss, acc, auc
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_everything(seed)
    model = build_model(num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = make_loader(train_shards, augment=True,
                               batch_size=batch_size, num_workers=num_workers)
    val_loader   = make_loader(val_shards,   augment=False,
                               batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader  = make_loader(test_shards,  augment=False,
                               batch_size=batch_size, num_workers=num_workers, shuffle=False)

    rows = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        for split_name, loader in [("val", val_loader), ("test", test_loader)]:
            ev = eval_epoch(model, loader, criterion, device)
            auc = roc_auc(ev["labels"], ev["probs"])
            rows.append(dict(
                method=method_key, seed=seed, epoch=epoch,
                split=split_name,
                loss=round(ev["loss"], 5),
                acc=round(ev["acc"],  5),
                auc=round(auc,        5),
            ))

        elapsed = time.time() - t0
        val_row = next(r for r in rows[::-1] if r["split"] == "val")
        print(f"  epoch {epoch:>2}/{epochs}  "
              f"train_loss={tr['loss']:.4f}  train_acc={tr['acc']:.4f}  "
              f"val_acc={val_row['acc']:.4f}  val_auc={val_row['auc']:.4f}  "
              f"({elapsed:.0f}s)", flush=True)

    return pd.DataFrame(rows)
