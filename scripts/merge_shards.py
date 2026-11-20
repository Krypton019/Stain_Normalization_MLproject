"""Concatenate per-shard .npy files into full split arrays.

Reads:   CACHE_DIR/{method}/{split}_{shard:04d}.npy
Writes:  CACHE_DIR/{method}/X_{split}.npy
         CACHE_DIR/{method}/y_{split}.npy
         CACHE_DIR/{method}/hospital_{split}.npy
         CACHE_DIR/{method}/idx_{split}.npy

Usage
─────
    # merge all methods and all splits
    python scripts/merge_shards.py

    # single method
    python scripts/merge_shards.py --method ot_bal

    # specific splits
    python scripts/merge_shards.py --method macenko --splits train val

    # do not delete shards after merging (default: keep)
    python scripts/merge_shards.py --method none --delete_shards
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import config as cfg

VALID_METHODS = ["none", "reinhard", "macenko", "ot_bal", "ot_unbal"]
ALL_SPLITS    = ["train", "val", "test"]


def _read_shard(path: Path) -> dict:
    return np.load(path, allow_pickle=True).item()


def merge_split(method_dir: Path, split: str, delete_shards: bool) -> None:
    shards = sorted(method_dir.glob(f"{split}_????.npy"))
    if not shards:
        print(f"  {split}: no shards found — skipping")
        return

    # check none of the merged outputs is itself a shard glob match
    shards = [s for s in shards
              if s.stem[len(split)+1:].isdigit()]   # e.g. train_0000

    if not shards:
        print(f"  {split}: no shards found — skipping")
        return

    print(f"  {split}: merging {len(shards)} shard(s) …", end=" ", flush=True)

    Xs, ys, hs, idxs = [], [], [], []
    for path in shards:
        d = _read_shard(path)
        Xs.append(d["X"])
        ys.append(d["y"])
        hs.append(d["hospital"])
        idxs.append(d["idx"])

    X   = np.concatenate(Xs,   axis=0)
    y   = np.concatenate(ys,   axis=0)
    h   = np.concatenate(hs,   axis=0)
    idx = np.concatenate(idxs, axis=0)

    np.save(method_dir / f"X_{split}.npy",        X)
    np.save(method_dir / f"y_{split}.npy",        y)
    np.save(method_dir / f"hospital_{split}.npy", h)
    np.save(method_dir / f"idx_{split}.npy",      idx)

    size_mb = X.nbytes / 1e6
    print(f"{len(X):,} patches  ({size_mb:.0f} MB)")

    if delete_shards:
        for path in shards:
            path.unlink()
        print(f"    deleted {len(shards)} shard file(s)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Concatenate normalize_cache shards into full split arrays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--method", nargs="+", default=VALID_METHODS,
        choices=VALID_METHODS, metavar="METHOD",
    )
    p.add_argument(
        "--splits", nargs="+", default=ALL_SPLITS,
        choices=ALL_SPLITS, metavar="SPLIT",
    )
    p.add_argument(
        "--cache_dir", type=Path, default=cfg.CACHE_DIR,
        help="Root cache directory.",
    )
    p.add_argument(
        "--delete_shards", action="store_true",
        help="Remove individual shard files after merging.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for method in args.method:
        method_dir = args.cache_dir / method
        if not method_dir.exists():
            print(f"\n{method}: directory {method_dir} not found — skipping")
            continue
        print(f"\n{method}:")
        for split in args.splits:
            merge_split(method_dir, split, args.delete_shards)
    print("\nDone.")


if __name__ == "__main__":
    main()
