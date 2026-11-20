"""Build a reproducible subset of CAMELYON17-WILDS and save it to SUBSET_DIR.

Outputs per split (train / val / test)
───────────────────────────────────────
  X_{split}.npy          uint8  (N, 96, 96, 3)
  y_{split}.npy          int32  (N,)
  hospital_{split}.npy   int32  (N,)
  idx_{split}.npy        int64  (N,)   original dataset indices (for traceability)

Usage
─────
  # full run
  python scripts/make_subset.py

  # quick smoke-test (~200 patches total)
  python scripts/make_subset.py --debug

  # custom sizes
  python scripts/make_subset.py --n_per_hospital 5000 --n_val 2000 --n_test 3000

  # explicit output dir / data root
  python scripts/make_subset.py --subset_dir /tmp/subset --data_root /data/wilds

Defaults for sizes come from config.py and can also be overridden via env vars
(N_PER_HOSPITAL, N_VAL, N_TEST, SEED).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import numpy as np

# ── repo-root bootstrap (works whether or not the package is installed) ──────
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# pure helpers — no WILDS import at module level
# ─────────────────────────────────────────────────────────────────────────────

def _balanced_idx(ds, hospitals: List[int], n_per_hospital: int,
                  seed: int) -> np.ndarray:
    """Return indices that are class-balanced within each hospital.

    For each hospital: draw min(n_per_hospital//2, available_class) samples
    from each class so the result is always exactly 50 / 50, or as close as
    the rarest class allows.
    """
    col      = ds.metadata_fields.index("hospital")
    hosp_arr = ds.metadata_array[:, col].numpy()
    y_arr    = ds.y_array.numpy()
    rng      = np.random.default_rng(seed)
    chosen: List[np.ndarray] = []

    for h in hospitals:
        in_h     = np.where(hosp_arr == h)[0]
        pos_pool = in_h[y_arr[in_h] == 1]
        neg_pool = in_h[y_arr[in_h] == 0]
        half     = n_per_hospital // 2
        n_take   = min(half, len(pos_pool), len(neg_pool))
        if n_take == 0:
            raise RuntimeError(
                f"Hospital {h} has no samples in one class — "
                "check DATA_ROOT and dataset integrity."
            )
        pos_idx = rng.choice(pos_pool, size=n_take, replace=False)
        neg_idx = rng.choice(neg_pool, size=n_take, replace=False)
        chosen.append(np.concatenate([pos_idx, neg_idx]))

    return np.concatenate(chosen)


def _random_idx(ds, hospitals: List[int], n: int, seed: int) -> np.ndarray:
    """Return up to n randomly sampled indices from the given hospitals."""
    col      = ds.metadata_fields.index("hospital")
    hosp_arr = ds.metadata_array[:, col].numpy()
    pool     = np.where(np.isin(hosp_arr, hospitals))[0]
    rng      = np.random.default_rng(seed)
    return rng.choice(pool, size=min(n, len(pool)), replace=False)


def _load_patches(ds, indices: np.ndarray) -> np.ndarray:
    """Load uint8 RGB patches for the given dataset indices with a progress bar."""
    from tqdm import tqdm
    patches = np.empty((len(indices), 96, 96, 3), dtype=np.uint8)
    for i, j in enumerate(tqdm(indices, desc="  loading", unit="patch",
                                dynamic_ncols=True)):
        patches[i] = np.asarray(ds.get_input(int(j)).convert("RGB"))
    return patches


def _print_stats(split: str, y: np.ndarray, hospital: np.ndarray) -> None:
    total = len(y)
    n_pos = int((y == 1).sum())
    n_neg = total - n_pos
    print(f"\n  {split}:  {total:,} patches  "
          f"(pos={n_pos:,} {100*n_pos/total:.1f}%,  "
          f"neg={n_neg:,} {100*n_neg/total:.1f}%)")
    for h in np.unique(hospital):
        m  = hospital == h
        hp = int((y[m] == 1).sum())
        hn = int((y[m] == 0).sum())
        print(f"    hospital {h}: {m.sum():>6,}  "
              f"pos={hp:,} ({100*hp/m.sum():.1f}%)  "
              f"neg={hn:,} ({100*hn/m.sum():.1f}%)")


def _save(out_dir: Path, split: str,
          X: np.ndarray, y: np.ndarray,
          hospital: np.ndarray, idx: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"X_{split}.npy",        X)
    np.save(out_dir / f"y_{split}.npy",        y.astype(np.int32))
    np.save(out_dir / f"hospital_{split}.npy", hospital.astype(np.int32))
    np.save(out_dir / f"idx_{split}.npy",      idx.astype(np.int64))
    size_mb = sum(
        (out_dir / f"{name}_{split}.npy").stat().st_size
        for name in ("X", "y", "hospital", "idx")
    ) / 1e6
    print(f"  → saved to {out_dir}  ({size_mb:.0f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a reproducible CAMELYON17-WILDS subset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--n_per_hospital", type=int,
        default=int(os.environ.get("N_PER_HOSPITAL", 20_000)),
        help="Train patches per hospital (class-balanced).",
    )
    p.add_argument(
        "--n_val", type=int,
        default=int(os.environ.get("N_VAL", 10_000)),
        help="Val patches from hospital 1.",
    )
    p.add_argument(
        "--n_test", type=int,
        default=int(os.environ.get("N_TEST", 15_000)),
        help="Test patches from hospital 2.",
    )
    p.add_argument(
        "--seed", type=int,
        default=cfg.SEED,
        help="Random seed.",
    )
    p.add_argument(
        "--subset_dir", type=Path,
        default=cfg.SUBSET_DIR,
        help="Output directory.",
    )
    p.add_argument(
        "--data_root", type=Path,
        default=cfg.DATA_ROOT,
        help="WILDS data root (contains camelyon17_v1.0/).",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Tiny subset (~200 patches) for quick smoke-tests.",
    )
    p.add_argument(
        "--download", action="store_true",
        help="Download CAMELYON17 dataset automatically if not present in data_root.",
    )
    return p.parse_args()


def main() -> None:
    # deferred so tests can import helpers without installing wilds
    from src.data import _dataset

    args = parse_args()

    if args.debug:
        # 50 per hospital × 3 = 150 train, 25 val, 25 test
        args.n_per_hospital = 50
        args.n_val          = 25
        args.n_test         = 25
        print("── DEBUG MODE: tiny subset ──")

    print(f"data_root  : {args.data_root}")
    print(f"subset_dir : {args.subset_dir}")
    print(f"seed       : {args.seed}")
    print(f"train      : {len(cfg.TRAIN_HOSPITALS)} hospitals × "
          f"{args.n_per_hospital} patches (class-balanced)")
    print(f"val        : hospitals {cfg.VAL_HOSPITALS}  {args.n_val} patches")
    print(f"test       : hospitals {cfg.TEST_HOSPITALS}  {args.n_test} patches")

    # ── load WILDS dataset once ──────────────────────────────────────────────
    print("\nLoading WILDS dataset …")
    ds = _dataset(root=str(args.data_root), download=args.download)

    # ── train ────────────────────────────────────────────────────────────────
    print("\n[train]  sampling class-balanced indices …")
    tr_idx = _balanced_idx(ds, cfg.TRAIN_HOSPITALS, args.n_per_hospital, args.seed)
    np.random.default_rng(args.seed).shuffle(tr_idx)   # interleave hospitals
    print(f"  {len(tr_idx):,} indices selected, loading patches …")
    X_tr = _load_patches(ds, tr_idx)
    h_col = ds.metadata_fields.index("hospital")
    y_tr  = ds.y_array[tr_idx].numpy().astype(np.int32)
    h_tr  = ds.metadata_array[tr_idx, h_col].numpy().astype(np.int32)
    _print_stats("train", y_tr, h_tr)
    _save(args.subset_dir, "train", X_tr, y_tr, h_tr, tr_idx)

    # ── val ──────────────────────────────────────────────────────────────────
    print("\n[val]  sampling indices …")
    va_idx = _random_idx(ds, cfg.VAL_HOSPITALS, args.n_val, args.seed)
    print(f"  {len(va_idx):,} indices selected, loading patches …")
    X_va = _load_patches(ds, va_idx)
    y_va  = ds.y_array[va_idx].numpy().astype(np.int32)
    h_va  = ds.metadata_array[va_idx, h_col].numpy().astype(np.int32)
    _print_stats("val", y_va, h_va)
    _save(args.subset_dir, "val", X_va, y_va, h_va, va_idx)

    # ── test ─────────────────────────────────────────────────────────────────
    print("\n[test]  sampling indices …")
    te_idx = _random_idx(ds, cfg.TEST_HOSPITALS, args.n_test, args.seed)
    print(f"  {len(te_idx):,} indices selected, loading patches …")
    X_te = _load_patches(ds, te_idx)
    y_te  = ds.y_array[te_idx].numpy().astype(np.int32)
    h_te  = ds.metadata_array[te_idx, h_col].numpy().astype(np.int32)
    _print_stats("test", y_te, h_te)
    _save(args.subset_dir, "test", X_te, y_te, h_te, te_idx)

    print("\nDone.")


if __name__ == "__main__":
    main()
