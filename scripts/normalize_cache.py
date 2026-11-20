"""Apply stain normalisation to subset .npy files and write shards to CACHE_DIR.

Input  (SUBSET_DIR):   X_{split}.npy, y_{split}.npy, hospital_{split}.npy, idx_{split}.npy
Output (CACHE_DIR/{method}/):
    {split}_{shard:04d}.npy   – dict with keys X, y, hospital, idx (uint8/int32/int32/int64)
    reference.pkl              – fitted reference object (shared across methods)
    failures.csv               – patches that fell back to the original

Usage
─────
    # full run, all splits, one method
    python scripts/normalize_cache.py --method ot_bal

    # multiple methods at once
    python scripts/normalize_cache.py --method none reinhard macenko ot_bal ot_unbal

    # specific splits only
    python scripts/normalize_cache.py --method macenko --splits train val

    # dry run: 200 patches, saves a before/after grid PNG
    python scripts/normalize_cache.py --method ot_bal --dry_run

    # limit to first N patches per split (useful for benchmarking)
    python scripts/normalize_cache.py --method ot_bal --limit 500

    # resume interrupted run (existing shards are skipped automatically)
    python scripts/normalize_cache.py --method ot_bal

Methods: none  reinhard  macenko  ot_bal  ot_unbal
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import pickle
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── repo-root bootstrap ──────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import config as cfg

# ── constants ────────────────────────────────────────────────────────────────
VALID_METHODS = ["none", "reinhard", "macenko", "ot_bal", "ot_unbal"]
ALL_SPLITS    = ["train", "val", "test"]

# Reference fitted on this many train patches (saves time vs fitting on all 60 k)
N_REF = 500
REF_SEED = cfg.SEED


# ─────────────────────────────────────────────────────────────────────────────
# Reference fitting and persistence
# ─────────────────────────────────────────────────────────────────────────────

def _build_reference(subset_dir: Path, cache_dir: Path, seed: int) -> dict:
    """Fit a shared reference from X_train.npy and return a reference dict.

    The reference dict contains everything needed to run any method:
        ref_img       – single uint8 (96,96,3) image for Reinhard
        W_ref         – (3,2) mean stain matrix for Macenko
        p99_ref       – (2,) 99th-percentile concentrations for Macenko
        ot_bal        – fitted OTNormalizer(mode='balanced')
        ot_unbal      – fitted OTNormalizer(mode='unbalanced')
    """
    # deferred imports so the module can be imported without heavy deps
    from src import stainops as so
    from src.ot_norm import OTNormalizer

    X_train = np.load(subset_dir / "X_train.npy", mmap_mode="r")
    rng     = np.random.default_rng(seed)
    idx     = rng.choice(len(X_train), size=min(N_REF, len(X_train)), replace=False)
    ref_imgs = [X_train[i] for i in idx]

    # ── Reinhard reference: single tissue-rich patch ──────────────────────
    ref_img = ref_imgs[0]
    for img in ref_imgs:
        if so.tissue_mask(img).sum() > 500:
            ref_img = img
            break

    # ── Macenko W_ref and p99 ─────────────────────────────────────────────
    Ws, Hs = [], []
    for img in ref_imgs:
        m = so.tissue_mask(img)
        if m.sum() < 200:
            continue
        W = so.macenko_stain_matrix(img)
        if np.linalg.cond(W) > 20:
            continue
        H = so.concentrations(so.rgb_to_od(img), W)[m]
        Ws.append(W)
        Hs.append(H)
    if not Ws:
        raise RuntimeError("No usable reference patches found in X_train.npy")
    W_ref = np.mean(Ws, axis=0)
    W_ref = W_ref / np.linalg.norm(W_ref, axis=0, keepdims=True)
    H_all = np.concatenate(Hs)
    p99_ref = np.percentile(H_all, 99, axis=0)

    # ── OT normalisers ────────────────────────────────────────────────────
    print(f"  fitting OTNormalizer (balanced)   on {len(ref_imgs)} patches …")
    ot_bal   = OTNormalizer(mode="balanced",   eps=0.1,          n_sub=500, seed=seed)
    ot_bal.fit(ref_imgs)
    print(f"  fitting OTNormalizer (unbalanced) on {len(ref_imgs)} patches …")
    ot_unbal = OTNormalizer(mode="unbalanced", eps=0.1, rho=1.0, n_sub=500, seed=seed)
    ot_unbal.fit(ref_imgs)

    return dict(
        ref_img=ref_img,
        W_ref=W_ref,
        p99_ref=p99_ref,
        ot_bal=ot_bal,
        ot_unbal=ot_unbal,
    )


def load_or_build_reference(subset_dir: Path, cache_dir: Path, seed: int) -> dict:
    ref_path = cache_dir / "reference.pkl"
    if ref_path.exists():
        print(f"  reusing reference from {ref_path}")
        with open(ref_path, "rb") as fh:
            return pickle.load(fh)
    print("  building reference …")
    ref = _build_reference(subset_dir, cache_dir, seed)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(ref_path, "wb") as fh:
        pickle.dump(ref, fh, protocol=4)
    print(f"  saved reference → {ref_path}")
    return ref


# ─────────────────────────────────────────────────────────────────────────────
# Per-patch normalisation (called in worker processes)
# ─────────────────────────────────────────────────────────────────────────────

# Global in each worker process (set by _worker_init)
_WORKER_METHOD: str = ""
_WORKER_REF:    dict = {}


def _worker_init(method: str, ref: dict) -> None:
    global _WORKER_METHOD, _WORKER_REF
    _WORKER_METHOD = method
    _WORKER_REF    = ref


def _normalise_one(patch: np.ndarray) -> np.ndarray:
    """Normalise a single uint8 (96,96,3) patch. Returns uint8 (96,96,3)."""
    from src import stainops as so

    method = _WORKER_METHOD
    ref    = _WORKER_REF

    if method == "none":
        return patch

    if method == "reinhard":
        return so.reinhard(patch, ref["ref_img"])

    if method == "macenko":
        W_ref   = ref["W_ref"]
        p99_ref = ref["p99_ref"]
        m = so.tissue_mask(patch)
        if m.sum() < 200:
            return patch.copy()
        W = so.macenko_stain_matrix(patch)
        if np.linalg.cond(W) > 20:
            W = W_ref
        H = so.concentrations(so.rgb_to_od(patch), W)[m]
        p99_src = np.percentile(H, 99, axis=0) + 1e-8
        Hn = np.clip(H * (p99_ref / p99_src), 0, None)
        out = patch.copy()
        out[m] = so.od_to_rgb(Hn @ W_ref.T)
        return out

    if method == "ot_bal":
        return ref["ot_bal"].transform(patch)

    if method == "ot_unbal":
        return ref["ot_unbal"].transform(patch)

    raise ValueError(f"Unknown method: {method!r}")


def _safe_normalise_one(args: Tuple[int, np.ndarray]) -> Tuple[int, np.ndarray, Optional[str]]:
    """Wrapper that catches exceptions and returns the original patch on failure."""
    patch_idx, patch = args
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _normalise_one(patch)
        return patch_idx, result, None
    except Exception:
        tb = traceback.format_exc().strip().replace("\n", " | ")
        return patch_idx, patch.copy(), tb


# ─────────────────────────────────────────────────────────────────────────────
# Shard I/O
# ─────────────────────────────────────────────────────────────────────────────

def _shard_path(method_dir: Path, split: str, shard_idx: int) -> Path:
    return method_dir / f"{split}_{shard_idx:04d}.npy"


def _write_shard(path: Path,
                 X: np.ndarray, y: np.ndarray,
                 hospital: np.ndarray, idx: np.ndarray) -> None:
    np.save(path, {           # save as object array containing a dict
        "X":        X.astype(np.uint8),
        "y":        y.astype(np.int32),
        "hospital": hospital.astype(np.int32),
        "idx":      idx.astype(np.int64),
    })


def _read_shard(path: Path) -> dict:
    return np.load(path, allow_pickle=True).item()


# ─────────────────────────────────────────────────────────────────────────────
# Core processing loop
# ─────────────────────────────────────────────────────────────────────────────

def _process_split(
    method: str,
    split: str,
    subset_dir: Path,
    method_dir: Path,
    ref: dict,
    shard_size: int,
    workers: int,
    limit: Optional[int],
    failure_rows: list,
) -> None:
    X_all   = np.load(subset_dir / f"X_{split}.npy",        mmap_mode="r")
    y_all   = np.load(subset_dir / f"y_{split}.npy",        mmap_mode="r")
    h_all   = np.load(subset_dir / f"hospital_{split}.npy", mmap_mode="r")
    idx_all = np.load(subset_dir / f"idx_{split}.npy",      mmap_mode="r")

    n_total = len(X_all) if limit is None else min(limit, len(X_all))
    n_shards = (n_total + shard_size - 1) // shard_size
    print(f"  {split}: {n_total:,} patches → {n_shards} shard(s) of ≤{shard_size}")

    pool_ctx = (
        mp.Pool(workers, initializer=_worker_init, initargs=(method, ref))
        if workers > 1 else None
    )
    # single-process: initialise worker globals directly
    if pool_ctx is None:
        _worker_init(method, ref)

    try:
        for shard_idx in range(n_shards):
            out_path = _shard_path(method_dir, split, shard_idx)
            if out_path.exists():
                print(f"    shard {shard_idx:04d}  SKIP (exists)")
                continue

            lo = shard_idx * shard_size
            hi = min(lo + shard_size, n_total)
            patches  = X_all[lo:hi]
            y_shard  = y_all[lo:hi]
            h_shard  = h_all[lo:hi]
            ix_shard = idx_all[lo:hi]
            n_shard  = hi - lo

            t0 = time.perf_counter()
            args = [(i, patches[i]) for i in range(n_shard)]

            if pool_ctx is not None:
                results = pool_ctx.map(_safe_normalise_one, args, chunksize=16)
            else:
                results = [_safe_normalise_one(a) for a in args]

            elapsed = time.perf_counter() - t0
            ms_per  = 1000 * elapsed / n_shard

            X_out = np.empty_like(patches)
            n_fail = 0
            for patch_idx, out_patch, err in results:
                X_out[patch_idx] = out_patch
                if err is not None:
                    n_fail += 1
                    failure_rows.append({
                        "method":     method,
                        "split":      split,
                        "shard":      shard_idx,
                        "patch_pos":  lo + patch_idx,
                        "orig_idx":   int(ix_shard[patch_idx]),
                        "error":      err[:200],
                    })

            _write_shard(out_path, X_out, y_shard, h_shard, ix_shard)
            fail_str = f"  {n_fail} fallback(s)" if n_fail else ""
            print(f"    shard {shard_idx:04d}  "
                  f"{n_shard} patches  {ms_per:.1f} ms/patch{fail_str}")
    finally:
        if pool_ctx is not None:
            pool_ctx.close()
            pool_ctx.join()


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run grid figure
# ─────────────────────────────────────────────────────────────────────────────

def _dry_run_figure(method: str, ref: dict,
                    subset_dir: Path, figures_dir: Path,
                    seed: int, n: int = 200) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _worker_init(method, ref)
    X_train = np.load(subset_dir / "X_train.npy", mmap_mode="r")
    rng     = np.random.default_rng(seed)
    picks   = rng.choice(len(X_train), size=min(n, len(X_train)), replace=False)

    n_show   = min(8, len(picks))
    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 2, 4.4))
    for col, i in enumerate(picks[:n_show]):
        src = X_train[i]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = _normalise_one(src)
        axes[0, col].imshow(src)
        axes[1, col].imshow(out)
        axes[0, col].axis("off")
        axes[1, col].axis("off")
    axes[0, 0].set_ylabel("before", fontsize=9)
    axes[1, 0].set_ylabel("after",  fontsize=9)
    fig.suptitle(f"Dry run — method: {method}", y=1.01)
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / f"dry_run_{method}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  saved dry-run figure → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stain-normalise subset .npy files and write shards to CACHE_DIR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--method", nargs="+", default=VALID_METHODS,
        choices=VALID_METHODS, metavar="METHOD",
        help=f"One or more of: {', '.join(VALID_METHODS)}.",
    )
    p.add_argument(
        "--splits", nargs="+", default=ALL_SPLITS,
        choices=ALL_SPLITS, metavar="SPLIT",
        help="Splits to process.",
    )
    p.add_argument(
        "--shard_size", type=int, default=2000,
        help="Patches per shard file.",
    )
    p.add_argument(
        "--workers", type=int, default=max(1, mp.cpu_count() - 1),
        help="Worker processes for normalisation.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N patches per split (dry-run / benchmark).",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Process 200 patches, save a before/after PNG, do not write shards.",
    )
    p.add_argument(
        "--subset_dir", type=Path, default=cfg.SUBSET_DIR,
        help="Directory containing X_{split}.npy files.",
    )
    p.add_argument(
        "--cache_dir",  type=Path, default=cfg.CACHE_DIR,
        help="Root output directory for shards.",
    )
    p.add_argument(
        "--seed", type=int, default=cfg.SEED,
        help="Seed for reference sampling.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    methods = args.method
    splits  = args.splits

    print(f"subset_dir : {args.subset_dir}")
    print(f"cache_dir  : {args.cache_dir}")
    print(f"methods    : {methods}")
    print(f"splits     : {splits}")
    print(f"shard_size : {args.shard_size}")
    print(f"workers    : {args.workers}")

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── fit / load reference ─────────────────────────────────────────────────
    print("\nReference:")
    ref = load_or_build_reference(args.subset_dir, args.cache_dir, args.seed)

    # ── dry run ──────────────────────────────────────────────────────────────
    if args.dry_run:
        for method in methods:
            print(f"\n── dry run: {method} ──")
            _dry_run_figure(method, ref, args.subset_dir,
                            cfg.FIGURES_DIR, args.seed)
        return

    # ── main loop ────────────────────────────────────────────────────────────
    all_failures: List[dict] = []

    for method in methods:
        method_dir = args.cache_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}\nmethod: {method}\n{'='*60}")

        for split in splits:
            src_X = args.subset_dir / f"X_{split}.npy"
            if not src_X.exists():
                print(f"  {split}: {src_X} not found — skipping")
                continue
            _process_split(
                method, split,
                args.subset_dir, method_dir, ref,
                args.shard_size, args.workers,
                args.limit, all_failures,
            )

    # ── write failures log ───────────────────────────────────────────────────
    if all_failures:
        fail_path = args.cache_dir / "failures.csv"
        fieldnames = ["method", "split", "shard", "patch_pos", "orig_idx", "error"]
        write_header = not fail_path.exists()
        with open(fail_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            w.writerows(all_failures)
        print(f"\n{len(all_failures)} failure(s) logged → {fail_path}")
    else:
        print("\nNo failures.")

    print("\nDone.")


if __name__ == "__main__":
    main()
