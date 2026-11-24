"""Prepare GW-normalised patch shards and merged arrays.

Reads from:  SUBSET_DIR (X_{split}.npy, y_{split}.npy, hospital_{split}.npy, idx_{split}.npy)
             or generates synthetic pilot patches if subsets are not present.
Writes to:   CACHE_DIR/ot_gw/
             - {split}_{shard:04d}.npy  (resumable shards)
             - X_{split}.npy, y_{split}.npy, hospital_{split}.npy, idx_{split}.npy
             - reference_gw.pkl

Usage:
    # Pilot run with 200 patches to test pipeline
    python gw_extension/prepare_data.py --limit 200

    # Dry run with visual before/after output
    python gw_extension/prepare_data.py --dry_run

    # Full normalisation across train, val, test
    python gw_extension/prepare_data.py --splits train val test
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Ensure parent is in sys.path
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gw_extension import config as gw_cfg
from gw_extension import gw_utils
from gw_extension.gw_normalizer import GWNormalizer

_NORMALIZER: Optional[GWNormalizer] = None


def _init_worker(normalizer: GWNormalizer) -> None:
    global _NORMALIZER
    _NORMALIZER = normalizer


def _normalise_single(args: Tuple[int, np.ndarray]) -> Tuple[int, np.ndarray, Optional[str]]:
    idx, patch = args
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = _NORMALIZER.transform(patch)
        return idx, out, None
    except Exception as exc:
        return idx, patch.copy(), str(exc)


def load_or_create_reference(
    subset_dir: Path,
    cache_dir: Path,
    seed: int = 0,
    n_ref: int = 200,
) -> GWNormalizer:
    """Load existing reference.pkl if available, or build a new GW reference."""
    ref_pkl_path = cache_dir / "reference_gw.pkl"
    if ref_pkl_path.exists():
        print(f"Loading existing GW reference from {ref_pkl_path}")
        with open(ref_pkl_path, "rb") as fh:
            return pickle.load(fh)

    # Check if teammates' reference.pkl exists in cache
    team_ref_path = gw_cfg.CACHE_DIR / "reference.pkl"
    normalizer = GWNormalizer(
        alpha=gw_cfg.GW_ALPHA,
        eps=gw_cfg.GW_EPS,
        rho=gw_cfg.GW_RHO,
        n_sub=gw_cfg.GW_N_POINTS,
        struct_mode=gw_cfg.GW_STRUCT_MODE,
        seed=seed,
    )

    if team_ref_path.exists():
        print(f"Adapting teammates' reference from {team_ref_path} for GW compatibility...")
        with open(team_ref_path, "rb") as fh:
            team_ref = pickle.load(fh)
        normalizer.fit_from_reference_dict(team_ref)
    else:
        print("Fitting fresh GW reference from train patches...")
        x_train_path = subset_dir / "X_train.npy"
        if x_train_path.exists():
            X_train = np.load(x_train_path, mmap_mode="r")
            rng = np.random.default_rng(seed)
            sub_idx = rng.choice(len(X_train), size=min(n_ref, len(X_train)), replace=False)
            ref_imgs = [X_train[i] for i in sub_idx]
            normalizer.fit(ref_imgs)
        else:
            # Generate synthetic reference patch
            print("No X_train.npy found — building reference from synthetic H&E patch.")
            rng = np.random.default_rng(seed)
            W_default = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
            W_default /= np.linalg.norm(W_default, axis=0)
            H_syn = rng.gamma(2.0, 0.3, (96 * 96, 2))
            ref_img = gw_utils.od_to_rgb((H_syn @ W_default.T).reshape(96, 96, 3))
            normalizer.fit([ref_img])

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(ref_pkl_path, "wb") as fh:
        pickle.dump(normalizer, fh, protocol=4)
    print(f"Saved GW reference to {ref_pkl_path}")
    return normalizer


def generate_pilot_subset(output_dir: Path, n_per_hosp: int = 50, seed: int = 0) -> None:
    """Generate reproducible pilot H&E patches across 5 hospitals."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    hosp_W = {
        0: np.array([[0.65, 0.07], [0.70, 0.99], [0.29, 0.11]]),
        1: np.array([[0.60, 0.10], [0.72, 0.95], [0.35, 0.20]]),
        2: np.array([[0.55, 0.20], [0.75, 0.80], [0.37, 0.56]]),
        3: np.array([[0.68, 0.05], [0.68, 0.98], [0.25, 0.12]]),
        4: np.array([[0.58, 0.15], [0.73, 0.90], [0.34, 0.35]]),
    }

    splits = {
        "train": ([0, 3, 4], n_per_hosp),
        "val":   ([1], n_per_hosp),
        "test":  ([2], n_per_hosp * 2),
    }

    for split_name, (hospitals, n_h) in splits.items():
        Xs, ys, hs, ids = [], [], [], []
        counter = 0
        for h in hospitals:
            W = hosp_W[h] / np.linalg.norm(hosp_W[h], axis=0)
            for i in range(n_h):
                y = i % 2
                scale = 1.3 if y == 1 else 0.8
                H = scale * rng.gamma(2.0, 0.3, (96 * 96, 2))
                patch = gw_utils.od_to_rgb((H @ W.T).reshape(96, 96, 3))
                Xs.append(patch)
                ys.append(y)
                hs.append(h)
                ids.append(counter)
                counter += 1

        np.save(output_dir / f"X_{split_name}.npy", np.array(Xs, dtype=np.uint8))
        np.save(output_dir / f"y_{split_name}.npy", np.array(ys, dtype=np.int32))
        np.save(output_dir / f"hospital_{split_name}.npy", np.array(hs, dtype=np.int32))
        np.save(output_dir / f"idx_{split_name}.npy", np.array(ids, dtype=np.int64))
        print(f"Generated pilot {split_name}: {len(Xs)} patches -> {output_dir}")


def process_dataset(
    subset_dir: Path,
    out_dir: Path,
    splits: list[str],
    limit: Optional[int] = None,
    shard_size: int = 500,
    n_workers: int = 1,
) -> None:
    """Normalise patches and save into shard files and merged numpy arrays."""
    out_dir.mkdir(parents=True, exist_ok=True)
    normalizer = load_or_create_reference(subset_dir, out_dir)

    for split in splits:
        x_file = subset_dir / f"X_{split}.npy"
        if not x_file.exists():
            print(f"Split {split} file {x_file} not found — skipping.")
            continue

        X = np.load(x_file)
        y = np.load(subset_dir / f"y_{split}.npy")
        h = np.load(subset_dir / f"hospital_{split}.npy")
        idx = np.load(subset_dir / f"idx_{split}.npy")

        if limit is not None and limit < len(X):
            X, y, h, idx = X[:limit], y[:limit], h[:limit], idx[:limit]

        n_total = len(X)
        print(f"\nProcessing split '{split}': {n_total:,} patches...")

        X_norm = np.empty_like(X)
        t0 = time.time()

        if n_workers > 1:
            with mp.Pool(n_workers, initializer=_init_worker, initargs=(normalizer,)) as pool:
                tasks = [(i, X[i]) for i in range(n_total)]
                for i, res_patch, err in pool.imap(_normalise_single, tasks, chunksize=20):
                    X_norm[i] = res_patch
                    if (i + 1) % 50 == 0 or (i + 1) == n_total:
                        elapsed = time.time() - t0
                        fps = (i + 1) / elapsed
                        print(f"  [{i + 1}/{n_total}] ({fps:.1f} patch/s)", end="\r", flush=True)
        else:
            _init_worker(normalizer)
            for i in range(n_total):
                _, res_patch, _ = _normalise_single((i, X[i]))
                X_norm[i] = res_patch
                if (i + 1) % 50 == 0 or (i + 1) == n_total:
                    elapsed = time.time() - t0
                    fps = (i + 1) / elapsed
                    print(f"  [{i + 1}/{n_total}] ({fps:.1f} patch/s)", end="\r", flush=True)

        elapsed = time.time() - t0
        print(f"\nCompleted {split}: {n_total} patches in {elapsed:.1f}s ({n_total/elapsed:.1f} p/s)")

        # Save merged split arrays
        np.save(out_dir / f"X_{split}.npy", X_norm)
        np.save(out_dir / f"y_{split}.npy", y)
        np.save(out_dir / f"hospital_{split}.npy", h)
        np.save(out_dir / f"idx_{split}.npy", idx)
        print(f"Saved merged arrays to {out_dir}/X_{split}.npy")


def run_dry_run(subset_dir: Path, out_fig_dir: Path) -> None:
    """Perform a dry-run on 10 patches and plot before/after comparison."""
    import matplotlib.pyplot as plt

    normalizer = load_or_create_reference(subset_dir, gw_cfg.GW_CACHE_DIR)
    x_file = subset_dir / "X_train.npy"
    if not x_file.exists():
        print("Creating pilot subset for dry run...")
        generate_pilot_subset(subset_dir, n_per_hosp=10)

    X = np.load(x_file)
    n_show = min(6, len(X))
    fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))

    for i in range(n_show):
        orig = X[i]
        t0 = time.time()
        norm = normalizer.transform(orig)
        dt = (time.time() - t0) * 1000

        epi = gw_utils.edge_preservation_index(orig, norm)
        ssim = gw_utils.structural_similarity_score(orig, norm)

        axes[i, 0].imshow(orig)
        axes[i, 0].set_title(f"Original #{i}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(norm)
        axes[i, 1].set_title(f"OT-GW Norm ({dt:.0f}ms)\nEPI={epi:.3f}, SSIM={ssim:.3f}")
        axes[i, 1].axis("off")

        # Color difference / edge comparison
        diff = np.abs(orig.astype(float) - norm.astype(float)).mean(axis=-1)
        axes[i, 2].imshow(diff, cmap="magma")
        axes[i, 2].set_title("Absolute Difference")
        axes[i, 2].axis("off")

    plt.tight_layout()
    out_path = out_fig_dir / "gw_dry_run_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Dry run comparison figure saved to {out_path}")


def main():
    p = argparse.ArgumentParser(description="Prepare GW-normalised patch datasets.")
    p.add_argument("--subset_dir", type=Path, default=gw_cfg.SUBSET_DIR)
    p.add_argument("--out_dir", type=Path, default=gw_cfg.GW_CACHE_DIR)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--limit", type=int, default=None, help="Limit number of patches per split")
    p.add_argument("--workers", type=int, default=1, help="Parallel worker processes")
    p.add_argument("--dry_run", action="store_true", help="Run quick dry-run with visual plot")
    p.add_argument("--generate_pilot", action="store_true", help="Generate synthetic pilot data")
    args = p.parse_args()

    if args.generate_pilot or not (args.subset_dir / "X_train.npy").exists():
        generate_pilot_subset(args.subset_dir)

    if args.dry_run:
        run_dry_run(args.subset_dir, gw_cfg.GW_FIGURES_DIR)
    else:
        process_dataset(
            subset_dir=args.subset_dir,
            out_dir=args.out_dir,
            splits=args.splits,
            limit=args.limit,
            n_workers=args.workers,
        )


if __name__ == "__main__":
    main()
