"""Train a ResNet-18 classifier on GW-normalised patches.

This script is a thin, GW-aware wrapper around the team's scripts/train.py.
It calls that script with --method ot_gw so that all training logic (AMP,
checkpointing, cosine LR, JSON output) is shared without modification.

Optionally it can first run prepare_data.py to generate the ot_gw cache
before training, making this a single entry-point for the full GW pipeline.

Outputs (independent, never overwrites team results):
    gw_extension/results/ot_gw_seed{seed}.json  — GW AUC / accuracy
    CACHE_DIR/ot_gw/ckpt_seed{seed}.pt          — checkpoint

Usage
─────
    # 1. Pilot: 200 patches, verify pipeline end-to-end
    python gw_extension/train_gw.py --pilot

    # 2. Full run with defaults (3 seeds)
    python gw_extension/train_gw.py

    # 3. Single seed, skip data prep (cache already built)
    python gw_extension/train_gw.py --seed 1 --skip_data_prep

    # 4. Full control
    python gw_extension/train_gw.py --seeds 0 1 2 --epochs 15 --batch_size 256
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── repo-root bootstrap ──────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gw_extension import config as gw_cfg


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], desc: str) -> int:
    """Run a subprocess, stream stdout/stderr to the console, return exit code."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  {' '.join(str(c) for c in cmd)}")
    print("="*60)
    result = subprocess.run(cmd, check=False)
    return result.returncode


def _ensure_data(subset_dir: Path, limit: int | None, splits: list[str],
                 n_workers: int) -> int:
    """Run prepare_data.py to build the ot_gw cache."""
    cmd = [
        sys.executable,
        str(_REPO / "gw_extension" / "prepare_data.py"),
        "--subset_dir", str(subset_dir),
        "--out_dir",    str(gw_cfg.GW_CACHE_DIR),
        "--splits",    *splits,
        "--n_workers", str(n_workers),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    return _run(cmd, "Step 1 — Build GW-normalised patch cache")


def _train_seed(seed: int, epochs: int, batch_size: int, lr: float,
                lr_sched: str, backbone: str, subsample: int | None,
                color_aug: bool, results_dir: Path, no_checkpoint: bool) -> int:
    """Run the team's train.py with method=ot_gw for one seed."""
    cmd = [
        sys.executable,
        str(_REPO / "scripts" / "train.py"),
        "--method",      "ot_gw",
        "--seed",        str(seed),
        "--epochs",      str(epochs),
        "--batch_size",  str(batch_size),
        "--lr",          str(lr),
        "--backbone",    backbone,
        "--lr_sched",    lr_sched,
        "--cache_dir",   str(gw_cfg.CACHE_DIR),
        "--data_dir",    str(gw_cfg.CACHE_DIR),
        "--ckpt_dir",    str(gw_cfg.CACHE_DIR),
        "--results_dir", str(results_dir),
        "--weight_decay", "1e-4",
    ]
    if subsample is not None:
        cmd += ["--subsample", str(subsample)]
    if color_aug:
        cmd.append("--color_aug")
    if no_checkpoint:
        cmd.append("--no_checkpoint")
    return _run(cmd, f"Step 2 — Train ResNet-18 (ot_gw, seed={seed})")


def _copy_to_gw_results(main_results_dir: Path, gw_results_dir: Path,
                         seeds: list[int]) -> None:
    """Copy the JSON output files the team's train.py wrote into gw_extension/results/."""
    gw_results_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        src = main_results_dir / f"ot_gw_seed{seed}.json"
        dst = gw_results_dir / f"ot_gw_seed{seed}.json"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  copied {src.name} → {dst}")
        else:
            print(f"  warning: expected result file not found: {src}")


def _aggregate(gw_results_dir: Path) -> None:
    """Print a quick mean ± std summary from GW result JSONs."""
    import numpy as np

    rows = []
    for path in sorted(gw_results_dir.glob("ot_gw_seed*.json")):
        with open(path) as fh:
            rows.append(json.load(fh))

    if not rows:
        print("No result files found — nothing to aggregate.")
        return

    metrics = ["val_auc", "test_auc", "test_acc", "runtime_s"]
    print("\n" + "="*55)
    print(f"{'GW Results':^55}")
    print("="*55)
    print(f"  Seeds: {len(rows)}")
    for m in metrics:
        vals = [r[m] for r in rows if r.get(m) is not None]
        if vals:
            mean, std = float(np.mean(vals)), float(np.std(vals))
            print(f"  {m:<15}: {mean:.4f} ± {std:.4f}")
    print("="*55)

    # Save aggregated CSV
    import pandas as pd
    summary_rows = []
    for m in metrics:
        vals = [r[m] for r in rows if r.get(m) is not None]
        summary_rows.append({
            "method": "ot_gw",
            "metric": m,
            "mean":   float(np.mean(vals)) if vals else float("nan"),
            "std":    float(np.std(vals))  if len(vals) > 1 else float("nan"),
            "n_seeds": len(vals),
        })
    out_csv = gw_results_dir / "ot_gw_aggregate.csv"
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print(f"  → {out_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train ResNet-18 on GW-normalised patches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                   help="Random seeds (matches team protocol).")
    p.add_argument("--seed",  type=int, default=None,
                   help="Run a single seed (overrides --seeds).")
    p.add_argument("--epochs",     type=int,   default=gw_cfg.DEFAULT_EPOCHS)
    p.add_argument("--batch_size", type=int,   default=gw_cfg.DEFAULT_BATCH_SIZE)
    p.add_argument("--lr",         type=float, default=gw_cfg.DEFAULT_LR)
    p.add_argument("--lr_sched",   default="cosine", choices=["cosine", "step"])
    p.add_argument("--backbone",   default=gw_cfg.DEFAULT_BACKBONE,
                   choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--color_aug",  action="store_true")
    p.add_argument("--subsample",  type=int, default=None,
                   help="Subsample training patches (pilot).")

    # Data prep
    p.add_argument("--skip_data_prep", action="store_true",
                   help="Skip prepare_data.py (use existing cache).")
    p.add_argument("--subset_dir", type=Path, default=gw_cfg.SUBSET_DIR)
    p.add_argument("--splits",     nargs="+", default=["train", "val", "test"])
    p.add_argument("--limit",      type=int,  default=None,
                   help="Limit patches per split during data prep.")
    p.add_argument("--n_workers",  type=int,  default=1,
                   help="Parallel workers for data preparation.")

    # Pilot shortcut
    p.add_argument("--pilot", action="store_true",
                   help="Quick end-to-end test: 200 patches, 2 epochs, seed 0.")

    # Output
    p.add_argument("--results_dir", type=Path, default=None,
                   help="Where the team's train.py writes JSON (default: RESULTS_DIR).")
    p.add_argument("--no_checkpoint", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.pilot:
        args.seeds = [0]
        args.epochs = 2
        args.subsample = 200
        args.limit = 200
        args.splits = ["train", "val", "test"]
        print("PILOT MODE: 200 patches, 2 epochs, seed 0")

    if args.seed is not None:
        args.seeds = [args.seed]

    # Where the team's train.py writes its JSON files
    main_results_dir = args.results_dir or gw_cfg.BASE_RESULTS
    main_results_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ── Step 1: Data preparation ─────────────────────────────────────────────
    if not args.skip_data_prep:
        rc = _ensure_data(
            subset_dir=args.subset_dir,
            limit=args.limit,
            splits=args.splits,
            n_workers=args.n_workers,
        )
        if rc != 0:
            print(f"\n[ERROR] prepare_data.py exited with code {rc}")
            sys.exit(rc)
    else:
        ot_gw_dir = gw_cfg.GW_CACHE_DIR
        if not (ot_gw_dir / "X_train.npy").exists():
            print(f"[ERROR] --skip_data_prep set but {ot_gw_dir}/X_train.npy not found.")
            print("       Run without --skip_data_prep first.")
            sys.exit(1)
        print(f"[skip] Using existing cache at {ot_gw_dir}")

    # ── Step 2: Training (one process per seed) ───────────────────────────────
    failed_seeds = []
    for seed in args.seeds:
        rc = _train_seed(
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_sched=args.lr_sched,
            backbone=args.backbone,
            subsample=args.subsample,
            color_aug=args.color_aug,
            results_dir=main_results_dir,
            no_checkpoint=args.no_checkpoint,
        )
        if rc != 0:
            print(f"[WARNING] train.py exited with code {rc} for seed {seed}")
            failed_seeds.append(seed)

    # ── Step 3: Copy results into gw_extension/results/ ──────────────────────
    _copy_to_gw_results(main_results_dir, gw_cfg.GW_RESULTS_DIR, args.seeds)

    # ── Step 4: Quick aggregate ───────────────────────────────────────────────
    _aggregate(gw_cfg.GW_RESULTS_DIR)

    total = time.time() - t0
    print(f"\nTotal wall time: {total/60:.1f} min")

    if failed_seeds:
        print(f"[WARNING] The following seeds had non-zero exit codes: {failed_seeds}")
        sys.exit(1)


if __name__ == "__main__":
    main()
