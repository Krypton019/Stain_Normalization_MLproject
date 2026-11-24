"""Structural and runtime evaluation for the GW normaliser.

Measures four independent quantities on a sample of patches:
  - Edge Preservation Index (EPI): gradient-field correlation, [0, 1]
  - Structural Similarity (SSIM): grayscale SSIM, [0, 1]
  - GW Distortion Objective: sum_ijkl |C_s(i,k)-C_t(j,l)|^2 pi_ij pi_kl
  - Runtime per patch (seconds)

None of these touch the team's existing code.

Usage
─────
    # Quick sanity check on 20 synthetic patches
    python gw_extension/evaluate.py --pilot

    # Evaluate on the first 200 patches of the train split
    python gw_extension/evaluate.py --limit 200

    # Full evaluation across all splits
    python gw_extension/evaluate.py --splits train val test

    # Save figure
    python gw_extension/evaluate.py --pilot --save_fig

Outputs
───────
    gw_extension/results/structural_metrics.csv
    gw_extension/figures/structural_eval.png  (if --save_fig)
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gw_extension import config as gw_cfg
from gw_extension import gw_utils
from gw_extension.gw_normalizer import GWNormalizer
from gw_extension.prepare_data import load_or_create_reference


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_patches(
    normalizer: GWNormalizer,
    patches: np.ndarray,
    verbose: bool = True,
) -> list[dict]:
    """Return per-patch structural metrics for a uint8 (N, H, W, 3) array."""
    records = []
    n = len(patches)
    for i, patch in enumerate(patches):
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            norm = normalizer.transform(patch)
        elapsed = time.perf_counter() - t0

        epi  = gw_utils.edge_preservation_index(patch, norm)
        ssim = gw_utils.structural_similarity_score(patch, norm)

        # GW distortion for this patch — recompute the last transport plan
        # via a lightweight re-solve at reduced n_sub for speed
        gw_obj = _gw_distortion_for_patch(normalizer, patch)

        records.append({
            "epi":        epi,
            "ssim":       ssim,
            "gw_obj":     gw_obj,
            "runtime_s":  elapsed,
        })

        if verbose and ((i + 1) % 10 == 0 or (i + 1) == n):
            mean_epi  = np.mean([r["epi"]  for r in records])
            mean_ssim = np.mean([r["ssim"] for r in records])
            mean_rt   = np.mean([r["runtime_s"] for r in records])
            print(
                f"  [{i+1:>4}/{n}]  "
                f"EPI={mean_epi:.3f}  SSIM={mean_ssim:.3f}  "
                f"rt={mean_rt*1000:.0f}ms",
                end="\r", flush=True,
            )
    if verbose:
        print()
    return records


def _gw_distortion_for_patch(normalizer: GWNormalizer, patch: np.ndarray) -> float:
    """Compute the GW structural distortion for one patch (uses a compact sub-solve)."""
    from gw_extension import gw_solver

    m = gw_utils.tissue_mask(patch)
    if m.sum() < normalizer.min_tissue:
        return float("nan")

    # Use a smaller n_sub for speed (50 points is enough to measure distortion)
    n_eval = min(50, normalizer.n_sub)
    coords_s, _, morpho_s = gw_utils.extract_tissue_points(
        patch, n_points=n_eval, seed=normalizer.seed
    )

    C_s = gw_utils.build_distance_matrix(
        coords_s, morpho_s, mode=normalizer.struct_mode
    )

    # Sample matching number of reference points
    rng = np.random.default_rng(normalizer.seed)
    ref_idx = rng.choice(len(normalizer.C_ref), size=n_eval, replace=False)
    C_t_sub = normalizer.C_ref[np.ix_(ref_idx, ref_idx)]

    od = gw_utils.rgb_to_od(patch)
    from gw_extension.gw_normalizer import macenko_stain_matrix, concentrations
    W = macenko_stain_matrix(patch)
    if np.linalg.cond(W) > normalizer.cond_max:
        W = normalizer.W_ref

    H_shape, W_shape = patch.shape[:2]
    y_s = np.clip(np.round(coords_s[:, 0] * (H_shape - 1)).astype(int), 0, H_shape - 1)
    x_s = np.clip(np.round(coords_s[:, 1] * (W_shape - 1)).astype(int), 0, W_shape - 1)
    H_s = concentrations(od, W)[y_s, x_s]

    H_ref_sub = normalizer.H_ref[ref_idx]
    X = H_s / normalizer.scale
    Y = H_ref_sub / normalizer.scale
    diff = X[:, None, :] - Y[None, :, :]
    M = np.sum(diff**2, axis=-1)

    pi, _ = gw_solver.unbalanced_fused_gromov_wasserstein(
        M=M, C_s=C_s, C_t=C_t_sub,
        alpha=normalizer.alpha, eps=normalizer.eps,
        rho=normalizer.rho, max_iter=20,
    )
    return gw_utils.gw_distortion_objective(C_s, C_t_sub, pi)


# ─────────────────────────────────────────────────────────────────────────────
# Summary helpers
# ─────────────────────────────────────────────────────────────────────────────

def summarise(records: list[dict]) -> dict:
    keys = ["epi", "ssim", "gw_obj", "runtime_s"]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in records if not np.isnan(r.get(k, float("nan")))])
        out[f"{k}_mean"] = float(np.mean(vals))  if len(vals) else float("nan")
        out[f"{k}_std"]  = float(np.std(vals))   if len(vals) > 1 else float("nan")
        out[f"{k}_med"]  = float(np.median(vals)) if len(vals) else float("nan")
    out["n_patches"] = len(records)
    return out


def print_summary(summary: dict, split: str = "") -> None:
    label = f" [{split}]" if split else ""
    print(f"\n{'='*55}")
    print(f"  Structural Evaluation{label}  (n={summary['n_patches']})")
    print("="*55)
    for k in ["epi", "ssim", "gw_obj", "runtime_s"]:
        mean = summary[f"{k}_mean"]
        std  = summary[f"{k}_std"]
        if not np.isnan(mean):
            print(f"  {k:<12}: {mean:.4f} ± {std:.4f}  (median {summary[f'{k}_med']:.4f})")
    print("="*55)


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def save_figure(
    orig_patches: np.ndarray,
    normalizer: GWNormalizer,
    out_path: Path,
    n_show: int = 6,
) -> None:
    import matplotlib.pyplot as plt

    n_show = min(n_show, len(orig_patches))
    fig, axes = plt.subplots(n_show, 3, figsize=(10, 3.5 * n_show),
                             gridspec_kw={"hspace": 0.4})

    if n_show == 1:
        axes = axes[None, :]

    for i in range(n_show):
        orig = orig_patches[i]
        t0 = time.perf_counter()
        norm = normalizer.transform(orig)
        rt   = (time.perf_counter() - t0) * 1000

        epi  = gw_utils.edge_preservation_index(orig, norm)
        ssim = gw_utils.structural_similarity_score(orig, norm)
        diff = np.abs(orig.astype(float) - norm.astype(float)).mean(axis=-1)

        axes[i, 0].imshow(orig)
        axes[i, 0].set_title(f"Original #{i}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(norm)
        axes[i, 1].set_title(
            f"OT-GW Normalised  ({rt:.0f} ms)\n"
            f"EPI={epi:.3f}  SSIM={ssim:.3f}",
            fontsize=8,
        )
        axes[i, 1].axis("off")

        im = axes[i, 2].imshow(diff, cmap="magma")
        axes[i, 2].set_title("Mean |orig − norm|", fontsize=9)
        axes[i, 2].axis("off")
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

    fig.suptitle(
        "GW Stain Normalisation — Structural Evaluation",
        fontsize=11, fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate GW normaliser — structural metrics and runtime.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--subset_dir",  type=Path, default=gw_cfg.SUBSET_DIR)
    p.add_argument("--splits",      nargs="+", default=["train"])
    p.add_argument("--limit",       type=int,  default=100,
                   help="Max patches per split to evaluate.")
    p.add_argument("--save_fig",    action="store_true")
    p.add_argument("--results_dir", type=Path, default=gw_cfg.GW_RESULTS_DIR)
    p.add_argument("--figures_dir", type=Path, default=gw_cfg.GW_FIGURES_DIR)
    p.add_argument("--pilot",       action="store_true",
                   help="Quick test on 20 synthetic patches.")
    return p.parse_args()


def main() -> None:
    import pandas as pd

    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    # Load normalizer
    print("Loading GW normalizer reference...")
    normalizer = load_or_create_reference(
        subset_dir=args.subset_dir,
        cache_dir=gw_cfg.GW_CACHE_DIR,
    )

    all_rows = []

    if args.pilot:
        # Generate synthetic pilot patches
        from gw_extension.prepare_data import generate_pilot_subset
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            generate_pilot_subset(tmp_path, n_per_hosp=4)
            X = np.load(tmp_path / "X_train.npy")
        args.splits = ["(pilot)"]
        print(f"\nEvaluating {len(X)} pilot patches...")
        records = evaluate_patches(normalizer, X)
        summary = summarise(records)
        print_summary(summary, split="pilot")
        for r in records:
            r["split"] = "pilot"
        all_rows.extend(records)
        if args.save_fig:
            save_figure(X, normalizer,
                        args.figures_dir / "structural_eval.png")
    else:
        for split in args.splits:
            x_path = args.subset_dir / f"X_{split}.npy"
            if not x_path.exists():
                print(f"  skip {split}: {x_path} not found")
                continue

            X = np.load(x_path, mmap_mode="r")
            if args.limit and args.limit < len(X):
                X = X[:args.limit]

            print(f"\nEvaluating {len(X)} patches from split '{split}'...")
            records = evaluate_patches(normalizer, X)
            summary = summarise(records)
            print_summary(summary, split=split)
            for r in records:
                r["split"] = split
            all_rows.extend(records)

            if args.save_fig:
                save_figure(X[:min(6, len(X))], normalizer,
                            args.figures_dir / f"structural_eval_{split}.png")

    if not all_rows:
        print("No patches evaluated.")
        return

    # Save CSV
    out_csv = args.results_dir / "structural_metrics.csv"
    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    print(f"\nPer-patch metrics saved → {out_csv}")

    # Also save summary
    summary_all = summarise(all_rows)
    summary_all["split"] = "all"
    summary_csv = args.results_dir / "structural_metrics_summary.csv"
    pd.DataFrame([summary_all]).to_csv(summary_csv, index=False)
    print(f"Summary saved → {summary_csv}")


if __name__ == "__main__":
    main()
