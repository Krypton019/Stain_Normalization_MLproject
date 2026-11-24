"""Compare GW results against team baselines (none, Reinhard, Macenko, OT-bal, OT-unbal).

Reads:
  - RESULTS_DIR/*_seed*.json         (team's training results)
  - gw_extension/results/ot_gw_seed*.json  (your GW results)
  - gw_extension/results/structural_metrics_summary.csv  (optional, adds EPI/SSIM columns)

Writes:
  - gw_extension/results/comparison_table.csv
  - gw_extension/figures/comparison_auc.png
  - gw_extension/figures/comparison_structural.png  (if structural_metrics_summary.csv exists)

Usage
─────
    python gw_extension/compare.py
    python gw_extension/compare.py --no_fig
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gw_extension import config as gw_cfg


# ─────────────────────────────────────────────────────────────────────────────
# Method display names and ordering
# ─────────────────────────────────────────────────────────────────────────────

METHOD_LABELS = {
    "none":        "No Norm",
    "reinhard":    "Reinhard",
    "macenko":     "Macenko",
    "ot_balanced": "OT-Balanced",
    "ot_bal":      "OT-Balanced",
    "ot_unbal":    "OT-Unbalanced",
    "ot_gw":       "OT-GW (ours)",
}

METHOD_ORDER = ["none", "reinhard", "macenko", "ot_bal", "ot_balanced",
                "ot_unbal", "ot_gw"]

METHOD_COLORS = {
    "none":        "#aaaaaa",
    "reinhard":    "#4e79a7",
    "macenko":     "#f28e2b",
    "ot_balanced": "#59a14f",
    "ot_bal":      "#59a14f",
    "ot_unbal":    "#e15759",
    "ot_gw":       "#76b7b2",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_all_results(
    team_results_dir: Path,
    gw_results_dir: Path,
) -> pd.DataFrame:
    rows = []

    # Team results
    for path in sorted(team_results_dir.glob("*_seed*.json")):
        try:
            with open(path) as fh:
                d = json.load(fh)
            rows.append({
                "method":    d.get("method", path.stem.rsplit("_seed", 1)[0]),
                "seed":      d.get("seed", -1),
                "val_auc":   d.get("val_auc"),
                "test_auc":  d.get("test_auc"),
                "test_acc":  d.get("test_acc"),
                "runtime_s": d.get("runtime_s"),
                "source":    "team",
            })
        except Exception as e:
            print(f"  warning: could not read {path}: {e}")

    # GW results
    for path in sorted(gw_results_dir.glob("ot_gw_seed*.json")):
        try:
            with open(path) as fh:
                d = json.load(fh)
            rows.append({
                "method":    "ot_gw",
                "seed":      d.get("seed", -1),
                "val_auc":   d.get("val_auc"),
                "test_auc":  d.get("test_auc"),
                "test_acc":  d.get("test_acc"),
                "runtime_s": d.get("runtime_s"),
                "source":    "gw",
            })
        except Exception as e:
            print(f"  warning: could not read {path}: {e}")

    if not rows:
        raise FileNotFoundError(
            f"No result files found.\n"
            f"  Team: {team_results_dir}\n"
            f"  GW:   {gw_results_dir}\n"
            "Run train_gw.py and/or the team's scripts/train.py first."
        )

    return pd.DataFrame(rows)


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["val_auc", "test_auc", "test_acc", "runtime_s"]
    rows = []
    for method, grp in df.groupby("method", sort=False):
        row = {"method": method}
        for m in metrics:
            vals = grp[m].dropna().values.astype(float)
            row[f"{m}_mean"] = float(np.mean(vals))  if len(vals) else float("nan")
            row[f"{m}_std"]  = float(np.std(vals))   if len(vals) > 1 else float("nan")
            row[f"{m}_n"]    = len(vals)
        rows.append(row)

    summary = pd.DataFrame(rows)
    # Sort by METHOD_ORDER, put unknowns at the end
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    summary["_order"] = summary["method"].map(lambda x: order.get(x, 99))
    summary = summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return summary


def _fmt(mean: float, std: float) -> str:
    if np.isnan(mean):
        return "   —"
    if np.isnan(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def print_comparison_table(summary: pd.DataFrame) -> None:
    print("\n" + "="*72)
    print(f"{'Method':<18} {'val_AUC':>18} {'test_AUC':>18} {'test_Acc':>14}")
    print("="*72)
    for _, r in summary.iterrows():
        label = METHOD_LABELS.get(r["method"], r["method"])
        marker = " ◄" if r["method"] == "ot_gw" else ""
        print(
            f"{label:<18} "
            f"{_fmt(r['test_auc_mean'], r['test_auc_std']):>18} "
            f"{_fmt(r['val_auc_mean'],  r['val_auc_std']):>18} "
            f"{_fmt(r['test_acc_mean'], r['test_acc_std']):>14}"
            f"{marker}"
        )
    print("="*72)


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_auc_comparison(
    summary: pd.DataFrame,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, ylabel in [
        (axes[0], "test_auc", "Test AUC"),
        (axes[1], "val_auc",  "Val AUC"),
    ]:
        methods = []
        means   = []
        errs    = []
        colors  = []

        for _, r in summary.iterrows():
            m = r["method"]
            mean = r[f"{metric}_mean"]
            std  = r[f"{metric}_std"]
            if np.isnan(mean):
                continue
            methods.append(METHOD_LABELS.get(m, m))
            means.append(mean)
            errs.append(std if not np.isnan(std) else 0.0)
            colors.append(METHOD_COLORS.get(m, "#999999"))

        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=errs, capsize=4,
                      color=colors, edgecolor="black", linewidth=0.6, alpha=0.85)

        # Highlight the GW bar with a thicker edge
        gw_label = METHOD_LABELS["ot_gw"]
        for bar, label in zip(bars, methods):
            if label == gw_label:
                bar.set_edgecolor("black")
                bar.set_linewidth(2.0)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

        # Annotate bar heights
        for bar, mean_val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{mean_val:.3f}",
                ha="center", va="bottom", fontsize=7.5,
            )

    fig.suptitle(
        "Stain Normalisation — AUC Comparison (mean ± std over 3 seeds)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"AUC comparison figure → {out_path}")


def plot_structural_comparison(
    struct_summary: pd.DataFrame,
    auc_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    """Side-by-side: GW structural metrics vs OT-unbalanced."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    gw_row  = struct_summary[struct_summary["split"] == "all"].iloc[0] \
              if "split" in struct_summary.columns \
              else struct_summary.iloc[0]

    ot_row_q  = auc_summary[auc_summary["method"] == "ot_unbal"]
    gw_auc_q  = auc_summary[auc_summary["method"] == "ot_gw"]

    for ax, (metric, label, fmt) in zip(axes, [
        ("epi_mean",     "Edge Preservation Index (EPI)",      ".3f"),
        ("ssim_mean",    "SSIM",                               ".3f"),
        ("runtime_s_mean", "Avg Runtime / patch  (s)",         ".2f"),
    ]):
        val = gw_row.get(metric, float("nan"))
        if np.isnan(val):
            ax.set_visible(False)
            continue
        bar = ax.bar(["OT-GW (ours)"], [val],
                     color=METHOD_COLORS["ot_gw"], edgecolor="black", linewidth=1.2)
        ax.set_title(label, fontsize=10)
        ax.set_ylim(0, max(val * 1.3, 0.1))
        ax.text(0, val + val * 0.03, f"{val:{fmt}}", ha="center", fontsize=10)

    fig.suptitle(
        "GW Extension — Structural Metrics",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Structural figure → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare GW against team baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--team_results_dir", type=Path, default=gw_cfg.BASE_RESULTS)
    p.add_argument("--gw_results_dir",   type=Path, default=gw_cfg.GW_RESULTS_DIR)
    p.add_argument("--figures_dir",      type=Path, default=gw_cfg.GW_FIGURES_DIR)
    p.add_argument("--no_fig",           action="store_true",
                   help="Skip figure generation (useful when matplotlib is unavailable).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.gw_results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Team results:  {args.team_results_dir}")
    print(f"GW results:    {args.gw_results_dir}")

    df = load_all_results(args.team_results_dir, args.gw_results_dir)
    print(f"\nLoaded {len(df)} run(s) across {df['method'].nunique()} method(s).")

    summary = compute_summary(df)
    print_comparison_table(summary)

    # Save CSV
    out_csv = args.gw_results_dir / "comparison_table.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\nComparison table → {out_csv}")

    if not args.no_fig:
        try:
            import matplotlib  # noqa: F401
            plot_auc_comparison(
                summary,
                out_path=args.figures_dir / "comparison_auc.png",
            )

            # Structural metrics figure (optional — only if evaluate.py has been run)
            struct_csv = args.gw_results_dir / "structural_metrics_summary.csv"
            if struct_csv.exists():
                struct_df = pd.read_csv(struct_csv)
                plot_structural_comparison(
                    struct_df, summary,
                    out_path=args.figures_dir / "comparison_structural.png",
                )
        except ImportError:
            warnings.warn("matplotlib not available — skipping figures.")


if __name__ == "__main__":
    main()
