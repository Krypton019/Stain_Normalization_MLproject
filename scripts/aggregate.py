"""Aggregate per-run JSON result files into a summary table.

Reads:   RESULTS_DIR/{method}_seed{seed}.json  (written by train.py)
Prints:  mean ± std per method for val_auc, test_acc, test_auc, runtime_s
Writes:  RESULTS_DIR/aggregate.csv

Usage
─────
    python scripts/aggregate.py
    python scripts/aggregate.py --results_dir /tmp/results
    python scripts/aggregate.py --metrics test_auc test_acc val_auc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import config as cfg

DEFAULT_METRICS = ["val_auc", "test_acc", "test_auc", "runtime_s"]


def load_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("*_seed*.json")):
        try:
            with open(path) as fh:
                d = json.load(fh)
        except Exception as e:
            print(f"  warning: could not read {path}: {e}")
            continue
        row = {
            "method":     d.get("method", path.stem),
            "seed":       d.get("seed",   -1),
            "best_epoch": d.get("best_epoch"),
        }
        for key in ("val_acc", "val_auc", "test_acc", "test_auc", "runtime_s"):
            row[key] = d.get(key)
        rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No *_seed*.json files found in {results_dir}.\n"
            "Run scripts/train.py first."
        )
    return pd.DataFrame(rows)


def _fmt(mean: float, std: float) -> str:
    if np.isnan(mean):
        return "   n/a"
    if np.isnan(std) or std == 0:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def summarise(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for method, grp in df.groupby("method", sort=False):
        row: dict = {"method": method, "n_seeds": len(grp)}
        for m in metrics:
            vals = grp[m].dropna().values.astype(float)
            row[f"{m}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            row[f"{m}_std"]  = float(np.std(vals))  if len(vals) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def print_table(summary: pd.DataFrame, metrics: list[str]) -> None:
    # Build a readable display table
    display_rows = []
    for _, r in summary.iterrows():
        row = {"method": r["method"], "n": int(r["n_seeds"])}
        for m in metrics:
            row[m] = _fmt(r[f"{m}_mean"], r[f"{m}_std"])
        display_rows.append(row)
    disp = pd.DataFrame(display_rows).set_index("method")

    sep = "─" * (14 + 20 * len(metrics))
    print(f"\n{sep}")
    print(disp.to_string())
    print(sep)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate train.py JSON results into mean ± std table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results_dir", type=Path, default=cfg.RESULTS_DIR)
    p.add_argument(
        "--metrics", nargs="+", default=DEFAULT_METRICS,
        help="Metrics to include in the summary.",
    )
    p.add_argument(
        "--csv", type=Path, default=None,
        help="Override output CSV path (default: RESULTS_DIR/aggregate.csv).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Reading results from {args.results_dir} …")
    df = load_results(args.results_dir)
    print(f"  {len(df)} run(s) across {df['method'].nunique()} method(s)")

    summary = summarise(df, args.metrics)
    print_table(summary, args.metrics)

    csv_path = args.csv or (args.results_dir / "aggregate.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}")


if __name__ == "__main__":
    main()
