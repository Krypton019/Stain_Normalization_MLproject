"""Copy result files into docs/results/ so GitHub Pages can serve them.

Run this after training completes (locally or after pulling from Kaggle):
    python scripts/publish_results.py

What gets copied
────────────────
  results/*.json         → docs/results/
  results/aggregate.csv  → docs/results/
  results/exp1_*.csv     → docs/results/
  results/exp2_*.csv     → docs/results/
  figures/*.png          → docs/figures/   (experiment figures shown inline)
"""

from __future__ import annotations
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import config as cfg

DOCS_RESULTS = _REPO / "docs" / "results"
DOCS_FIGURES = _REPO / "docs" / "figures"

DOCS_RESULTS.mkdir(parents=True, exist_ok=True)
DOCS_FIGURES.mkdir(parents=True, exist_ok=True)

copied = 0

# JSON training results
for src in sorted(cfg.RESULTS_DIR.glob("*.json")):
    dst = DOCS_RESULTS / src.name
    shutil.copy2(src, dst)
    print(f"  {src.name}")
    copied += 1

# CSVs (aggregate + experiments)
for pattern in ("aggregate.csv", "exp1_*.csv", "exp2_*.csv"):
    for src in sorted(cfg.RESULTS_DIR.glob(pattern)):
        dst = DOCS_RESULTS / src.name
        shutil.copy2(src, dst)
        print(f"  {src.name}")
        copied += 1

# Experiment figures
for src in sorted(cfg.FIGURES_DIR.glob("*.png")):
    dst = DOCS_FIGURES / src.name
    shutil.copy2(src, dst)
    print(f"  figures/{src.name}")
    copied += 1

print(f"\nCopied {copied} file(s) to docs/.")
print("Now commit and push:\n  git add docs/ && git commit -m 'update results' && git push")
