import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # lets this script import from src/

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import load_wilds
from src import stainops as so

N_GRID, N_STATS = 8, 100
(ROOT / "figures").mkdir(exist_ok=True)

fig, axes = plt.subplots(5, N_GRID, figsize=(N_GRID * 1.6, 5 * 1.7))
print(f"{'hospital':>8} {'mean H':>8} {'mean E':>8} {'mean OD':>8}  (tissue pixels only)")

for h in range(5):
    x, y, _ = load_wilds([h], N_STATS, seed=0)
    for j in range(N_GRID):
        ax = axes[h, j]
        ax.imshow(x[j])
        ax.axis("off")
        if j == 0:
            ax.text(-0.08, 0.5, f"Hosp {h}", transform=ax.transAxes,
                    ha="right", va="center", fontsize=10)

    hs, es, ods = [], [], []
    for img in x:
        m = so.tissue_mask(img)
        if m.sum() < 100:
            continue
        od = so.rgb_to_od(img)
        W = so.macenko_stain_matrix(img)
        H = so.concentrations(od, W)
        hs.append(H[m, 0].mean())
        es.append(H[m, 1].mean())
        ods.append(od[m].sum(axis=-1).mean())
    print(f"{h:>8} {np.mean(hs):>8.3f} {np.mean(es):>8.3f} {np.mean(ods):>8.3f}")

fig.suptitle("CAMELYON17-WILDS: same task, five hospitals", y=0.94)
fig.savefig(ROOT / "figures" / "hospitals.png", dpi=150, bbox_inches="tight")
print("saved figures/hospitals.png")