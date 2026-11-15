import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.data import load_wilds
from src import stainops as so

print(f"{'hosp':>4} {'med p99 H':>10} {'max p99 H':>10} {'imgs p99>5':>11} "
      f"{'dark px %':>10} {'med cond W':>11} {'max cond W':>11}")
for h in range(5):
    x, _, _ = load_wilds([h], 100, seed=0)
    p99s, dark, conds = [], [], []
    for img in x:
        m = so.tissue_mask(img)
        if m.sum() < 200:
            continue
        od = so.rgb_to_od(img)
        W = so.macenko_stain_matrix(img)
        H = so.concentrations(od, W)
        p99s.append(np.percentile(H[m, 0], 99))
        dark.append((od.sum(-1)[m] > 6).mean())     # very dark pixels
        conds.append(np.linalg.cond(W))
    p99s = np.array(p99s)
    print(f"{h:>4} {np.median(p99s):>10.2f} {p99s.max():>10.2f} {(p99s > 5).sum():>11d} "
          f"{100 * np.mean(dark):>10.2f} {np.median(conds):>11.2f} {np.max(conds):>11.2f}")