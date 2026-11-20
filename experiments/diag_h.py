import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as cfg
import numpy as np
from src.data import load_wilds
from src import stainops as so

print(f"{'hosp':>4} {'pixels':>9} {'zeros%':>7} {'mean':>7} {'std':>7} {'p99':>7}")
for h in range(5):
    x, _, _ = load_wilds([h], 100, seed=0)
    vals = []
    for img in x:
        m = so.tissue_mask(img)
        if m.sum() < 200:
            continue
        H = so.concentrations(so.rgb_to_od(img), so.macenko_stain_matrix(img))
        vals.append(H[m, 0])
    v = np.concatenate(vals)
    print(f"{h:>4} {len(v):>9,} {100 * (v == 0).mean():>7.1f} {v.mean():>7.3f} {v.std():>7.3f} {np.percentile(v, 99):>7.3f}")
