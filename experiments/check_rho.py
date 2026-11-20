import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as cfg
import numpy as np
from src.data import load_wilds
from src import stainops as so
from src.ot_norm import OTNormalizer

ref_imgs, _, _ = load_wilds([0], 20, seed=0)
ref_mean = np.mean([i[so.tissue_mask(i)].mean(axis=0) for i in ref_imgs
                    if so.tissue_mask(i).sum() > 200], axis=0)
x, _, _ = load_wilds([2], 20, seed=1)
masks = [so.tissue_mask(i) for i in x]
use = [k for k, m in enumerate(masks) if m.sum() > 200]

print("hospital 2, distance to reference colour after normalising (balanced gave 10.1)")
for rho in (0.1, 1, 10, 100, 1000):
    nm = OTNormalizer(mode="unbalanced", rho=rho).fit(ref_imgs)
    outs = [nm.transform(x[k]) for k in use]
    d = np.mean([np.linalg.norm(o[masks[k]].mean(axis=0) - ref_mean) for o, k in zip(outs, use)])
    print(f"rho={rho:>6}: {d:6.1f}   nonconverged={nm.n_nonconverged}  slow_path={nm.n_slow_path}")
