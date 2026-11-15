import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.data import load_wilds
from src import stainops as so
from src.ot_norm import OTNormalizer


def mean_rgb(img, mask):
    return img[mask].mean(axis=0)


ref_imgs, _, _ = load_wilds([0], 20, seed=0)
ref_mean = np.mean([mean_rgb(i, so.tissue_mask(i)) for i in ref_imgs
                    if so.tissue_mask(i).sum() > 200], axis=0)

print(f"{'mode':>10} {'hosp':>4} {'s/img':>6} {'pixels changed':>15} "
      f"{'dist before':>11} {'dist after':>10} {'unstableW':>9} {'nonconv':>7} {'slow':>4}")
for mode in ("balanced", "unbalanced"):
    nm = OTNormalizer(mode=mode).fit(ref_imgs)
    for h in (1, 2, 3, 4):
        x, _, _ = load_wilds([h], 20, seed=1)
        nm.n_images = nm.n_fallback = nm.n_nonconverged = nm.n_slow_path = 0
        t = time.time()
        outs = [nm.transform(i) for i in x]
        dt = (time.time() - t) / len(x)
        masks = [so.tissue_mask(i) for i in x]
        use = [k for k, m in enumerate(masks) if m.sum() > 200]
        before = np.mean([np.linalg.norm(mean_rgb(x[k], masks[k]) - ref_mean) for k in use])
        after = np.mean([np.linalg.norm(mean_rgb(outs[k], masks[k]) - ref_mean) for k in use])
        changed = np.mean([(outs[k] != x[k]).any(-1).mean() for k in use])
        print(f"{mode:>10} {h:>4} {dt:>6.3f} {changed:>15.2f} {before:>11.1f} {after:>10.1f} "
              f"{nm.n_fallback:>9d} {nm.n_nonconverged:>7d} {nm.n_slow_path:>4d}")