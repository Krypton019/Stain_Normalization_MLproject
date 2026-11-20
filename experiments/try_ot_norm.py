import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as cfg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import load_wilds
from src.ot_norm import OTNormalizer

cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ref_imgs, _, _ = load_wilds([0], 20, seed=0)               # reference = hospital 0
norms = {m: OTNormalizer(mode=m).fit(ref_imgs) for m in ("balanced", "unbalanced")}

fig, axes = plt.subplots(4, 4, figsize=(8.5, 8.6))
for r, h in enumerate([1, 2, 3, 4]):
    img = load_wilds([h], 1, seed=1)[0][0]
    panels = [("source", img), ("reference", ref_imgs[0])]
    for name, nm in norms.items():
        t = time.time()
        out = nm.transform(img)
        dt = time.time() - t
        print(f"hospital {h} {name}: {dt:.2f}s")
        panels.append((name, out))
    for c, (title, im) in enumerate(panels):
        axes[r, c].imshow(im)
        axes[r, c].axis("off")
        if r == 0:
            axes[r, c].set_title(title, fontsize=9)
    axes[r, 0].text(-0.1, 0.5, f"Hosp {h}", transform=axes[r, 0].transAxes,
                    ha="right", va="center", fontsize=9)
fig.savefig(cfg.FIGURES_DIR / "ot_norm_demo.png", dpi=150, bbox_inches="tight")
print(f"saved {cfg.FIGURES_DIR / 'ot_norm_demo.png'}")
