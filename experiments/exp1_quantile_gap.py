import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gamma

from src import quantile as q
from src import stainops as so
from src.data import load_wilds

(ROOT / "figures").mkdir(exist_ok=True)
(ROOT / "results").mkdir(exist_ok=True)

U = q.quantile_grid(1000)
METHODS = ["affine_percentile", "macenko_scale", "affine_moment"]
rows = []


def three_maps(qs, qt):
    return {
        "affine_percentile": q.affine_percentile(qs, qt, U),
        "macenko_scale": q.scale_p99(qs, qt, U),
        "affine_moment": q.affine_moment(qs, qt),
    }


# ---------- (a) synthetic: Gamma source vs Gamma targets of different shape ----------
K1, TH1 = 2.0, 0.5                       # source: mean 1, skewness 2/sqrt(k)
qs = gamma.ppf(U, K1, scale=TH1)
skew_s = 2 / np.sqrt(K1)
ks = np.unique(np.append(np.logspace(np.log10(0.5), np.log10(30), 25), K1))

syn = {m: [] for m in METHODS}
skew_diffs = []
for k2 in ks:
    qt = gamma.ppf(U, k2, scale=1.0 / k2)  # target: mean 1, variance 1/k2
    ds = 2 / np.sqrt(k2) - skew_s
    skew_diffs.append(ds)
    rows.append(dict(dataset="synthetic", method="-", split=f"k2={k2:.3f}",
                     seed=0, metric="skew_diff", value=ds))
    for name, (a, b) in three_maps(qs, qt).items():
        rel = q.relative_gap(qs, qt, a, b)
        syn[name].append(rel)
        rows.append(dict(dataset="synthetic", method=name, split=f"k2={k2:.3f}",
                         seed=0, metric="rel_w2sq", value=rel))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
k_show = 10.0
qt_show = gamma.ppf(U, k_show, scale=1.0 / k_show)
ax1.plot(qs, qt_show, "k-", lw=2.5, label="optimal map T*")
for name, (a, b) in three_maps(qs, qt_show).items():
    ax1.plot(qs, a * qs + b, "--", label=name)
ax1.set_xlabel("source stain amount x")
ax1.set_ylabel("mapped amount T(x)")
ax1.set_title(f"Maps for Gamma(2) -> Gamma({k_show:g})")
ax1.legend(fontsize=8)

for name in METHODS:
    ax2.plot(skew_diffs, syn[name], "o-", ms=3, label=name)
ax2.axvline(0, color="gray", lw=0.8)
ax2.set_xlabel("skewness(target) - skewness(source)")
ax2.set_ylabel("relative gap  W2^2 / Var(target)")
ax2.set_title("Gap left by a straight line")
ax2.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "exp1_synthetic.png", dpi=150)
print("saved figures/exp1_synthetic.png")

# ---------- (b) real: hematoxylin amounts of each hospital ----------
N_PATCHES = 100
pools = {}
for h in range(5):
    x, _, _ = load_wilds([h], N_PATCHES, seed=0)
    vals = []
    for img in x:
        m = so.tissue_mask(img)
        if m.sum() < 200:
            continue
        od = so.rgb_to_od(img)
        W = so.macenko_stain_matrix(img)
        H = so.concentrations(od, W)
        vals.append(H[m, 0])               # hematoxylin amount of tissue pixels
    pools[h] = np.concatenate(vals)
    print(f"hospital {h}: {len(pools[h]):,} tissue pixels")

Q = {h: np.quantile(pools[h], U) for h in pools}
mats = {m: np.zeros((5, 5)) for m in METHODS}
for s in range(5):
    for t in range(5):
        for name, (a, b) in three_maps(Q[s], Q[t]).items():
            rel = q.relative_gap(Q[s], Q[t], a, b)
            mats[name][s, t] = rel
            rows.append(dict(dataset="wilds", method=name, split=f"hosp{s}->hosp{t}",
                             seed=0, metric="rel_w2sq", value=rel))

vmax = max(M.max() for M in mats.values())
fig2, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, (name, M) in zip(axes, mats.items()):
    im = ax.imshow(M, vmin=0, vmax=vmax, cmap="viridis")
    ax.set_title(name)
    ax.set_xlabel("target hospital")
    ax.set_ylabel("source hospital")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    for s in range(5):
        for t in range(5):
            ax.text(t, s, f"{M[s, t]:.2f}", ha="center", va="center", fontsize=8,
                    color="w" if M[s, t] < 0.6 * vmax else "k")
fig2.colorbar(im, ax=axes, shrink=0.8, label="relative gap")
fig2.savefig(ROOT / "figures" / "exp1_hospital_heatmap.png", dpi=150, bbox_inches="tight")
print("saved figures/exp1_hospital_heatmap.png")

pd.DataFrame(rows).to_csv(ROOT / "results" / "exp1_quantile_gap.csv", index=False)
print("saved results/exp1_quantile_gap.csv")

off = ~np.eye(5, dtype=bool)
print("\nmean relative gap over hospital pairs (off-diagonal):")
for name in METHODS:
    print(f"  {name:>18}: {mats[name][off].mean():.4f}   max {mats[name][off].max():.4f}")