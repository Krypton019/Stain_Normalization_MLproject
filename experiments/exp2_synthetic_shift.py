import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as cfg
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

from src import stainops as so
from src.data import load_wilds
from src.ot_norm import OTNormalizer

MODE = sys.argv[1] if len(sys.argv) > 1 else "random"   # "random" or "mismatch"
assert MODE in ("random", "mismatch"), "usage: python exp2_synthetic_shift.py [random|mismatch]"
N_POOL, N_REF, N_TEST = 300, 20, 100
S_GRID = [0.6, 0.8, 1.0, 1.2, 1.4]
RHOS = [0.1, 1.0, 10.0, 100.0]

cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def base_H(img):
    """Per-pixel stain amounts (96,96,2) of a real patch; zero on background. None if unusable."""
    m = so.tissue_mask(img)
    if m.sum() < 200:
        return None
    W = so.macenko_stain_matrix(img)
    if np.linalg.cond(W) > 20:
        return None
    H = so.concentrations(so.rgb_to_od(img), W)
    H[~m] = 0.0
    return H


def render(H, W):
    """Stain amounts + stain matrix -> uint8 RGB image."""
    od = H.reshape(-1, 2) @ W.T
    return so.od_to_rgb(od.reshape(*H.shape[:2], 3))


def mean_h(H):
    t = H.sum(-1) > 0
    return H[:, :, 0].sum() / max(int(t.sum()), 1)


# ---------- build the pool of base concentrations from real hospital-0 patches ----------
imgs, _, _ = load_wilds([0], N_POOL, seed=0)
pool = [h for h in (base_H(i) for i in imgs) if h is not None]
print(f"usable base patches: {len(pool)} of {N_POOL}")
assert len(pool) >= N_REF + N_TEST, "not enough usable patches; raise N_POOL"
mh = np.array([mean_h(h) for h in pool])

rng0 = np.random.default_rng(0)
if MODE == "random":
    order = rng0.permutation(len(pool))
    ref_idx, test_idx = order[:N_REF], order[N_REF:N_REF + N_TEST]
else:
    order = np.argsort(mh)
    ref_idx, test_idx = order[:N_REF], order[::-1][:N_TEST]
print(f"mode={MODE}: mean H of reference {mh[ref_idx].mean():.3f}, of test {mh[test_idx].mean():.3f}")

# ---------- reference domain and normalizers ----------
W_T = so.DEFAULT_HE / np.linalg.norm(so.DEFAULT_HE, axis=0, keepdims=True)
ref_imgs = [render(pool[i], W_T) for i in ref_idx]
norms = {"ot_balanced": OTNormalizer(mode="balanced").fit(ref_imgs)}
for rho in RHOS:
    norms[f"ot_unbal_rho{rho:g}"] = OTNormalizer(mode="unbalanced", rho=rho).fit(ref_imgs)

W_gt = norms["ot_balanced"].W_ref            # every stain method renders into this matrix
H_ref = norms["ot_balanced"].H_ref
p99_ref = np.percentile(H_ref, 99, axis=0)
ref_gt = render(pool[ref_idx[0]], W_gt)      # reference image for Reinhard


def macenko_norm(img):
    """Classical Macenko: estimate W and H, rescale each H channel by 99th-percentile ratio."""
    m = so.tissue_mask(img)
    if m.sum() < 200:
        return img.copy()
    W = so.macenko_stain_matrix(img)
    if np.linalg.cond(W) > 20:
        W = W_gt
    H = so.concentrations(so.rgb_to_od(img), W)[m]
    Hn = H * (p99_ref / (np.percentile(H, 99, axis=0) + 1e-8))
    out = img.copy()
    out[m] = so.od_to_rgb(Hn @ W_gt.T)
    return out


def shift(H, s, rng):
    """Synthetic scanner/lab shift: mix the two stain colours and scale the stain amounts by s."""
    M = np.array([[1.0, rng.uniform(-0.25, 0.25)], [rng.uniform(-0.25, 0.25), 1.0]])
    W2 = np.clip(W_gt @ M, 0.01, None)
    W2 = W2 / np.linalg.norm(W2, axis=0, keepdims=True)
    return render(s * H, W2)


# ---------- main loop ----------
methods = ["no_norm", "reinhard", "macenko", *norms]
rows = []
for si, s in enumerate(S_GRID):
    for k, ti in enumerate(test_idx):
        H = pool[ti]
        rng = np.random.default_rng(1000 * si + k)
        gt = render(H, W_gt)
        mask = so.tissue_mask(gt)
        if mask.sum() < 50:
            continue
        src = shift(H, s, rng)
        outs = {"no_norm": src, "reinhard": so.reinhard(src, ref_gt), "macenko": macenko_norm(src)}
        for name, nm in norms.items():
            outs[name] = nm.transform(src)
        for name, out in outs.items():
            err = out[mask].astype(float) - gt[mask].astype(float)
            rows.append(dict(dataset="wilds_synth_" + MODE, method=name, split=f"s={s}", seed=k,
                             metric="rmse", value=float(np.sqrt((err ** 2).mean()))))
            rows.append(dict(dataset="wilds_synth_" + MODE, method=name, split=f"s={s}", seed=k,
                             metric="ssim",
                             value=float(structural_similarity(gt, out, channel_axis=2, data_range=255))))
    print(f"finished s={s}")

df = pd.DataFrame(rows)
df.to_csv(cfg.RESULTS_DIR / f"exp2_synthetic_{MODE}.csv", index=False)

print("\nMean RMSE vs ground truth (lower is better):")
print(df[df.metric == "rmse"].groupby(["method", "split"]).value.mean().unstack("split").round(1).to_string())
print("\nMean SSIM vs ground truth (higher is better):")
print(df[df.metric == "ssim"].groupby(["method", "split"]).value.mean().unstack("split").round(3).to_string())
print("\nsolver counters (images, unstable W fallbacks, non-converged, slow path):")
for name, nm in norms.items():
    print(f"  {name:>18}: {nm.n_images}, {nm.n_fallback}, {nm.n_nonconverged}, {nm.n_slow_path}")

# ---------- figure ----------
keys = [f"s={s}" for s in S_GRID]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for ax, metric, label in zip(axes, ["rmse", "ssim"],
                             ["RMSE vs ground truth (0-255)", "SSIM vs ground truth"]):
    for name in methods:
        g = df[(df.method == name) & (df.metric == metric)].groupby("split").value
        ax.errorbar(S_GRID, g.mean().reindex(keys).values, yerr=g.sem().reindex(keys).values,
                    marker="o", ms=3, capsize=2, label=name)
    ax.set_xlabel("stain-strength factor s")
    ax.set_ylabel(label)
    ax.set_title(f"Synthetic shift ({MODE})")
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(cfg.FIGURES_DIR / f"exp2_synthetic_{MODE}.png", dpi=150)
print(f"saved {cfg.FIGURES_DIR / f'exp2_synthetic_{MODE}.png'}")
