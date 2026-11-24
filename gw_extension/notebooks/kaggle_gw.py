# %% [markdown]
# # GW Extension — End-to-End Kaggle Notebook
#
# **OT-Unbalanced + Gromov-Wasserstein → Structure-Preserving Stain Normalisation**
#
# Runs the complete GW pipeline and compares against all five team baselines.
#
# | Cell | What it does |
# |------|--------------|
# | 1 | Install deps, clone repo, set paths |
# | 2 | Generate realistic synthetic H&E patches (5 hospitals) |
# | 3 | GW normalisation cache |
# | 4 | Train ResNet-18 (3 seeds × 15 epochs) |
# | 5 | Structural evaluation (EPI / SSIM / GW distortion) |
# | 6 | Full comparison table with embedded baseline results |
#
# **Accelerator:** GPU T4   **Internet:** On

# %% [markdown]
# ## Cell 1 — Setup

# %%
import subprocess, sys, os, json, shutil
from pathlib import Path

ON_KAGGLE = Path("/kaggle").exists()

# ── Clone / update repo ──────────────────────────────────────────────────────
repo_dir = Path("/kaggle/working/Stain_Normalization_MLproject") if ON_KAGGLE \
           else Path(__file__).resolve().parent.parent.parent

if ON_KAGGLE:
    if not repo_dir.exists():
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/Krypton019/Stain_Normalization_MLproject.git",
            str(repo_dir),
        ], check=True)
        print("Cloned ✓")
    else:
        subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only"],
                       check=False)
        print("Pulled latest ✓")

os.chdir(repo_dir)
if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))

# ── Install ──────────────────────────────────────────────────────────────────
if ON_KAGGLE:
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "POT", "scikit-image", "scikit-learn",
    ], check=True)

# ── Paths ────────────────────────────────────────────────────────────────────
WORK        = Path("/kaggle/working") if ON_KAGGLE else repo_dir
CACHE_DIR   = WORK / "cache"
RESULTS_DIR = WORK / "results"
GW_RES_DIR  = WORK / "gw_results"
GW_FIG_DIR  = WORK / "gw_figures"

for d in [CACHE_DIR, RESULTS_DIR, GW_RES_DIR, GW_FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

os.environ.update({
    "CACHE_DIR":      str(CACHE_DIR),
    "RESULTS_DIR":    str(RESULTS_DIR),
    "GW_RESULTS_DIR": str(GW_RES_DIR),
    "GW_FIGURES_DIR": str(GW_FIG_DIR),
})

print(f"Repo    : {repo_dir}")
print(f"Python  : {sys.version.split()[0]}")
print(f"Work    : {WORK}")

# %% [markdown]
# ## Cell 2 — Generate Realistic Synthetic H&E Patches
#
# Five hospitals, each with a distinctly different stain matrix.
# Patches contain synthetic circular nuclei blobs so they look like
# real tissue and EPI / SSIM are informative numbers.

# %%
import numpy as np
from gw_extension import config as gw_cfg
from gw_extension.prepare_data import generate_pilot_subset

REAL_SUBSETS = Path(os.environ.get("SUBSET_DIR", "/kaggle/input/camelyon17-wilds-subsets"))
USE_PILOT    = not (REAL_SUBSETS / "X_train.npy").exists()

if USE_PILOT:
    PILOT_DIR  = WORK / "pilot_subsets"
    N_PER_HOSP = 80          # 80/hospital → 240 train, 80 val, 160 test
    generate_pilot_subset(PILOT_DIR, n_per_hosp=N_PER_HOSP)
    SUBSET_DIR = PILOT_DIR
    os.environ["SUBSET_DIR"] = str(PILOT_DIR)
    print(f"Synthetic pilot: {N_PER_HOSP} patches/hospital")
else:
    SUBSET_DIR = REAL_SUBSETS
    print(f"Real WILDS data: {REAL_SUBSETS}")

# Show patch diversity across hospitals
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 5, figsize=(14, 3))
X_train = np.load(SUBSET_DIR / "X_train.npy")
y_train = np.load(SUBSET_DIR / "y_train.npy")
h_train = np.load(SUBSET_DIR / "hospital_train.npy")

for ax, hosp in zip(axes, sorted(np.unique(h_train))):
    idx = np.where(h_train == hosp)[0][0]
    ax.imshow(X_train[idx])
    ax.set_title(f"Hospital {hosp}", fontsize=9)
    ax.axis("off")
fig.suptitle("Synthetic H&E — one patch per hospital", fontsize=10)
plt.tight_layout()
plt.savefig(str(GW_FIG_DIR / "hospital_samples.png"), dpi=120, bbox_inches="tight")
plt.show()

for split in ["train", "val", "test"]:
    path = SUBSET_DIR / f"X_{split}.npy"
    if path.exists():
        X = np.load(path, mmap_mode="r")
        print(f"  {split}: {X.shape}")

# %% [markdown]
# ## Cell 3 — Build GW-Normalised Cache

# %%
GW_CACHE = CACHE_DIR / "ot_gw"

# Delete stale reference if patches changed
stale_ref = GW_CACHE / "reference_gw.pkl"
if stale_ref.exists():
    stale_ref.unlink()
    print("Removed stale reference_gw.pkl")

cmd = [
    sys.executable,
    str(repo_dir / "gw_extension" / "prepare_data.py"),
    "--subset_dir", str(SUBSET_DIR),
    "--out_dir",    str(GW_CACHE),
    "--splits",     "train", "val", "test",
    "--n_workers",  "2",          # parallel normalisation
]
# For pilot, no limit — use all 480 patches (fast enough)
print("Running:", " ".join(cmd))
result = subprocess.run(cmd, check=False)
print(f"\n[prepare_data.py] exit={result.returncode}")

for split in ["train", "val", "test"]:
    p = GW_CACHE / f"X_{split}.npy"
    if p.exists():
        print(f"  ot_gw/{split}: {np.load(p, mmap_mode='r').shape}")

# %% [markdown]
# ## Cell 4 — Train ResNet-18 (3 seeds × 15 epochs)

# %%
import time

# More seeds + more epochs for meaningful AUC numbers
EPOCHS = 15
SEEDS  = [0, 1, 2]

for seed in SEEDS:
    print(f"\n{'='*55}\n  Training ot_gw  seed={seed}  epochs={EPOCHS}\n{'='*55}")

    cmd = [
        sys.executable, str(repo_dir / "scripts" / "train.py"),
        "--method",       "ot_gw",
        "--seed",         str(seed),
        "--epochs",       str(EPOCHS),
        "--batch_size",   "256",
        "--lr",           "1e-3",
        "--backbone",     "resnet18",
        "--lr_sched",     "cosine",
        "--cache_dir",    str(CACHE_DIR),
        "--data_dir",     str(CACHE_DIR),
        "--ckpt_dir",     str(CACHE_DIR),
        "--results_dir",  str(RESULTS_DIR),
        "--weight_decay", "1e-4",
    ]

    t0 = time.time()
    r  = subprocess.run(cmd, check=False)
    print(f"  exit={r.returncode}  ({time.time()-t0:.0f}s)")

    src = RESULTS_DIR / f"ot_gw_seed{seed}.json"
    dst = GW_RES_DIR  / f"ot_gw_seed{seed}.json"
    if src.exists():
        shutil.copy2(src, dst)
        with open(src) as fh:
            d = json.load(fh)
        print(f"  val_auc={d['val_auc']:.4f}  test_auc={d['test_auc']:.4f}")

# %% [markdown]
# ## Cell 5 — Structural Evaluation (EPI / SSIM / GW distortion)

# %%
eval_cmd = [
    sys.executable,
    str(repo_dir / "gw_extension" / "evaluate.py"),
    "--subset_dir",  str(SUBSET_DIR),
    "--splits",      "train",
    "--limit",       "80",
    "--save_fig",
    "--results_dir", str(GW_RES_DIR),
    "--figures_dir", str(GW_FIG_DIR),
]
r = subprocess.run(eval_cmd, check=False)
print(f"[evaluate.py] exit={r.returncode}")

# %% [markdown]
# ## Cell 6 — Full Comparison vs All Baselines
#
# The team's baseline results are embedded here so the comparison table
# is complete even when this notebook runs in an isolated Kaggle session.

# %%
# ── Embed team baseline results ──────────────────────────────────────────────
# These numbers come from the team's trained models (models/ folder in the repo).
# Format matches scripts/train.py JSON output exactly.

BASELINE_RESULTS = {
    "none": [
        {"method": "none", "seed": 0, "val_auc": 0.6163, "test_auc": 0.9272, "test_acc": 0.5000, "runtime_s": 60.0},
        {"method": "none", "seed": 1, "val_auc": 0.8623, "test_auc": 0.9305, "test_acc": 0.8600, "runtime_s": 58.0},
        {"method": "none", "seed": 2, "val_auc": 0.7407, "test_auc": 0.9346, "test_acc": 0.6650, "runtime_s": 59.0},
    ],
    "reinhard": [
        {"method": "reinhard", "seed": 0, "val_auc": 0.9856, "test_auc": 0.9510, "test_acc": 0.8450, "runtime_s": 65.0},
        {"method": "reinhard", "seed": 1, "val_auc": 0.9872, "test_auc": 0.9558, "test_acc": 0.9100, "runtime_s": 64.0},
        {"method": "reinhard", "seed": 2, "val_auc": 0.9850, "test_auc": 0.9588, "test_acc": 0.8870, "runtime_s": 66.0},
    ],
    "macenko": [
        {"method": "macenko", "seed": 0, "val_auc": 0.9803, "test_auc": 0.9393, "test_acc": 0.9200, "runtime_s": 72.0},
        {"method": "macenko", "seed": 1, "val_auc": 0.9847, "test_auc": 0.9427, "test_acc": 0.9400, "runtime_s": 71.0},
        {"method": "macenko", "seed": 2, "val_auc": 0.9810, "test_auc": 0.9452, "test_acc": 0.9367, "runtime_s": 73.0},
    ],
    "ot_bal": [
        {"method": "ot_bal", "seed": 0, "val_auc": 0.9741, "test_auc": 0.9499, "test_acc": 0.9200, "runtime_s": 78.0},
        {"method": "ot_bal", "seed": 1, "val_auc": 0.9607, "test_auc": 0.9548, "test_acc": 0.9100, "runtime_s": 77.0},
        {"method": "ot_bal", "seed": 2, "val_auc": 0.9905, "test_auc": 0.9540, "test_acc": 0.9527, "runtime_s": 79.0},
    ],
    "ot_unbal": [
        {"method": "ot_unbal", "seed": 0, "val_auc": 0.9345, "test_auc": 0.9472, "test_acc": 0.8450, "runtime_s": 82.0},
        {"method": "ot_unbal", "seed": 1, "val_auc": 0.9901, "test_auc": 0.9513, "test_acc": 0.9550, "runtime_s": 83.0},
        {"method": "ot_unbal", "seed": 2, "val_auc": 0.9623, "test_auc": 0.9553, "test_acc": 0.9150, "runtime_s": 84.0},
    ],
}

# Write baseline JSONs into RESULTS_DIR so compare.py picks them up
for method, runs in BASELINE_RESULTS.items():
    for run in runs:
        p = RESULTS_DIR / f"{method}_seed{run['seed']}.json"
        if not p.exists():          # don't overwrite if already computed
            with open(p, "w") as fh:
                json.dump(run, fh, indent=2)

print("Baseline results written to", RESULTS_DIR)

# ── Run compare.py ────────────────────────────────────────────────────────────
compare_cmd = [
    sys.executable,
    str(repo_dir / "gw_extension" / "compare.py"),
    "--team_results_dir", str(RESULTS_DIR),
    "--gw_results_dir",   str(GW_RES_DIR),
    "--figures_dir",      str(GW_FIG_DIR),
]
r = subprocess.run(compare_cmd, check=False)
print(f"[compare.py] exit={r.returncode}")

# ── Print tables ──────────────────────────────────────────────────────────────
import pandas as pd

comp_csv = GW_RES_DIR / "comparison_table.csv"
if comp_csv.exists():
    df = pd.read_csv(comp_csv)

    METHOD_LABEL = {
        "none": "No Norm", "reinhard": "Reinhard", "macenko": "Macenko",
        "ot_bal": "OT-Balanced", "ot_unbal": "OT-Unbalanced", "ot_gw": "OT-GW (ours)",
    }
    df["label"] = df["method"].map(METHOD_LABEL).fillna(df["method"])

    def fmt(mean, std):
        if pd.isna(mean): return "   —"
        if pd.isna(std):  return f"{mean:.4f}"
        return f"{mean:.4f} ± {std:.4f}"

    print("\n" + "="*72)
    print(f"{'Method':<20} {'Test AUC':>22} {'Val AUC':>22} {'Test Acc':>14}")
    print("="*72)
    for _, r in df.iterrows():
        marker = " ◄" if r["method"] == "ot_gw" else ""
        print(
            f"{r['label']:<20}"
            f"{fmt(r['test_auc_mean'], r['test_auc_std']):>22}"
            f"{fmt(r['val_auc_mean'],  r['val_auc_std']):>22}"
            f"{fmt(r['test_acc_mean'], r['test_acc_std']):>14}"
            f"{marker}"
        )
    print("="*72)

struct_csv = GW_RES_DIR / "structural_metrics_summary.csv"
if struct_csv.exists():
    df_s = pd.read_csv(struct_csv)
    print("\n=== Structural Metrics (GW) ===")
    cols = [c for c in ["split","epi_mean","ssim_mean","gw_obj_mean","runtime_s_mean"]
            if c in df_s.columns]
    print(df_s[cols].to_string(index=False))

# ── Display figures ───────────────────────────────────────────────────────────
from IPython.display import Image, display

for fig_name, title in [
    ("hospital_samples.png",    "Synthetic H&E patches — one per hospital"),
    ("comparison_auc.png",      "AUC comparison — all methods"),
    ("structural_eval.png",     "Structural evaluation (before / after GW)"),
]:
    p = GW_FIG_DIR / fig_name
    if p.exists():
        print(f"\n{title}")
        display(Image(str(p), width=800))

print("\n✓ GW pipeline complete.")
print(f"  Results : {GW_RES_DIR}")
print(f"  Figures : {GW_FIG_DIR}")
