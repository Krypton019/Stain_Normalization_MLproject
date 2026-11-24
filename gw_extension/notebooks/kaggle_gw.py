# %% [markdown]
# # GW Extension — End-to-End Kaggle Notebook
#
# **OT-Unbalanced + Gromov-Wasserstein → Structure-Preserving Stain Normalisation**
#
# This notebook is the single entry-point for the GW extension experiment.
# It runs the full pipeline in five cells:
#
# | Cell | What it does | Time (GPU T4) |
# |------|--------------|---------------|
# | 1    | Install dependencies & clone repo | ~1 min |
# | 2    | Generate pilot subset (or use real WILDS data) | ~1 min |
# | 3    | Build GW-normalised patch cache | ~5–30 min depending on dataset size |
# | 4    | Train ResNet-18 on GW-normalised patches | ~20–60 min on GPU |
# | 5    | Evaluate & compare vs baselines | ~5 min |
#
# **Accelerator required:** GPU T4 or P100
# **Internet:** On (for `pip install` and optional git clone)

# %% [markdown]
# ## Cell 1 — Setup

# %%
import subprocess, sys, os, json
from pathlib import Path

ON_KAGGLE = Path("/kaggle").exists()

if ON_KAGGLE:
    # Install dependencies
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "POT", "scikit-image", "scikit-learn", "wilds",
    ], check=True)

    # Clone repo if not already present
    repo_dir = Path("/kaggle/working/Stain_Normalization_MLproject")
    if not repo_dir.exists():
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/Krypton019/Stain_Normalization_MLproject.git",
            str(repo_dir),
        ], check=True)

    os.chdir(repo_dir)
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    # Set Kaggle environment variables
    os.environ["SUBSET_DIR"]    = "/kaggle/input/camelyon17-wilds-subsets"
    os.environ["CACHE_DIR"]     = "/kaggle/working/cache"
    os.environ["RESULTS_DIR"]   = "/kaggle/working/results"
    os.environ["FIGURES_DIR"]   = "/kaggle/working/figures"
    os.environ["GW_RESULTS_DIR"] = "/kaggle/working/gw_results"
    os.environ["GW_FIGURES_DIR"] = "/kaggle/working/gw_figures"
else:
    repo_dir = Path(__file__).resolve().parent.parent.parent
    os.chdir(repo_dir)
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

print(f"Repo root : {repo_dir}")
print(f"Python    : {sys.version}")
print(f"On Kaggle : {ON_KAGGLE}")

# %% [markdown]
# ## Cell 2 — Prepare Pilot Subset
#
# If the CAMELYON17-WILDS dataset is attached, we sample from it.
# Otherwise we generate reproducible synthetic H&E patches so the full
# pipeline can be validated without any external data dependency.

# %%
import numpy as np
from gw_extension import config as gw_cfg
from gw_extension.prepare_data import generate_pilot_subset

SUBSET_DIR = gw_cfg.SUBSET_DIR
USE_PILOT  = not (SUBSET_DIR / "X_train.npy").exists()

if USE_PILOT:
    print("Real WILDS subsets not found — generating synthetic pilot patches.")
    PILOT_DIR = Path("/kaggle/working/pilot_subsets") if ON_KAGGLE \
                else repo_dir / "data" / "pilot_subsets"
    generate_pilot_subset(PILOT_DIR, n_per_hosp=50)
    SUBSET_DIR = PILOT_DIR
    # Override env var so downstream scripts pick it up
    os.environ["SUBSET_DIR"] = str(PILOT_DIR)
else:
    print(f"Using existing subset at {SUBSET_DIR}")

for split in ["train", "val", "test"]:
    path = SUBSET_DIR / f"X_{split}.npy"
    if path.exists():
        X = np.load(path, mmap_mode="r")
        print(f"  {split}: {X.shape}  dtype={X.dtype}")

# %% [markdown]
# ## Cell 3 — GW Normalisation (Build Cache)
#
# Runs `prepare_data.py` which:
# 1. Fits or loads the GW reference (from `reference_gw.pkl` or teammates' `reference.pkl`)
# 2. Applies `GWNormalizer.transform()` to every patch
# 3. Saves `X_{split}.npy`, `y_{split}.npy`, `hospital_{split}.npy` under `CACHE_DIR/ot_gw/`

# %%
# Pilot flag: use --limit to process a small batch quickly
LIMIT = 200 if USE_PILOT else None   # set to None for full run on real data

cmd = [
    sys.executable,
    str(repo_dir / "gw_extension" / "prepare_data.py"),
    "--subset_dir", str(SUBSET_DIR),
    "--out_dir",    str(gw_cfg.GW_CACHE_DIR),
    "--splits",     "train", "val", "test",
]
if LIMIT:
    cmd += ["--limit", str(LIMIT)]

print("Command:", " ".join(cmd))
result = subprocess.run(cmd, check=False)
print(f"\n[prepare_data.py] exit code: {result.returncode}")

# Verify output
for split in ["train", "val", "test"]:
    path = gw_cfg.GW_CACHE_DIR / f"X_{split}.npy"
    if path.exists():
        X = np.load(path, mmap_mode="r")
        print(f"  ot_gw/{split}: {X.shape}")

# %% [markdown]
# ## Cell 4 — Train ResNet-18 on GW-Normalised Patches
#
# Uses the team's `scripts/train.py` with `--method ot_gw`.
# Identical protocol: ResNet-18, no pretrained weights, AMP, cosine LR, 3 seeds.

# %%
import time

EPOCHS    = 2 if USE_PILOT else 10    # 2 for pilot, 10 for real run
SUBSAMPLE = 200 if USE_PILOT else None
SEEDS     = [0] if USE_PILOT else [0, 1, 2]

gw_results_dir = gw_cfg.GW_RESULTS_DIR
gw_results_dir.mkdir(parents=True, exist_ok=True)

main_results_dir = gw_cfg.BASE_RESULTS
main_results_dir.mkdir(parents=True, exist_ok=True)

for seed in SEEDS:
    print(f"\n{'='*55}")
    print(f"  Training seed {seed} / {SEEDS}")
    print("="*55)

    cmd = [
        sys.executable,
        str(repo_dir / "scripts" / "train.py"),
        "--method",      "ot_gw",
        "--seed",        str(seed),
        "--epochs",      str(EPOCHS),
        "--batch_size",  "256",
        "--lr",          "1e-3",
        "--backbone",    "resnet18",
        "--lr_sched",    "cosine",
        "--cache_dir",   str(gw_cfg.CACHE_DIR),
        "--data_dir",    str(gw_cfg.CACHE_DIR),
        "--ckpt_dir",    str(gw_cfg.CACHE_DIR),
        "--results_dir", str(main_results_dir),
        "--weight_decay", "1e-4",
    ]
    if SUBSAMPLE:
        cmd += ["--subsample", str(SUBSAMPLE)]

    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0

    print(f"  [train.py] exit code: {result.returncode}  ({elapsed:.0f}s)")

    # Copy result to gw_extension/results/
    import shutil
    src = main_results_dir / f"ot_gw_seed{seed}.json"
    dst = gw_results_dir   / f"ot_gw_seed{seed}.json"
    if src.exists():
        shutil.copy2(src, dst)
        with open(src) as fh:
            d = json.load(fh)
        print(f"  val_auc={d.get('val_auc', '?'):.4f}  "
              f"test_auc={d.get('test_auc', '?'):.4f}")
    else:
        print(f"  [warning] result file not found: {src}")

# %% [markdown]
# ## Cell 5 — Evaluate Structural Metrics & Compare vs Baselines

# %%
# --- 5a: Structural evaluation (EPI, SSIM, GW distortion, runtime) ---
eval_cmd = [
    sys.executable,
    str(repo_dir / "gw_extension" / "evaluate.py"),
    "--subset_dir",  str(SUBSET_DIR),
    "--splits",      "train",
    "--limit",       "50",
    "--save_fig",
    "--results_dir", str(gw_results_dir),
    "--figures_dir", str(gw_cfg.GW_FIGURES_DIR),
]
if USE_PILOT:
    eval_cmd.append("--pilot")

result = subprocess.run(eval_cmd, check=False)
print(f"[evaluate.py] exit code: {result.returncode}")

# --- 5b: AUC comparison table vs baselines ---
compare_cmd = [
    sys.executable,
    str(repo_dir / "gw_extension" / "compare.py"),
    "--team_results_dir", str(main_results_dir),
    "--gw_results_dir",   str(gw_results_dir),
    "--figures_dir",      str(gw_cfg.GW_FIGURES_DIR),
]
result = subprocess.run(compare_cmd, check=False)
print(f"[compare.py] exit code: {result.returncode}")

# --- Display results ---
import pandas as pd

comp_csv = gw_results_dir / "comparison_table.csv"
if comp_csv.exists():
    df = pd.read_csv(comp_csv)
    print("\n=== AUC Comparison ===")
    cols = ["method"] + [c for c in df.columns if "test_auc" in c or "val_auc" in c]
    print(df[cols].to_string(index=False))

struct_csv = gw_results_dir / "structural_metrics_summary.csv"
if struct_csv.exists():
    df_s = pd.read_csv(struct_csv)
    print("\n=== Structural Metrics (GW) ===")
    print(df_s[["split", "epi_mean", "ssim_mean", "gw_obj_mean", "runtime_s_mean"]].to_string(index=False))

# --- Show figure inline ---
from pathlib import Path as P
import os

fig_path = gw_cfg.GW_FIGURES_DIR / "comparison_auc.png"
if fig_path.exists():
    try:
        from IPython.display import Image, display
        display(Image(str(fig_path)))
    except ImportError:
        print(f"Figure saved at: {fig_path}")

eval_fig = gw_cfg.GW_FIGURES_DIR / "structural_eval.png"
if eval_fig.exists():
    try:
        from IPython.display import Image, display
        display(Image(str(eval_fig)))
    except ImportError:
        print(f"Structural eval figure: {eval_fig}")

print("\n✓ GW extension pipeline complete.")
print(f"  Results : {gw_results_dir}")
print(f"  Figures : {gw_cfg.GW_FIGURES_DIR}")
