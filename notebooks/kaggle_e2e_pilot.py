# %% [markdown]
# # Kaggle — End-to-End Pilot Run (Fast Verification)
#
# **GPU notebook** (Enable GPU accelerator in notebook settings: Settings → Accelerator → GPU T4 or P100).
#
# This notebook runs the **complete pipeline end-to-end** in ~5–10 minutes:
# 1. Clones the repository and installs dependencies.
# 2. Prepares a pilot dataset (uses attached `camelyon17-wilds-subsets` if available, or generates realistic H&E synthetic patches with hospital stain variations).
# 3. Normalises patches across all 5 methods (`none`, `reinhard`, `macenko`, `ot_bal`, `ot_unbal`).
# 4. Merges shards into full split arrays.
# 5. Trains ResNet-18 on GPU with mixed precision (AMP) for all methods.
# 6. Aggregates results into a summary table and writes `aggregate.csv`.

# %% [markdown]
# ## 0 · Clone repo and install dependencies

# %%
import subprocess, sys, os
from pathlib import Path

REPO_URL = "https://github.com/Krypton019/Stain_Normalization_MLproject.git"
REPO_DIR = "/kaggle/working/repo"
BRANCH   = "master"

def run(cmd, **kw):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, text=True,
                       capture_output=True, **kw)
    if r.stdout: print(r.stdout.rstrip())
    if r.stderr: print(r.stderr.rstrip(), file=sys.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed (exit {r.returncode}): {cmd}")
    return r

if not Path(REPO_DIR).exists():
    run(f"git clone --depth 1 --branch {BRANCH} {REPO_URL} {REPO_DIR}")
else:
    print(f"Repo already present at {REPO_DIR}, pulling …")
    run(f"git -C {REPO_DIR} pull --ff-only")

run(f"pip install -e {REPO_DIR} --quiet")
run(f"pip install -r {REPO_DIR}/requirements.txt --quiet")
print("Installation complete.")

# %% [markdown]
# ## 1 · Environment setup

# %%
os.environ["SRC_ROOT"]    = REPO_DIR
os.environ["CACHE_DIR"]   = "/kaggle/working/cache"
os.environ["RESULTS_DIR"] = "/kaggle/working/results"
os.environ["FIGURES_DIR"] = "/kaggle/working/figures"
os.environ["SHARD_SIZE"]  = "500"

DATA_CACHE = "/kaggle/working/cache"
CKPT_CACHE = "/kaggle/working/cache"

for d in (os.environ["CACHE_DIR"],
          os.environ["RESULTS_DIR"],
          os.environ["FIGURES_DIR"]):
    Path(d).mkdir(parents=True, exist_ok=True)

import torch
print("CUDA available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU            :", torch.cuda.get_device_name(0))

# %% [markdown]
# ## 2 · Data preparation (attached dataset or realistic H&E synthetic subset)

# %%
import numpy as np
sys.path.insert(0, REPO_DIR)
import config as cfg

candidate_subsets = [
    Path("/kaggle/input/camelyon17-wilds-subsets"),
    Path("/kaggle/input/camelyon17-wilds-subsets/subsets"),
    Path("/kaggle/input/camelyon17-wilds-subset"),
    Path("/kaggle/working/data/subsets"),
]

found_subset = None
for c in candidate_subsets:
    if (c / "X_train.npy").exists():
        found_subset = c
        break

if found_subset is not None:
    print(f"Found existing subset at {found_subset}")
    SUBSET_DIR = str(found_subset)
else:
    print("No attached subset found. Generating realistic pilot H&E patches with hospital stain variation …")
    from src import stainops as so
    SUBSET_DIR = "/kaggle/working/data/subsets"
    Path(SUBSET_DIR).mkdir(parents=True, exist_ok=True)

    # 5 hospital stain variations
    HOSP_W = {
        0: np.array([[0.65, 0.07], [0.70, 0.99], [0.29, 0.11]]),
        1: np.array([[0.60, 0.10], [0.72, 0.95], [0.35, 0.20]]),
        2: np.array([[0.55, 0.20], [0.75, 0.80], [0.37, 0.56]]),
        3: np.array([[0.68, 0.05], [0.68, 0.98], [0.25, 0.12]]),
        4: np.array([[0.58, 0.15], [0.73, 0.90], [0.34, 0.35]]),
    }

    def generate_pilot_patches(hospitals, n_per_hosp, seed=0):
        rng = np.random.default_rng(seed)
        all_X, all_y, all_h = [], [], []
        for h in hospitals:
            W = HOSP_W[h]
            W = W / np.linalg.norm(W, axis=0)
            for i in range(n_per_hosp):
                y = i % 2
                scale = 1.3 if y == 1 else 0.8
                H = scale * rng.gamma(2.0, 0.3, (96 * 96, 2))
                # Add background margin
                patch = so.od_to_rgb((H @ W.T).reshape(96, 96, 3))
                all_X.append(patch)
                all_y.append(y)
                all_h.append(h)
        return (np.array(all_X, dtype=np.uint8),
                np.array(all_y, dtype=np.int32),
                np.array(all_h, dtype=np.int32))

    # Generate 150 train (hosp 0,3,4), 50 val (hosp 1), 50 test (hosp 2)
    X_tr, y_tr, h_tr = generate_pilot_patches(cfg.TRAIN_HOSPITALS, 50, seed=0)
    X_va, y_va, h_va = generate_pilot_patches(cfg.VAL_HOSPITALS,   50, seed=1)
    X_te, y_te, h_te = generate_pilot_patches(cfg.TEST_HOSPITALS,  50, seed=2)

    for split, (X, y, h) in [("train", (X_tr, y_tr, h_tr)),
                              ("val",   (X_va, y_va, h_va)),
                              ("test",  (X_te, y_te, h_te))]:
        np.save(f"{SUBSET_DIR}/X_{split}.npy", X)
        np.save(f"{SUBSET_DIR}/y_{split}.npy", y)
        np.save(f"{SUBSET_DIR}/hospital_{split}.npy", h)
        np.save(f"{SUBSET_DIR}/idx_{split}.npy", np.arange(len(X), dtype=np.int64))
        print(f"  {split:<6}: {len(X)} patches  shape {X.shape}")

os.environ["SUBSET_DIR"] = SUBSET_DIR

# %% [markdown]
# ## 3 · Normalise patches across all 5 methods

# %%
METHODS = ["none", "reinhard", "macenko", "ot_bal", "ot_unbal"]
SPLITS  = ["train", "val", "test"]
WORKERS = min(4, os.cpu_count() or 2)

for method in METHODS:
    print(f"\n{'='*60}\n  Normalising method: {method}\n{'='*60}")
    run(
        f"python {REPO_DIR}/scripts/normalize_cache.py"
        f" --method  {method}"
        f" --splits  {' '.join(SPLITS)}"
        f" --workers {WORKERS}"
        f" --shard_size 500"
        f" --subset_dir {SUBSET_DIR}"
        f" --cache_dir  {DATA_CACHE}"
        f" --seed 0"
    )

# %% [markdown]
# ## 4 · Merge shards into full split arrays

# %%
run(
    f"python {REPO_DIR}/scripts/merge_shards.py"
    f" --method  {' '.join(METHODS)}"
    f" --splits  {' '.join(SPLITS)}"
    f" --cache_dir {DATA_CACHE}"
)
print("Merge complete.")

# %% [markdown]
# ## 5 · Train ResNet-18 classifier (GPU AMP)

# %%
SEEDS      = [0, 1]
EPOCHS     = 3
BATCH_SIZE = 64
BACKBONE   = "resnet18"

for method in METHODS:
    for seed in SEEDS:
        print(f"\n--- Training method={method} seed={seed} ---")
        run(
            f"python {REPO_DIR}/scripts/train.py"
            f" --method      {method}"
            f" --seed        {seed}"
            f" --epochs      {EPOCHS}"
            f" --backbone    {BACKBONE}"
            f" --batch_size  {BATCH_SIZE}"
            f" --data_dir    {DATA_CACHE}"
            f" --ckpt_dir    {CKPT_CACHE}"
            f" --results_dir {os.environ['RESULTS_DIR']}"
        )

# %% [markdown]
# ## 6 · Aggregate and display results

# %%
run(
    f"python {REPO_DIR}/scripts/aggregate.py"
    f" --results_dir {os.environ['RESULTS_DIR']}"
    f" --csv         {os.environ['RESULTS_DIR']}/aggregate.csv"
)

import pandas as pd
summary_path = Path(os.environ["RESULTS_DIR"]) / "aggregate.csv"
if summary_path.exists():
    df = pd.read_csv(summary_path)
    print("\n" + "="*60 + "\nPILOT RESULTS SUMMARY\n" + "="*60)
    print(df.to_string(index=False))
