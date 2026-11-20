# %% [markdown]
# # Kaggle — Train Classifier
#
# **GPU notebook** (enable GPU accelerator in notebook settings).  
# Clones the repo, installs it, then calls `scripts/train.py` for every
# method × seed combination.  Each run resumes from its checkpoint if one
# exists, so the notebook is safe to re-run after a session timeout.
#
# ### Inputs (Kaggle datasets to attach)
# | Dataset slug | Mount path | Contains |
# |---|---|---|
# | `your-username/camelyon17-wilds-normalised` | `/kaggle/input/camelyon17-wilds-normalised` | `{method}/X_{split}.npy` etc., `reference.pkl` |
#
# ### Outputs  (`/kaggle/working/`)
# ```
# results/
#   {method}_seed{seed}.json   — per-run metrics (best_epoch, val_auc, test_auc, …)
#   aggregate.csv              — mean ± std table across seeds
# cache/
#   {method}/ckpt_seed{seed}.pt   — checkpoints (resumed automatically)
# ```

# %% [markdown]
# ## 0 · Clone repo and install

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
print("Install complete.")

# %% [markdown]
# ## 1 · Environment variables

# %%
# DATA_CACHE points at the normalised arrays dataset (read-only /kaggle/input/…).
# CKPT_CACHE and RESULTS_DIR go to /kaggle/working/ (writable, persisted).

candidate_normalised_dirs = [
    Path(os.environ.get("CACHE_DIR", "/kaggle/input/camelyon17-wilds-normalised")),
    Path("/kaggle/input/camelyon17-wilds-normalised"),
    Path("/kaggle/input/camelyon17-wilds-normalised/cache"),
    Path("/kaggle/input/camelyon17-normalised"),
    Path("/kaggle/working/cache"),
]
found_cache = None
for c in candidate_normalised_dirs:
    if (c / "none" / "X_train.npy").exists():
        found_cache = c
        break

DATA_CACHE = str(found_cache) if found_cache is not None else "/kaggle/input/camelyon17-wilds-normalised"
CKPT_CACHE = "/kaggle/working/cache"

os.environ["SRC_ROOT"]    = REPO_DIR
os.environ["CACHE_DIR"]   = DATA_CACHE
os.environ["RESULTS_DIR"] = "/kaggle/working/results"
os.environ["FIGURES_DIR"] = "/kaggle/working/figures"

print(f"  SRC_ROOT        = {os.environ['SRC_ROOT']}")
print(f"  DATA_CACHE      = {DATA_CACHE}")
print(f"  CKPT_CACHE      = {CKPT_CACHE}")
print(f"  RESULTS_DIR     = {os.environ['RESULTS_DIR']}")

for d in (os.environ["RESULTS_DIR"],
          os.environ["FIGURES_DIR"],
          CKPT_CACHE):
    Path(d).mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 2 · Verify GPU and normalised arrays

# %%
import torch
print("CUDA available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU            :", torch.cuda.get_device_name(0))

cache_dir = Path(DATA_CACHE)
METHODS   = ["none", "reinhard", "macenko", "ot_bal", "ot_unbal"]
SPLITS    = ["train", "val", "test"]

import numpy as np
print(f"\n{'method':<12} {'split':<8} {'shape'}")
for method in METHODS:
    for split in SPLITS:
        p = cache_dir / method / f"X_{split}.npy"
        if p.exists():
            arr = np.load(p, mmap_mode="r")
            print(f"  {method:<10} {split:<8} {arr.shape}")
        else:
            print(f"  {method:<10} {split:<8} MISSING  ← re-run kaggle_normalize.py")

# %% [markdown]
# ## 3 · Training configuration

# %%
SEEDS      = [0, 1, 2]        # three seeds → mean ± std
EPOCHS     = 15
BATCH_SIZE = 256               # fits on T4 at resnet18
BACKBONE   = "resnet18"
LR_SCHED   = "cosine"
# Set COLOR_AUG_METHODS to run the colour-augmentation baseline for those methods
COLOR_AUG_METHODS = []         # e.g. ["none"] to run aug baseline on raw patches

print(f"methods    : {METHODS}")
print(f"seeds      : {SEEDS}")
print(f"epochs     : {EPOCHS}")
print(f"batch_size : {BATCH_SIZE}")
print(f"backbone   : {BACKBONE}")

# %% [markdown]
# ## 4 · Train — all methods × all seeds
#
# Each call resumes from `CKPT_CACHE/{method}/ckpt_seed{seed}.pt` if it exists.  
# Re-running this cell after a timeout picks up exactly where it left off.

# %%
import time

total_runs = len(METHODS) * len(SEEDS)
done = 0
failed = []

for method in METHODS:
    for seed in SEEDS:
        color_flag = "--color_aug" if method in COLOR_AUG_METHODS else ""
        print(f"\n{'='*64}")
        print(f"  [{done+1}/{total_runs}]  method={method}  seed={seed}")
        print(f"{'='*64}")
        t0 = time.time()
        try:
            run(
                f"python {REPO_DIR}/scripts/train.py"
                f" --method      {method}"
                f" --seed        {seed}"
                f" --epochs      {EPOCHS}"
                f" --backbone    {BACKBONE}"
                f" --batch_size  {BATCH_SIZE}"
                f" --lr_sched    {LR_SCHED}"
                f" --data_dir    {DATA_CACHE}"   # read-only input from /kaggle/input/…
                f" --ckpt_dir    {CKPT_CACHE}"   # writable checkpoints in /kaggle/working/cache/
                f" --results_dir {os.environ['RESULTS_DIR']}"
                f" {color_flag}"
            )
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            failed.append((method, seed))
        finally:
            done += 1
            print(f"  wall time: {(time.time()-t0)/60:.1f} min")

if failed:
    print(f"\nFailed runs: {failed}")
else:
    print("\nAll runs completed successfully.")

# %% [markdown]
# ## 5 · Aggregate results

# %%
run(
    f"python {REPO_DIR}/scripts/aggregate.py"
    f" --results_dir {os.environ['RESULTS_DIR']}"
    f" --csv         {os.environ['RESULTS_DIR']}/aggregate.csv"
)

# %% [markdown]
# ## 6 · Summary table

# %%
import pandas as pd

agg_path = Path(os.environ["RESULTS_DIR"]) / "aggregate.csv"
if agg_path.exists():
    summary = pd.read_csv(agg_path)
    print(summary.to_string(index=False))
else:
    print("aggregate.csv not found — check for errors above.")

# %% [markdown]
# ## Done
#
# Download `/kaggle/working/results/` to get the JSON files and `aggregate.csv`
# for the paper.
