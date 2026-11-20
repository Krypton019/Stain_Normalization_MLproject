# %% [markdown]
# # Kaggle — Stain Normalisation Cache
#
# **CPU notebook.**  
# Clones the repo, installs it, then calls `scripts/normalize_cache.py` for
# every method in sequence.  Existing shards are skipped automatically, so the
# notebook is safe to re-run after a session timeout.
#
# ### Inputs (Kaggle datasets to attach)
# | Dataset slug | Mount path | Contains |
# |---|---|---|
# | `your-username/camelyon17-wilds-subset` | `/kaggle/input/camelyon17-wilds-subsets` | `X_train.npy`, `y_train.npy`, `hospital_train.npy`, `idx_train.npy` (and val/test) |
#
# ### Outputs  (`/kaggle/working/`)
# ```
# cache/
#   reference.pkl
#   none/     train_0000.npy … val_0000.npy … test_0000.npy …
#   reinhard/ …
#   macenko/  …
#   ot_bal/   …
#   ot_unbal/ …
#   failures.csv   (patches that fell back to original; may be empty)
# figures/
#   dry_run_<method>.png   (one quick visual check per method)
# ```
# Save the `cache/` output as a new Kaggle dataset and attach it to the
# training notebook.

# %% [markdown]
# ## 0 · Clone repo and install

# %%
import subprocess, sys

REPO_URL    = "https://github.com/Krypton019/Stain_Normalization_MLproject.git"
REPO_DIR    = "/kaggle/working/repo"
BRANCH      = "master"   # change if your branch differs

def run(cmd, **kw):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, text=True,
                       capture_output=True, **kw)
    if r.stdout: print(r.stdout.rstrip())
    if r.stderr: print(r.stderr.rstrip(), file=sys.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed (exit {r.returncode}): {cmd}")
    return r

import os
from pathlib import Path

if not Path(REPO_DIR).exists():
    run(f"git clone --depth 1 --branch {BRANCH} {REPO_URL} {REPO_DIR}")
else:
    print(f"Repo already present at {REPO_DIR}, pulling latest …")
    run(f"git -C {REPO_DIR} pull --ff-only")

run(f"pip install -e {REPO_DIR} --quiet")
run(f"pip install -r {REPO_DIR}/requirements.txt --quiet")
print("Install complete.")

# %% [markdown]
# ## 1 · Environment variables

# %%
# All paths are picked up by config.py via os.environ.
# Edit SUBSET_DIR if your dataset slug differs.

os.environ["SRC_ROOT"]    = REPO_DIR
os.environ["CACHE_DIR"]   = "/kaggle/working/cache"
os.environ["RESULTS_DIR"] = "/kaggle/working/results"
os.environ["FIGURES_DIR"] = "/kaggle/working/figures"

# Optional: control shard size to fit within Kaggle's RAM
os.environ["SHARD_SIZE"]  = "2000"

# Auto-detect subset directory
candidate_subset_dirs = [
    Path(os.environ.get("SUBSET_DIR", "/kaggle/input/camelyon17-wilds-subsets")),
    Path("/kaggle/input/camelyon17-wilds-subsets"),
    Path("/kaggle/input/camelyon17-wilds-subsets/subsets"),
    Path("/kaggle/input/camelyon17-wilds-subset"),
    Path("/kaggle/input/camelyon17-wilds-subset/subsets"),
    Path("/kaggle/working/data/subsets"),
]
found_dir = None
for c in candidate_subset_dirs:
    if (c / "X_train.npy").exists():
        found_dir = c
        break

if found_dir is not None:
    os.environ["SUBSET_DIR"] = str(found_dir)
else:
    os.environ["SUBSET_DIR"] = "/kaggle/input/camelyon17-wilds-subsets"

for k in ("SRC_ROOT", "SUBSET_DIR", "CACHE_DIR", "FIGURES_DIR"):
    print(f"  {k:15s} = {os.environ[k]}")

# create output dirs up front so later cells can write without errors
for d in ("/kaggle/working/cache",
          "/kaggle/working/results",
          "/kaggle/working/figures"):
    Path(d).mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 2 · Verify subset files exist

# %%
import sys
sys.path.insert(0, REPO_DIR)

import numpy as np
from pathlib import Path

subset_dir = Path(os.environ["SUBSET_DIR"])
for split in ("train", "val", "test"):
    p = subset_dir / f"X_{split}.npy"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found.\n"
            "Make sure you attached the subset dataset to this notebook.\n"
            "Run scripts/make_subset.py locally (or generate it on Kaggle) first."
        )
    arr = np.load(p, mmap_mode="r")
    print(f"  X_{split}.npy  {arr.shape}  {arr.dtype}")

# %% [markdown]
# ## 3 · Dry-run visual check (optional — 200 patches, saves PNG)

# %%
DRY_RUN_METHODS = ["none", "macenko", "ot_bal"]   # edit as needed

for method in DRY_RUN_METHODS:
    print(f"\n── dry run: {method} ──")
    run(
        f"python {REPO_DIR}/scripts/normalize_cache.py"
        f" --method {method}"
        f" --dry_run"
        f" --subset_dir {os.environ['SUBSET_DIR']}"
        f" --cache_dir  {os.environ['CACHE_DIR']}"
        f" --seed 0"
    )
print("\nDry-run figures written to", os.environ["FIGURES_DIR"])

# %% [markdown]
# ## 4 · Full normalisation — all methods, all splits
#
# Each method writes `CACHE_DIR/{method}/{split}_{shard:04d}.npy`.  
# Existing shards are **skipped**, so this cell is safe to restart.

# %%
METHODS  = ["none", "reinhard", "macenko", "ot_bal", "ot_unbal"]
SPLITS   = ["train", "val", "test"]
WORKERS  = min(4, os.cpu_count() or 2)

for method in METHODS:
    print(f"\n{'='*60}\n  method: {method}\n{'='*60}")
    run(
        f"python {REPO_DIR}/scripts/normalize_cache.py"
        f" --method  {method}"
        f" --splits  {' '.join(SPLITS)}"
        f" --workers {WORKERS}"
        f" --shard_size {os.environ['SHARD_SIZE']}"
        f" --subset_dir {os.environ['SUBSET_DIR']}"
        f" --cache_dir  {os.environ['CACHE_DIR']}"
        f" --seed 0"
    )

print("\nAll methods done.")

# %% [markdown]
# ## 5 · Merge shards into full split arrays

# %%
run(
    f"python {REPO_DIR}/scripts/merge_shards.py"
    f" --method  {' '.join(METHODS)}"
    f" --splits  {' '.join(SPLITS)}"
    f" --cache_dir {os.environ['CACHE_DIR']}"
)
print("Merge complete.")

# %% [markdown]
# ## 6 · Verify outputs

# %%
cache_dir = Path(os.environ["CACHE_DIR"])
print(f"{'method':<12} {'split':<8} {'shape':<20} size MB")
for method in METHODS:
    for split in SPLITS:
        p = cache_dir / method / f"X_{split}.npy"
        if p.exists():
            arr = np.load(p, mmap_mode="r")
            mb  = p.stat().st_size / 1e6
            print(f"  {method:<10} {split:<8} {str(arr.shape):<20} {mb:.0f}")
        else:
            print(f"  {method:<10} {split:<8} MISSING")

failures_path = cache_dir / "failures.csv"
if failures_path.exists():
    import pandas as pd
    fail_df = pd.read_csv(failures_path)
    print(f"\nFailures logged: {len(fail_df)} patches")
    print(fail_df.head())
else:
    print("\nNo failures.csv — all patches normalised successfully.")

# %% [markdown]
# ## Done
#
# **Next step:** save `/kaggle/working/cache/` as a new Kaggle dataset
# (e.g. `your-username/camelyon17-wilds-normalised`) and attach it to
# `kaggle_train.py`.
