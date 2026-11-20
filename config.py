"""Central configuration — all paths come from environment variables.

Local defaults assume the repo is at the current working directory.
Kaggle defaults assume the standard /kaggle layout.

Environment variables
─────────────────────
DATA_ROOT      WILDS root dir (contains camelyon17_v1.0/).
               Local default : <repo>/data
               Kaggle default: /kaggle/input/camelyon17-wilds

SUBSET_DIR     Directory for any pre-extracted subset / preprocessed files.
               Local default : <repo>/data/subsets
               Kaggle default: /kaggle/input/camelyon17-wilds-subsets

CACHE_DIR      Where Stage-A normalised .npz shards are written/read.
               Local default : <repo>/results/cache
               Kaggle default: /kaggle/working/cache

RESULTS_DIR    Where CSV results land.
               Local default : <repo>/results
               Kaggle default: /kaggle/working/results

FIGURES_DIR    Where figures land.
               Local default : <repo>/figures
               Kaggle default: /kaggle/working/figures

SRC_ROOT       Repo root (so `import src` and `import config` work from
               notebooks that live outside the repo).
               Local default : directory containing this file
               Kaggle default: /kaggle/working

Usage
─────
    import config as cfg
    imgs, labels, hosps = load_wilds([0], 100, root=str(cfg.DATA_ROOT))
    df.to_csv(cfg.RESULTS_DIR / "my_exp.csv")
"""

import os
from pathlib import Path

# ── repo root ───────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # always the real repo root
_ON_KAGGLE = Path("/kaggle").exists()

def _p(env_var: str, local: str, kaggle: str) -> Path:
    """Return env-var value if set, else kaggle or local default."""
    val = os.environ.get(env_var)
    if val:
        return Path(val)
    return Path(kaggle) if _ON_KAGGLE else (_HERE / local)

# ── four required paths ──────────────────────────────────────────────────────
DATA_ROOT   = _p("DATA_ROOT",   "data",           "/kaggle/input/camelyon17-wilds")
SUBSET_DIR  = _p("SUBSET_DIR",  "data/subsets",   "/kaggle/input/camelyon17-wilds-subsets")
CACHE_DIR   = _p("CACHE_DIR",   "results/cache",  "/kaggle/working/cache")
RESULTS_DIR = _p("RESULTS_DIR", "results",        "/kaggle/working/results")

# ── normalised data input and checkpoint output dirs ────────────────────────
DATA_DIR    = _p("DATA_DIR",    "results/cache",  "/kaggle/input/camelyon17-wilds-normalised")
CKPT_DIR    = _p("CKPT_DIR",    "results/cache",  "/kaggle/working/cache")

# ── additional output dirs ───────────────────────────────────────────────────
FIGURES_DIR = _p("FIGURES_DIR", "figures",        "/kaggle/working/figures")

# ── repo root (for notebooks that need sys.path surgery) ────────────────────
SRC_ROOT = Path(os.environ.get("SRC_ROOT", str(_HERE)))

# ── backward-compatible alias kept for the Kaggle notebooks ─────────────────
CAMELYON_ROOT = DATA_ROOT   # Stage-A notebook uses this name

# ── experiment knobs ─────────────────────────────────────────────────────────
N_REF_PATCHES  = int(os.environ.get("N_REF_PATCHES",   200))
N_NORM_PATCHES = int(os.environ.get("N_NORM_PATCHES",  2000))
SHARD_SIZE     = int(os.environ.get("SHARD_SIZE",       500))
SEED           = int(os.environ.get("SEED",               0))

# ── normalisation methods (Stage A) ─────────────────────────────────────────
NORM_METHODS = [
    ("none",        None),
    ("reinhard",    None),
    ("macenko",     None),
    ("ot_balanced", dict(mode="balanced",   eps=0.1, n_sub=500)),
    ("ot_unbal",    dict(mode="unbalanced", eps=0.1, rho=1.0, n_sub=500)),
]

# ── hospital splits ──────────────────────────────────────────────────────────
TRAIN_HOSPITALS = [0, 3, 4]
VAL_HOSPITALS   = [1]
TEST_HOSPITALS  = [2]
ALL_HOSPITALS   = TRAIN_HOSPITALS + VAL_HOSPITALS + TEST_HOSPITALS
