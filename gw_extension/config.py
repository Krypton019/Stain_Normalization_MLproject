"""Independent configuration for Gromov-Wasserstein (GW) extension.

Maintains complete independence from the teammates' src/ and scripts/ code,
while remaining fully compatible with the overall project directory layout.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Repo root is the parent of gw_extension/
REPO_ROOT = Path(__file__).resolve().parent.parent
GW_ROOT   = Path(__file__).resolve().parent

# Ensure REPO_ROOT is in sys.path so we can import root config if available
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import config as root_cfg
    _HAVE_ROOT_CFG = True
except ImportError:
    _HAVE_ROOT_CFG = False

_ON_KAGGLE = Path("/kaggle").exists()

def _p(env_var: str, local_default: Path, kaggle_default: Path) -> Path:
    val = os.environ.get(env_var)
    if val:
        return Path(val)
    return kaggle_default if _ON_KAGGLE else local_default

# ── Core dataset & cache directories ─────────────────────────────────────────
if _HAVE_ROOT_CFG:
    DATA_ROOT   = root_cfg.DATA_ROOT
    SUBSET_DIR  = root_cfg.SUBSET_DIR
    CACHE_DIR   = root_cfg.CACHE_DIR
    BASE_RESULTS = root_cfg.RESULTS_DIR
    BASE_FIGURES = root_cfg.FIGURES_DIR
    SEED        = root_cfg.SEED
    TRAIN_HOSPITALS = root_cfg.TRAIN_HOSPITALS
    VAL_HOSPITALS   = root_cfg.VAL_HOSPITALS
    TEST_HOSPITALS  = root_cfg.TEST_HOSPITALS
    ALL_HOSPITALS   = root_cfg.ALL_HOSPITALS
else:
    DATA_ROOT   = _p("DATA_ROOT",   REPO_ROOT / "data",         Path("/kaggle/input/camelyon17-wilds"))
    SUBSET_DIR  = _p("SUBSET_DIR",  REPO_ROOT / "data/subsets", Path("/kaggle/input/camelyon17-wilds-subsets"))
    CACHE_DIR   = _p("CACHE_DIR",   REPO_ROOT / "results/cache", Path("/kaggle/working/cache"))
    BASE_RESULTS = _p("RESULTS_DIR", REPO_ROOT / "results",      Path("/kaggle/working/results"))
    BASE_FIGURES = _p("FIGURES_DIR", REPO_ROOT / "figures",      Path("/kaggle/working/figures"))
    SEED        = int(os.environ.get("SEED", 0))
    TRAIN_HOSPITALS = [0, 3, 4]
    VAL_HOSPITALS   = [1]
    TEST_HOSPITALS  = [2]
    ALL_HOSPITALS   = TRAIN_HOSPITALS + VAL_HOSPITALS + TEST_HOSPITALS

# ── GW specific directories (kept strictly inside gw_extension/) ─────────────
GW_RESULTS_DIR = _p("GW_RESULTS_DIR", GW_ROOT / "results", Path("/kaggle/working/gw_results"))
GW_FIGURES_DIR = _p("GW_FIGURES_DIR", GW_ROOT / "figures", Path("/kaggle/working/gw_figures"))
GW_CACHE_DIR   = CACHE_DIR / "ot_gw"

# Ensure output directories exist
GW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GW_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── GW Algorithm Hyperparameters ─────────────────────────────────────────────
# GW_ALPHA: Trade-off between linear feature transport (concentrations) and
# quadratic GW structure distortion.
# alpha = 0.0 -> Standard OT (no structural constraint)
# alpha = 1.0 -> Pure GW (relies purely on pairwise structural distances)
# 0 < alpha < 1 -> Fused Gromov-Wasserstein (FGW)
GW_ALPHA = float(os.environ.get("GW_ALPHA", 0.3))

# Entropic regularization for Sinkhorn iterations
GW_EPS = float(os.environ.get("GW_EPS", 0.05))

# Marginal relaxation parameter for Unbalanced OT (rho -> inf gives balanced OT)
GW_RHO = float(os.environ.get("GW_RHO", 1.0))

# Number of sampled tissue points for the quadratic problem
GW_N_POINTS = int(os.environ.get("GW_N_POINTS", 300))

# Structural representation mode: "spatial", "morphology", or "hybrid"
GW_STRUCT_MODE = os.environ.get("GW_STRUCT_MODE", "spatial")

# Optimization stopping criteria
GW_MAX_ITER = int(os.environ.get("GW_MAX_ITER", 50))
GW_TOL      = float(os.environ.get("GW_TOL", 1e-5))
GW_INNER_ITER = int(os.environ.get("GW_INNER_ITER", 200))

# ── Downstream Classifier Defaults ───────────────────────────────────────────
DEFAULT_BATCH_SIZE   = 256
DEFAULT_LR           = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_EPOCHS       = 10
DEFAULT_BACKBONE     = "resnet18"
