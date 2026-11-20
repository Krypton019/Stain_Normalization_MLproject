# Stain Normalisation as Optimal Transport (CS1503)

Downstream classification on CAMELYON17-WILDS under five stain-normalisation methods:
**none · Reinhard · Macenko · OT-balanced · OT-unbalanced**.

---

## Results dashboard (GitHub Pages)

The live dashboard at **`https://Krypton019.github.io/Stain_Normalization_MLproject/`**
shows training results, learning curves, and experiment figures automatically
whenever you push to `master`.

### One-time setup

1. Go to **Settings → Pages** in your GitHub repo.
2. Set **Source** to `Deploy from a branch`, branch `master`, folder `/docs`.
3. Save — the site is live within a minute.

### After each Kaggle training run

```bash
# 1. Pull the JSON results from Kaggle (download /kaggle/working/results/)
#    and place them in results/

# 2. Copy results + figures into docs/ and push
python scripts/publish_results.py
git add docs/
git commit -m "results: add training run"
git push
```

The dashboard updates automatically.

---

## Running on Kaggle

You have two workflows to run on Kaggle:
- **Fast Pilot Verification (~5–10 min):** Run `notebooks/kaggle_e2e_pilot.ipynb` on GPU to verify the whole pipeline end-to-end with zero manual file uploads.
- **Full Production Workflow:** Follow the 2-stage workflow below (CPU cache generation → GPU training).

---

### Option A · Fast Pilot Run (Zero setup, ~5–10 min)

1. Open Kaggle → **New Notebook**.
2. Click **File → Import Notebook** and upload `notebooks/kaggle_e2e_pilot.ipynb`.
3. In notebook settings: set **Accelerator** to **GPU T4** or **GPU P100**, and ensure **Internet** is turned **On**.
4. Click **Run All**. It will clone the repo, generate pilot patches, run all 5 normalization methods, train ResNet-18 with AMP, and display the summary AUC comparison table.

---

### Option B · Full Production Workflow (Two Stages)

#### Prerequisites

| What | Where to get it |
|---|---|
| This repo | Cloned automatically from `https://github.com/Krypton019/Stain_Normalization_MLproject.git` (`master`) |
| `camelyon17-wilds-subsets` dataset | Build with `python scripts/make_subset.py` (see [Local workflow](#local-workflow)) and upload to Kaggle as a dataset |

#### 1 · Build the patch subset (one-time)

```bash
# If running locally with CAMELYON17 installed:
python scripts/make_subset.py               # ~60 k train / 10 k val / 15 k test
# If WILDS dataset needs to be downloaded automatically:
python scripts/make_subset.py --download
# Outputs land in data/subsets/
```

Upload the `data/subsets/` folder to Kaggle as a dataset named **`camelyon17-wilds-subsets`**.

#### 2 · Stage 1: Normalisation notebook — CPU, ~2–4 h

Open **`notebooks/kaggle_normalize.ipynb`** (or `kaggle_A_normalize.ipynb`) on Kaggle.

**Notebook Settings:**
- Accelerator: **None (CPU)**
- Internet: **On**

**Dataset to attach:**
- `camelyon17-wilds-subsets` (mounted at `/kaggle/input/camelyon17-wilds-subsets`)

**What it does:**
1. Clones the repo and installs dependencies (`pip install -e .` and `pip install -r requirements.txt`).
2. Calls `scripts/normalize_cache.py` across all 5 methods (`none`, `reinhard`, `macenko`, `ot_bal`, `ot_unbal`), writing resumable shards.
3. Calls `scripts/merge_shards.py` to produce final `X_{split}.npy` arrays.
4. Verifies shapes and logs `failures.csv`.

**Outputs** (save `/kaggle/working/cache/` as a Kaggle dataset, e.g. **`camelyon17-wilds-normalised`**):
```
cache/
  reference.pkl
  none/      X_train.npy  y_train.npy  hospital_train.npy  idx_train.npy  (+ val/test)
  reinhard/  …
  macenko/   …
  ot_bal/    …
  ot_unbal/  …
  failures.csv
```

#### 3 · Stage 2: Training notebook — GPU (T4/P100), ~1–2 h

Open **`notebooks/kaggle_train.ipynb`** (or `kaggle_B_train.ipynb`) on Kaggle.

**Notebook Settings:**
- Accelerator: **GPU T4** or **GPU P100**
- Internet: **On**

**Dataset to attach:**
- `camelyon17-wilds-normalised` (mounted at `/kaggle/input/camelyon17-wilds-normalised`)

**What it does:**
1. Clones the repo and installs dependencies.
2. Loops over **5 methods × 3 seeds** calling `scripts/train.py`.
   - Reads directly from read-only `/kaggle/input/camelyon17-wilds-normalised` via `--data_dir`.
   - Saves checkpoints safely to writable `/kaggle/working/cache/` via `--ckpt_dir` (auto-resuming if interrupted).
   - Mixed-precision training (AMP) on GPU with AdamW + cosine LR.
3. Calls `scripts/aggregate.py` to generate `aggregate.csv`.

**Outputs** (`/kaggle/working/results/`):
```
results/
  none_seed0.json … ot_unbal_seed2.json   (15 files)
  aggregate.csv                            (mean ± std per method)
```

#### 4 · Read results

```python
import pandas as pd
summary = pd.read_csv("results/aggregate.csv")
print(summary[["method", "test_auc_mean", "test_auc_std"]].to_string(index=False))
```

---

## Local workflow

```bash
# 1. Install
git clone https://github.com/Krypton019/Stain_Normalization_MLproject.git
cd Stain_Normalization_MLproject
git checkout master
pip install -e .
pip install -r requirements.txt

# 2. Download CAMELYON17-WILDS data to data/
export DATA_ROOT=$(pwd)/data

# 3. Build subset
python scripts/make_subset.py --debug          # 200 patches, quick test
python scripts/make_subset.py                  # full run

# 4. Normalise
python scripts/normalize_cache.py --method none macenko ot_bal ot_unbal --dry_run
python scripts/normalize_cache.py --method none reinhard macenko ot_bal ot_unbal
python scripts/merge_shards.py

# 5. Train
python scripts/train.py --method ot_bal --seed 0 --epochs 15
python scripts/aggregate.py
```

---

## Repository layout

```
Stain_Normalization_MLproject/
├── config.py                   Central path / knob config (env-var driven)
├── requirements.txt
├── pyproject.toml              pip install -e . makes src/ importable
│
├── src/
│   ├── stainops.py             RGB ↔ OD, tissue mask, Macenko W, Reinhard
│   ├── quantile.py             1-D OT quantile maps and W₂² gap metrics
│   ├── sinkhorn.py             NumPy balanced + unbalanced Sinkhorn solvers
│   ├── ot_norm.py              OTNormalizer (balanced / unbalanced)
│   ├── classifier.py           ShardDataset, ResNet wrapper, AUC (Stage A/B)
│   └── data.py                 CAMELYON17-WILDS loader
│
├── scripts/
│   ├── make_subset.py          Sample reproducible train/val/test .npy arrays
│   ├── normalize_cache.py      Apply normalisation; write resumable shards
│   ├── merge_shards.py         Concatenate shards → X_{split}.npy per method
│   ├── train.py                GPU training with AMP, checkpoint, JSON output
│   └── aggregate.py            JSON → mean ± std table + aggregate.csv
│
├── notebooks/
│   ├── kaggle_normalize.py     Kaggle CPU notebook (cell-marker .py)
│   ├── kaggle_train.py         Kaggle GPU notebook (cell-marker .py)
│   ├── kaggle_A_normalize.ipynb  (earlier Stage-A notebook, kept for reference)
│   └── kaggle_B_train.ipynb      (earlier Stage-B notebook, kept for reference)
│
├── experiments/                Standalone analysis scripts (exp1, exp2, diagnostics)
├── tests/                      pytest suite (runs without WILDS or GPU)
├── figures/                    Experiment output figures
└── results/                    Experiment output CSVs
```

---

## Running tests

```bash
python -m pytest tests/test_stainops.py tests/test_quantile.py \
                 tests/test_classifier.py tests/test_make_subset.py \
                 tests/test_normalize_cache.py tests/test_train.py -q
# 59 passed
```

The two tests that require `POT` and `wilds` (`test_ot_norm.py`, `test_sinkhorn.py`)
are skipped locally unless those packages are installed.

---

## Configuration reference

All paths and knobs are set via environment variables; `config.py` applies
sensible defaults so the code works both locally and on Kaggle without changes.

| Variable | Local default | Kaggle default | Purpose |
|---|---|---|---|
| `DATA_ROOT` | `<repo>/data` | `/kaggle/input/camelyon17-wilds` | WILDS root |
| `SUBSET_DIR` | `<repo>/data/subsets` | `/kaggle/input/camelyon17-wilds-subsets` | Pre-sampled .npy arrays |
| `CACHE_DIR` | `<repo>/results/cache` | `/kaggle/working/cache` | Normalised shard cache |
| `RESULTS_DIR` | `<repo>/results` | `/kaggle/working/results` | JSON / CSV outputs |
| `FIGURES_DIR` | `<repo>/figures` | `/kaggle/working/figures` | PNG figures |
| `SEED` | `0` | `0` | Global random seed |
| `SHARD_SIZE` | `500` | `2000` | Patches per shard |
