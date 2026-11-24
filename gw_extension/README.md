# GW Extension — OT-Unbalanced + Gromov-Wasserstein Stain Normalisation

> **Independent contribution** — this folder is entirely self-contained.
> It does **not** modify any file in `src/`, `scripts/`, or `notebooks/`.

---

## What This Does

Classical OT stain normalisation (OT-balanced, OT-unbalanced) maps stain
concentrations independently of spatial tissue layout.  When tissue structures
are very different across hospitals, that can displace nuclei, break cell
boundaries, or create artefacts that hurt downstream classifiers.

This extension adds a **Gromov-Wasserstein structural penalty** to the transport
objective.  The combined **Unbalanced Fused Gromov-Wasserstein (U-FGW)** problem
minimises:

```
(1 − α) ⟨M, π⟩  +  α Σ_{ijkl} |C_s(i,k) − C_t(j,l)|² π_{ij} π_{kl}
        + ρ · KL(π 1 ‖ p_s)  +  ρ · KL(πᵀ 1 ‖ p_t)
        + ε · KL(π ‖ p_s ⊗ p_t)
```

| Term | Meaning |
|------|---------|
| M    | Stain-concentration cost (hematoxylin / eosin distances) |
| C_s, C_t | Intra-tissue pairwise distances (spatial or morphological) |
| ρ    | Marginal relaxation — allows total stain mass to change |
| ε    | Entropic regularisation — Sinkhorn-type fast solve |
| α    | Trade-off: 0 = pure OT, 1 = pure GW |

---

## Folder Layout

```
gw_extension/
├── config.py          Independent config (inherits root config.py if present)
├── gw_utils.py        OD conversion, tissue sampling, C_s/C_t, EPI/SSIM metrics
├── gw_solver.py       NumPy U-FGW solver (no POT dependency)
├── gw_normalizer.py   GWNormalizer class — fit() / transform()
├── prepare_data.py    Build ot_gw cache from patch arrays
├── train_gw.py        ← Entry-point: data prep + ResNet training
├── evaluate.py        Structural metrics (EPI, SSIM, GW distortion, runtime)
├── compare.py         Comparison table + figures vs team baselines
├── notebooks/
│   └── kaggle_gw.py   Single Kaggle notebook (5 cells, end-to-end)
├── results/           GW result JSONs and CSVs (gitignored content)
└── figures/           GW visualisation outputs
```

---

## Quick Start

### Option A — Pilot run (no real data needed, ~5 min on CPU)

```bash
# From repo root
python gw_extension/train_gw.py --pilot
```

This generates synthetic H&E patches for 5 hospitals, normalises them with
GW, trains ResNet-18 for 2 epochs on 200 patches, and prints a summary.

### Option B — Full run on real WILDS data

```bash
# Step 1: build GW-normalised cache (CPU, ~2–4 h on full dataset)
python gw_extension/prepare_data.py --splits train val test

# Step 2: train ResNet-18 on GW patches (GPU recommended)
python gw_extension/train_gw.py --skip_data_prep --seeds 0 1 2 --epochs 10

# Step 3: evaluate structural metrics
python gw_extension/evaluate.py --splits train --limit 500 --save_fig

# Step 4: compare vs team baselines
python gw_extension/compare.py
```

### Option C — Kaggle (GPU T4, end-to-end, ~30–90 min)

Upload `gw_extension/notebooks/kaggle_gw.py` as a Kaggle notebook.
Set Accelerator → GPU T4, Internet → On, then Run All.

If the team's `camelyon17-wilds-subsets` dataset is attached, it uses real
data; otherwise it auto-generates synthetic patches and runs the full pipeline
to completion.

---

## Configuration

All hyperparameters can be overridden via environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `GW_ALPHA` | 0.3 | FGW trade-off (0 = pure OT, 1 = pure GW) |
| `GW_EPS` | 0.05 | Entropic regularisation |
| `GW_RHO` | 1.0 | Marginal relaxation (unbalanced) |
| `GW_N_POINTS` | 300 | Tissue points sampled for GW solve |
| `GW_STRUCT_MODE` | spatial | `"spatial"`, `"morphology"`, or `"hybrid"` |
| `GW_MAX_ITER` | 50 | Outer FGW iterations |
| `GW_INNER_ITER` | 200 | Inner Sinkhorn iterations |
| `GW_RESULTS_DIR` | `gw_extension/results/` | GW output CSVs |
| `GW_FIGURES_DIR` | `gw_extension/figures/` | GW figures |

---

## Outputs

After a full run:

```
gw_extension/results/
├── ot_gw_seed0.json        AUC / accuracy per seed
├── ot_gw_seed1.json
├── ot_gw_seed2.json
├── ot_gw_aggregate.csv     Mean ± std across seeds
├── comparison_table.csv    All methods side-by-side
├── structural_metrics.csv  Per-patch EPI / SSIM / GW distortion / runtime
└── structural_metrics_summary.csv

gw_extension/figures/
├── comparison_auc.png       Bar chart: test AUC for all methods
├── comparison_structural.png  Structural metrics (GW only)
└── structural_eval_train.png  Before/after visual comparison
```

---

## How It Integrates with the Team's Pipeline

The GW normaliser writes its output arrays to exactly the same format and
location that the team's `scripts/train.py` expects:

```
CACHE_DIR/ot_gw/X_{split}.npy       uint8  (N, 96, 96, 3)
CACHE_DIR/ot_gw/y_{split}.npy       int32  (N,)
CACHE_DIR/ot_gw/hospital_{split}.npy int32  (N,)
```

Training is therefore:

```bash
python scripts/train.py --method ot_gw --seed 0
```

The GW extension result JSONs are written to `RESULTS_DIR/ot_gw_seed{seed}.json`
(same location as team results) **and** copied to `gw_extension/results/` so
they are easily findable in the independent folder.

---

## Dependencies

No new packages beyond what the team already uses:

```
numpy scipy scikit-image scikit-learn matplotlib pandas
```

The GW solver (`gw_solver.py`) is **pure NumPy** — it does not depend on POT.
POT is only used by the team's OT baselines (`src/ot_norm.py`).

---

## Key References

1. Macenko et al. *A method for normalizing histology slides.* ISBI 2009.  
2. Chizat, Peyré, Schmitzer, Vialard. *Scaling algorithms for unbalanced OT.* Math. Comp. 87, 2018.  
3. Mémoli. *Gromov-Wasserstein distances and the metric approach to object matching.* FoCM 11, 2011.  
4. Flamary et al. *POT: Python Optimal Transport.* JMLR 22, 2021.
