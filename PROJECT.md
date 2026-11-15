# Project: Stain Normalization as Structure-Preserving Optimal Transport in OD Space
Course project (CS1503). Target: 8-10 page paper (MICCAI-style).

## Model
Optical density OD = -log(I/I0) ≈ W H, W ∈ R^{3x2} stain matrix, H ≥ 0 concentrations.
Source concentration cloud μ_s, reference cloud μ_t. Find map T: μ_s -> μ_t that
(i) minimises transport cost, (ii) preserves tissue structure via a Gromov-Wasserstein
distortion term  Σ_ijkl |d_s(i,k) - d_t(j,l)|^2 π_ij π_kl.
Normalised image: I_norm = I0 * exp(-W_ref * T(H)).

## Three deliverables
1. Proposition: in 1-D with convex cost the optimal map is monotone quantile matching
   T = F_t^{-1} ∘ F_s. Macenko percentile rescaling is an affine approximation. Quantify the gap.
2. Unbalanced OT (Chizat et al. 2018): KL-relaxed marginals, handles stain-intensity (mass)
   changes. Derive Sinkhorn-type scaling updates.
3. Stability: normalised image is Lipschitz in perturbations of estimated W.

## Data (all four)
CAMELYON17-WILDS (hospitals 0,3,4 train; 1 val; 2 test), 96x96 patches.
PCam (patch), CAMELYON16/17 slides (leave-one-center-out), MIDOG (leave-one-scanner-out).
Slide-level: fit stain matrix and OT map once per slide on thumbnail tissue pixels.

## Methods to compare
none, colour augmentation only (HED jitter), Reinhard, Macenko, Vahadane,
OT-balanced, OT-unbalanced, OT-unbalanced+GW.

## Metrics
Downstream accuracy/AUC on held-out hospitals (3 seeds), edge structural similarity
before/after, runtime per image.
Every experiment writes a CSV under results/ with columns: dataset, method, split, seed, metric, value.

## Rules for the agent
- Tissue mask first (exclude background pixels with OD sum < 0.15).
- Subsample <= 2000 pixels per image for OT; extend the map to all pixels via kNN on barycentric projections.
- Use POT (ot.sinkhorn, ot.unbalanced.sinkhorn_unbalanced, ot.gromov). Fix random seeds.
- Write a pytest for every module. Do not write results into the paper without the code that produced them.
- Run Python as `python -m pip` / `python -m pytest` (the pip and pytest launchers are blocked on this laptop).
- Never delete files in data/.