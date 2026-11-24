"""Complete Gromov-Wasserstein Stain Normalizer pipeline.

Combines:
- Beer-Lambert Optical Density conversion and Macenko stain separation
- Intra-tissue distance matrices capturing spatial/morphological tissue layout
- Unbalanced Fused Gromov-Wasserstein (U-FGW) optimal transport
- Structure-preserving barycentric mapping and spatial KNN out-of-sample extension
- Exact background preservation
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from gw_extension import config as gw_cfg
from gw_extension import gw_solver
from gw_extension import gw_utils


def macenko_stain_matrix(img: np.ndarray, alpha: float = 1.0, min_pixels: int = 100) -> np.ndarray:
    """Estimate the 3x2 stain matrix W using SVD/PCA on tissue optical density.

    Columns correspond to Hematoxylin and Eosin (unit norm).
    """
    default_HE = np.array([[0.5626, 0.2159],
                           [0.7201, 0.8012],
                           [0.4062, 0.5581]])

    od = gw_utils.rgb_to_od(img).reshape(-1, 3)
    m = gw_utils.tissue_mask(img).reshape(-1)
    od = od[m]

    if len(od) < min_pixels:
        return default_HE / np.linalg.norm(default_HE, axis=0, keepdims=True)

    eigvals, eigvecs = np.linalg.eigh(np.cov(od.T))
    v = eigvecs[:, 1:3].copy()
    for k in range(2):
        if v[0, k] < 0:
            v[:, k] *= -1

    proj = od @ v
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(phi, [alpha, 100.0 - alpha])
    v1 = v @ np.array([np.cos(lo), np.sin(lo)])
    v2 = v @ np.array([np.cos(hi), np.sin(hi)])

    # Hematoxylin has the larger red-channel component
    hem, eos = (v1, v2) if v1[0] > v2[0] else (v2, v1)
    W = np.stack([hem, eos], axis=1)
    return W / np.linalg.norm(W, axis=0, keepdims=True)


def concentrations(od: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Calculate stain concentrations H = pinv(W) @ OD >= 0."""
    flat = od.reshape(-1, 3)
    H = np.clip(flat @ np.linalg.pinv(W).T, 0.0, None)
    return H.reshape(*od.shape[:-1], 2)


class GWNormalizer:
    """Structure-Preserving Optimal Transport Stain Normalizer.

    Parameters
    ----------
    alpha : float
        Trade-off between stain feature transport (concentration) and
        Gromov-Wasserstein structure preservation.
    eps : float
        Entropic regularizer for Sinkhorn updates.
    rho : float
        Marginal relaxation for unbalanced transport.
    n_sub : int
        Number of representative points sampled for the quadratic solve.
    struct_mode : {"spatial", "morphology", "hybrid"}
        How intra-tissue distance matrices C_s and C_t are constructed.
    spatial_weight : float
        Influence of spatial coordinates in the out-of-sample KNN interpolation.
    k : int
        Nearest neighbors for out-of-sample mapping extension.
    seed : int
        Random seed for reproducible point sampling.
    cond_max : float
        Condition number limit above which reference stain matrix W_ref is used.
    dark_od : float
        Maximum total optical density for sampled tissue pixels.
    """

    def __init__(
        self,
        alpha: float = gw_cfg.GW_ALPHA,
        eps: float = gw_cfg.GW_EPS,
        rho: float = gw_cfg.GW_RHO,
        n_sub: int = gw_cfg.GW_N_POINTS,
        struct_mode: str = gw_cfg.GW_STRUCT_MODE,
        spatial_weight: float = 0.5,
        k: int = 5,
        seed: int = 0,
        min_tissue: int = 200,
        cond_max: float = 20.0,
        dark_od: float = 6.0,
        max_iter: int = gw_cfg.GW_MAX_ITER,
    ):
        self.alpha = alpha
        self.eps = eps
        self.rho = rho
        self.n_sub = n_sub
        self.struct_mode = struct_mode
        self.spatial_weight = spatial_weight
        self.k = k
        self.seed = seed
        self.min_tissue = min_tissue
        self.cond_max = cond_max
        self.dark_od = dark_od
        self.max_iter = max_iter

        # Reference state
        self.W_ref: Optional[np.ndarray] = None
        self.H_ref: Optional[np.ndarray] = None
        self.coords_ref: Optional[np.ndarray] = None
        self.C_ref: Optional[np.ndarray] = None
        self.scale: float = 1.0

        # Diagnostics counters
        self.n_images = 0
        self.n_fallback = 0
        self.n_nonconverged = 0

    def fit(self, ref_images: List[np.ndarray]) -> GWNormalizer:
        """Fit reference stain matrix and structure representations from reference images."""
        rng = np.random.default_rng(self.seed)
        Ws = []
        sampled_H = []
        sampled_coords = []
        sampled_morpho = []

        per_img = max(50, int(np.ceil(2 * self.n_sub / max(len(ref_images), 1))))

        for img in ref_images:
            m = gw_utils.tissue_mask(img)
            if m.sum() < self.min_tissue:
                continue

            W = macenko_stain_matrix(img)
            if np.linalg.cond(W) > self.cond_max:
                continue

            od = gw_utils.rgb_to_od(img)
            keep = m & (od.sum(axis=-1) <= self.dark_od)
            if keep.sum() < self.min_tissue:
                continue

            H_img = concentrations(od, W)
            coords, _, morpho = gw_utils.extract_tissue_points(
                img, n_points=per_img, seed=self.seed
            )

            # Sample concentrations at those coordinates
            H_pts = []
            H_img_shape = img.shape[:2]
            for coord in coords:
                y = int(np.clip(coord[0] * (H_img_shape[0] - 1), 0, H_img_shape[0] - 1))
                x = int(np.clip(coord[1] * (H_img_shape[1] - 1), 0, H_img_shape[1] - 1))
                H_pts.append(H_img[y, x])

            Ws.append(W)
            sampled_H.append(np.array(H_pts))
            sampled_coords.append(coords)
            sampled_morpho.append(morpho)

        if not Ws:
            raise ValueError("No usable reference images with stable stain estimation found.")

        W = np.mean(Ws, axis=0)
        self.W_ref = W / np.linalg.norm(W, axis=0, keepdims=True)

        all_H = np.concatenate(sampled_H, axis=0)
        all_coords = np.concatenate(sampled_coords, axis=0)
        all_morpho = np.concatenate(sampled_morpho, axis=0)

        if len(all_H) > self.n_sub:
            idx = rng.choice(len(all_H), size=self.n_sub, replace=False)
            self.H_ref = all_H[idx]
            self.coords_ref = all_coords[idx]
            morpho_sub = all_morpho[idx]
        else:
            self.H_ref = all_H
            self.coords_ref = all_coords
            morpho_sub = all_morpho

        self.scale = float(np.sqrt(self.H_ref.var(axis=0).sum())) + 1e-8

        # Build reference distance matrix C_t
        self.C_ref = gw_utils.build_distance_matrix(
            coords=self.coords_ref,
            morpho=morpho_sub,
            mode=self.struct_mode,
        )
        return self

    def fit_from_reference_dict(self, ref_dict: dict, ref_img: Optional[np.ndarray] = None) -> GWNormalizer:
        """Initialize reference stain matrix from teammates' existing reference.pkl.

        Guarantees strict compatibility with OT-balanced and OT-unbalanced baselines.
        """
        self.W_ref = ref_dict["W_ref"]
        # Use existing ot_unbal H_ref if available
        if "ot_unbal" in ref_dict and hasattr(ref_dict["ot_unbal"], "H_ref"):
            ot_u = ref_dict["ot_unbal"]
            self.H_ref = ot_u.H_ref.copy()
            self.scale = ot_u.scale
        elif "ot_bal" in ref_dict and hasattr(ref_dict["ot_bal"], "H_ref"):
            ot_b = ref_dict["ot_bal"]
            self.H_ref = ot_b.H_ref.copy()
            self.scale = ot_b.scale

        # Build structural reference points from ref_img
        target_img = ref_img if ref_img is not None else ref_dict.get("ref_img")
        if target_img is not None:
            coords, _, morpho = gw_utils.extract_tissue_points(
                target_img, n_points=len(self.H_ref), seed=self.seed
            )
            self.coords_ref = coords
            self.C_ref = gw_utils.build_distance_matrix(
                coords=coords, morpho=morpho, mode=self.struct_mode
            )
        else:
            # Synthetic spatial coordinates on uniform grid
            grid_side = int(np.ceil(np.sqrt(len(self.H_ref))))
            gy, gx = np.meshgrid(np.linspace(0, 1, grid_side), np.linspace(0, 1, grid_side))
            self.coords_ref = np.stack([gy.ravel(), gx.ravel()], axis=1)[:len(self.H_ref)]
            self.C_ref = gw_utils.build_distance_matrix(self.coords_ref, mode="spatial")

        return self

    def transform(self, img: np.ndarray) -> np.ndarray:
        """Normalise one uint8 RGB patch to the reference using Unbalanced GW.

        Parameters
        ----------
        img : np.ndarray of shape (H, W, 3), dtype uint8

        Returns
        -------
        norm_img : np.ndarray of shape (H, W, 3), dtype uint8
        """
        if self.H_ref is None or self.C_ref is None or self.W_ref is None:
            raise RuntimeError("GWNormalizer must be fitted before calling transform().")

        m = gw_utils.tissue_mask(img, thresh=0.15)
        if m.sum() < self.min_tissue:
            return img.copy()

        self.n_images += 1
        H_shape, W_shape, _ = img.shape
        od = gw_utils.rgb_to_od(img)

        # Estimate source stain matrix W
        W = macenko_stain_matrix(img)
        if np.linalg.cond(W) > self.cond_max:
            W = self.W_ref
            self.n_fallback += 1

        # All tissue concentrations: shape (N_tissue, 2)
        H_tissue = concentrations(od, W)[m]

        # All tissue spatial coordinates: shape (N_tissue, 2)
        y_tissue, x_tissue = np.where(m)
        coords_tissue = np.stack([y_tissue / (H_shape - 1), x_tissue / (W_shape - 1)], axis=1)

        # Sample representative points for GW solve
        n_sample = min(self.n_sub, len(H_tissue))
        coords_s, _, morpho_s = gw_utils.extract_tissue_points(
            img, n_points=n_sample, seed=self.seed, dark_od_thresh=self.dark_od
        )

        # Interpolate concentration at sampled coordinates
        y_samp = np.clip(np.round(coords_s[:, 0] * (H_shape - 1)).astype(int), 0, H_shape - 1)
        x_samp = np.clip(np.round(coords_s[:, 1] * (W_shape - 1)).astype(int), 0, W_shape - 1)
        H_s = concentrations(od, W)[y_samp, x_samp]

        # 1. Feature cost M in concentration space
        X = H_s / self.scale
        Y = self.H_ref / self.scale
        diff_feature = X[:, None, :] - Y[None, :, :]
        M = np.sum(diff_feature**2, axis=-1)   # (n, m)

        # 2. Source structural distance matrix C_s
        C_s = gw_utils.build_distance_matrix(
            coords=coords_s,
            morpho=morpho_s,
            mode=self.struct_mode,
        )

        # 3. Solve Unbalanced Fused Gromov-Wasserstein
        pi, info = gw_solver.unbalanced_fused_gromov_wasserstein(
            M=M,
            C_s=C_s,
            C_t=self.C_ref,
            alpha=self.alpha,
            eps=self.eps,
            rho=self.rho,
            max_iter=self.max_iter,
        )

        if not info["converged"]:
            self.n_nonconverged += 1

        # 4. Barycentric projection for sampled points
        # T_sub(i) = sum_j pi(i, j) * Y(j) / sum_j pi(i, j)
        row_mass = pi.sum(axis=1)
        ok = np.isfinite(row_mass) & (row_mass > 1e-12 / len(X))

        if ok.sum() < 2:
            # Fallback to nearest neighbor in feature space if degenerate
            nearest_idx = M.argmin(axis=1)
            T_sub = self.H_ref[nearest_idx]
            ok = np.ones(len(X), dtype=bool)
        else:
            T_sub = np.zeros_like(X)
            T_sub[ok] = (pi[ok] @ Y) / row_mass[ok, None]
            T_sub = T_sub * self.scale

        # 5. Out-of-sample extension via spatial-concentration KNN
        # Feature vector combines concentration and spatial location:
        # phi = [H, spatial_weight * scale * coords]
        phi_sub = np.concatenate([
            H_s[ok],
            self.spatial_weight * self.scale * coords_s[ok]
        ], axis=1)

        phi_all = np.concatenate([
            H_tissue,
            self.spatial_weight * self.scale * coords_tissue
        ], axis=1)

        knn = KNeighborsRegressor(
            n_neighbors=min(self.k, int(ok.sum())),
            weights="distance",
        )
        knn.fit(phi_sub, T_sub[ok])
        T_all = np.clip(knn.predict(phi_all), 0.0, None)

        # 6. Reconstruct normalized RGB image
        out = img.copy()
        norm_od_tissue = T_all @ self.W_ref.T
        out[m] = gw_utils.od_to_rgb(norm_od_tissue)
        return out
