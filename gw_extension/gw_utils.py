"""Tissue feature representations, distance matrices, and structural metrics for GW.

Provides:
- Extraction of spatial coordinates and morphological descriptors from tissue patches.
- Construction and normalization of intra-domain distance matrices (C_s, C_t).
- Structural evaluation metrics:
    - Edge preservation index (gradient correlation)
    - Structural similarity (SSIM)
    - Gromov-Wasserstein distortion objective
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import color


# ─────────────────────────────────────────────────────────────────────────────
# 1. Optical Density & Tissue Helpers (Self-contained, non-modifying)
# ─────────────────────────────────────────────────────────────────────────────

I0 = 255.0
EPS = 1.0

def rgb_to_od(img: np.ndarray) -> np.ndarray:
    """uint8 RGB (..., 3) -> optical density (float64, same shape)."""
    I = img.astype(np.float64)
    return -np.log((I + EPS) / (I0 + EPS))


def od_to_rgb(od: np.ndarray) -> np.ndarray:
    """float OD (..., 3) -> uint8 RGB."""
    I = (I0 + EPS) * np.exp(-od) - EPS
    return np.clip(np.round(I), 0, 255).astype(np.uint8)


def tissue_mask(img: np.ndarray, thresh: float = 0.15) -> np.ndarray:
    """Boolean mask: True where optical density summed over channels > thresh."""
    return rgb_to_od(img).sum(axis=-1) > thresh


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tissue Feature Extraction & Distance Matrices
# ─────────────────────────────────────────────────────────────────────────────

def extract_tissue_points(
    img: np.ndarray,
    n_points: int = 300,
    seed: int = 0,
    min_od_thresh: float = 0.15,
    dark_od_thresh: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample representative tissue points and return their coordinates, OD, and local features.

    Returns
    -------
    coords : np.ndarray of shape (K, 2)
        Spatial (row, col) coordinates normalized to [0, 1] x [0, 1].
    od_pts : np.ndarray of shape (K, 3)
        Optical density values at the sampled points.
    morpho : np.ndarray of shape (K, 2)
        Morphological features (local OD gradient magnitude and local OD variance).
    """
    rng = np.random.default_rng(seed)
    H, W, _ = img.shape

    od = rgb_to_od(img)
    od_sum = od.sum(axis=-1)
    mask = (od_sum > min_od_thresh) & (od_sum <= dark_od_thresh)

    y_indices, x_indices = np.where(mask)
    n_available = len(y_indices)

    if n_available == 0:
        # Fallback to center grid if no tissue detected
        grid_y, grid_x = np.meshgrid(
            np.linspace(0.2, 0.8, int(np.sqrt(n_points))),
            np.linspace(0.2, 0.8, int(np.sqrt(n_points)))
        )
        coords = np.stack([grid_y.ravel(), grid_x.ravel()], axis=1)
        od_pts = np.zeros((len(coords), 3), dtype=np.float64)
        morpho = np.zeros((len(coords), 2), dtype=np.float64)
        return coords, od_pts, morpho

    if n_available > n_points:
        chosen = rng.choice(n_available, size=n_points, replace=False)
        y_pts = y_indices[chosen]
        x_pts = x_indices[chosen]
    else:
        chosen = rng.choice(n_available, size=n_points, replace=True)
        y_pts = y_indices[chosen]
        x_pts = x_indices[chosen]

    # Normalized spatial coordinates in [0, 1]
    coords = np.stack([y_pts / (H - 1), x_pts / (W - 1)], axis=1)

    # OD values at sampled locations
    od_pts = od[y_pts, x_pts]

    # Morphological features: gradient magnitude of total OD
    sobel_y = ndimage.sobel(od_sum, axis=0)
    sobel_x = ndimage.sobel(od_sum, axis=1)
    grad_mag = np.hypot(sobel_y, sobel_x)
    grad_pts = grad_mag[y_pts, x_pts, None]

    # Local variance (texture descriptor)
    local_mean = ndimage.uniform_filter(od_sum, size=5)
    local_sq_mean = ndimage.uniform_filter(od_sum**2, size=5)
    local_var = np.maximum(local_sq_mean - local_mean**2, 0.0)
    var_pts = local_var[y_pts, x_pts, None]

    morpho = np.concatenate([grad_pts, var_pts], axis=1)
    return coords, od_pts, morpho


def build_distance_matrix(
    coords: np.ndarray,
    morpho: np.ndarray | None = None,
    mode: str = "spatial",
    beta: float = 0.7,
) -> np.ndarray:
    """Construct an intra-domain pairwise distance matrix C for Gromov-Wasserstein.

    Parameters
    ----------
    coords : np.ndarray of shape (N, 2)
        Spatial coordinates in [0, 1]^2.
    morpho : np.ndarray of shape (N, D), optional
        Morphological feature descriptors.
    mode : {"spatial", "morphology", "hybrid"}
        - "spatial": Euclidean pairwise distance between tissue locations.
        - "morphology": Pairwise distance in morphological descriptor space.
        - "hybrid": Convex combination beta * C_spatial + (1 - beta) * C_morphology.
    beta : float
        Weight for spatial distance in "hybrid" mode.

    Returns
    -------
    C : np.ndarray of shape (N, N)
        Symmetric pairwise distance matrix, normalized such that max(C) = 1.0.
    """
    # 1. Spatial pairwise distance matrix
    # ||p_i - p_j||_2
    diff_spatial = coords[:, None, :] - coords[None, :, :]
    C_spatial = np.sqrt(np.sum(diff_spatial**2, axis=-1))
    max_s = np.max(C_spatial)
    if max_s > 1e-8:
        C_spatial = C_spatial / max_s

    if mode == "spatial" or morpho is None:
        return C_spatial

    # 2. Morphological pairwise distance matrix
    diff_morpho = morpho[:, None, :] - morpho[None, :, :]
    C_morpho = np.sqrt(np.sum(diff_morpho**2, axis=-1))
    max_m = np.max(C_morpho)
    if max_m > 1e-8:
        C_morpho = C_morpho / max_m

    if mode == "morphology":
        return C_morpho

    # 3. Hybrid mode
    C_hybrid = beta * C_spatial + (1.0 - beta) * C_morpho
    max_h = np.max(C_hybrid)
    if max_h > 1e-8:
        C_hybrid = C_hybrid / max_h
    return C_hybrid


# ─────────────────────────────────────────────────────────────────────────────
# 3. Structural Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────

def edge_preservation_index(img_orig: np.ndarray, img_norm: np.ndarray) -> float:
    """Compute normalized cross-correlation between gradient fields of original and normalized image.

    High EPI (~0.9-1.0) indicates that cell boundaries, nuclear edges, and tissue
    morphology are strictly preserved under normalisation.
    """
    # Convert to grayscale luminance
    gray_orig = color.rgb2gray(img_orig)
    gray_norm = color.rgb2gray(img_norm)

    # Compute horizontal and vertical gradients
    gx_orig = ndimage.sobel(gray_orig, axis=1)
    gy_orig = ndimage.sobel(gray_orig, axis=0)
    mag_orig = np.hypot(gx_orig, gy_orig)

    gx_norm = ndimage.sobel(gray_norm, axis=1)
    gy_norm = ndimage.sobel(gray_norm, axis=0)
    mag_norm = np.hypot(gx_norm, gy_norm)

    num = np.sum(mag_orig * mag_norm)
    den = np.sqrt(np.sum(mag_orig**2) * np.sum(mag_norm**2)) + 1e-8
    return float(np.clip(num / den, 0.0, 1.0))


def structural_similarity_score(img_orig: np.ndarray, img_norm: np.ndarray) -> float:
    """Compute SSIM between original and normalized image on grayscale channel."""
    from skimage.metrics import structural_similarity as ssim
    gray_orig = color.rgb2gray(img_orig)
    gray_norm = color.rgb2gray(img_norm)
    val = ssim(gray_orig, gray_norm, data_range=1.0)
    return float(val)


def gw_distortion_objective(
    C_s: np.ndarray,
    C_t: np.ndarray,
    pi: np.ndarray,
) -> float:
    """Compute the Gromov-Wasserstein structural distortion:

    E(pi) = sum_{i,j,k,l} |C_s(i,k) - C_t(j,l)|^2 pi_{i,j} pi_{k,l}

    Vectorized implementation:
    E(pi) = Tr(C_s^2 pi 1 1^T pi^T) + Tr(pi C_t^2 1 1^T pi) - 2 Tr(C_s pi C_t^T pi^T)
    """
    p_s = pi.sum(axis=1)      # (n,)
    p_t = pi.sum(axis=0)      # (m,)

    term1 = np.sum((C_s**2 @ p_s) * p_s) * p_t.sum()
    term2 = np.sum((C_t**2 @ p_t) * p_t) * p_s.sum()
    term3 = 2.0 * np.sum((C_s @ pi @ C_t.T) * pi)

    distortion = term1 + term2 - term3
    return float(np.maximum(distortion, 0.0))
