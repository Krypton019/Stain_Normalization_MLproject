"""Stain toolkit: RGB <-> optical density, tissue mask, Macenko W, concentrations, Reinhard."""
import numpy as np
from skimage import color

I0 = 255.0
EPS = 1.0  # added to pixel values so log never sees 0

# Macenko's published reference H&E matrix (columns: hematoxylin, eosin); fallback only
DEFAULT_HE = np.array([[0.5626, 0.2159],
                       [0.7201, 0.8012],
                       [0.4062, 0.5581]])


def rgb_to_od(img):
    """uint8 RGB image (..., 3) -> optical density (float, same shape). White -> 0."""
    I = img.astype(np.float64)
    return -np.log((I + EPS) / (I0 + EPS))


def od_to_rgb(od):
    """Inverse of rgb_to_od. Returns uint8 RGB."""
    I = (I0 + EPS) * np.exp(-od) - EPS
    return np.clip(np.round(I), 0, 255).astype(np.uint8)


def tissue_mask(img, thresh=0.15):
    """Boolean mask (H, W): True where the pixel is tissue (OD summed over channels > thresh)."""
    return rgb_to_od(img).sum(axis=-1) > thresh


def macenko_stain_matrix(img, alpha=1.0, min_pixels=100):
    """Estimate the 3x2 stain matrix W (columns = hematoxylin, eosin, unit norm).

    Steps: take tissue OD pixels -> eigenvectors of their 3x3 covariance (the SVD of a
    symmetric matrix) -> project on the top-2 plane -> take the alpha / 100-alpha
    percentile angles as the two stain directions.
    """
    od = rgb_to_od(img).reshape(-1, 3)
    od = od[tissue_mask(img).reshape(-1)]
    if len(od) < min_pixels:
        return DEFAULT_HE / np.linalg.norm(DEFAULT_HE, axis=0, keepdims=True)

    eigvals, eigvecs = np.linalg.eigh(np.cov(od.T))  # ascending eigenvalues
    v = eigvecs[:, 1:3].copy()                       # two largest directions
    for k in range(2):
        if v[0, k] < 0:
            v[:, k] *= -1

    proj = od @ v
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(phi, [alpha, 100 - alpha])
    v1 = v @ np.array([np.cos(lo), np.sin(lo)])
    v2 = v @ np.array([np.cos(hi), np.sin(hi)])

    # Convention: hematoxylin is the direction with the larger red-channel component
    hem, eos = (v1, v2) if v1[0] > v2[0] else (v2, v1)
    W = np.stack([hem, eos], axis=1)
    return W / np.linalg.norm(W, axis=0, keepdims=True)


def concentrations(od, W):
    """Least-squares stain amounts H = pinv(W) @ od, clipped to >= 0. od (..., 3) -> H (..., 2)."""
    flat = od.reshape(-1, 3)
    H = np.clip(flat @ np.linalg.pinv(W).T, 0, None)
    return H.reshape(*od.shape[:-1], 2)


def reinhard(img, ref_img):
    """Reinhard colour transfer: match per-channel mean/std in Lab space. Returns uint8 RGB."""
    src, ref = color.rgb2lab(img), color.rgb2lab(ref_img)
    s_mu, s_sd = src.mean(axis=(0, 1)), src.std(axis=(0, 1)) + 1e-8
    r_mu, r_sd = ref.mean(axis=(0, 1)), ref.std(axis=(0, 1))
    out = (src - s_mu) / s_sd * r_sd + r_mu
    return (np.clip(color.lab2rgb(out), 0, 1) * 255).round().astype(np.uint8)