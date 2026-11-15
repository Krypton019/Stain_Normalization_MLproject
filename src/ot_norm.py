"""OT stain normalizer: map a source image's stain amounts H onto a reference cloud."""
import warnings

import numpy as np
import ot
from sklearn.neighbors import KNeighborsRegressor

from src import stainops as so


class OTNormalizer:
    """mode: "balanced" or "unbalanced".
    eps: entropic blur. rho: marginal-relaxation strength (unbalanced only; larger = stricter).
    n_sub: points sampled per cloud for the OT solve. k: neighbours for extending the map.
    max_iter: Sinkhorn iteration cap.
    cond_max: if the estimated stain matrix has a condition number above this, the
              reference stain matrix is used for that image instead.
    dark_od: pixels with OD summed over channels above this are not used for the OT solve.
    Counters after transform() calls: n_images, n_fallback (unstable W), n_nonconverged,
    n_slow_path (fast solver gave NaN, stable solver used).
    """

    def __init__(self, mode="unbalanced", eps=0.1, rho=1.0, n_sub=500, k=5, seed=0,
                 min_tissue=200, cond_max=20.0, dark_od=6.0, max_iter=1000):
        if mode not in ("balanced", "unbalanced"):
            raise ValueError("mode must be 'balanced' or 'unbalanced'")
        self.mode, self.eps, self.rho = mode, eps, rho
        self.n_sub, self.k, self.seed, self.min_tissue = n_sub, k, seed, min_tissue
        self.cond_max, self.dark_od, self.max_iter = cond_max, dark_od, max_iter
        self.W_ref = None
        self.H_ref = None
        self.scale = None
        self.n_images = self.n_fallback = self.n_nonconverged = self.n_slow_path = 0

    def fit(self, ref_images):
        """Build the reference: average stain matrix + a sampled cloud of stain amounts."""
        rng = np.random.default_rng(self.seed)
        per_image = max(50, int(np.ceil(2 * self.n_sub / len(ref_images))))
        Ws, pts = [], []
        for img in ref_images:
            m = so.tissue_mask(img)
            if m.sum() < self.min_tissue:
                continue
            W = so.macenko_stain_matrix(img)
            if np.linalg.cond(W) > self.cond_max:
                continue
            od = so.rgb_to_od(img)
            keep = m & (od.sum(axis=-1) <= self.dark_od)
            if keep.sum() < self.min_tissue:
                continue
            H = so.concentrations(od, W)[keep]
            idx = rng.choice(len(H), size=min(per_image, len(H)), replace=False)
            Ws.append(W)
            pts.append(H[idx])
        if not Ws:
            raise ValueError("no usable reference image (needs tissue and a stable stain estimate)")
        W = np.mean(Ws, axis=0)
        self.W_ref = W / np.linalg.norm(W, axis=0, keepdims=True)
        H = np.concatenate(pts)
        if len(H) > self.n_sub:
            H = H[rng.choice(len(H), size=self.n_sub, replace=False)]
        self.H_ref = H
        self.scale = float(np.sqrt(H.var(axis=0).sum())) + 1e-8
        return self

    def _solve(self, a, b, C, fast):
        """Fast = plain Sinkhorn; slow = log-domain / stabilised (safe for extreme costs)."""
        if self.mode == "balanced":
            method = "sinkhorn" if fast else "sinkhorn_log"
            return ot.sinkhorn(a, b, C, self.eps, method=method,
                               numItermax=self.max_iter, stopThr=1e-6)
        method = "sinkhorn" if fast else "sinkhorn_stabilized"
        return ot.unbalanced.sinkhorn_unbalanced(a, b, C, self.eps, self.rho, method=method,
                                                 numItermax=self.max_iter, stopThr=1e-6)

    def _map(self, Hs):
        """Barycentric-projection map for sampled source amounts. Returns (T, ok mask)."""
        X, Y = Hs / self.scale, self.H_ref / self.scale
        C = ot.dist(X, Y)                              # squared Euclidean
        n, m = len(X), len(Y)
        a, b = np.full(n, 1.0 / n), np.full(m, 1.0 / m)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            P = self._solve(a, b, C, fast=True)
            if not np.isfinite(P).all() or P.sum() < 1e-12:
                self.n_slow_path += 1
                P = self._solve(a, b, C, fast=False)
        if any("converge" in str(w.message).lower() for w in caught):
            self.n_nonconverged += 1
        mass = P.sum(axis=1)
        ok = np.isfinite(mass) & (mass > 1e-12 / n)
        if ok.sum() < 2:                               # degenerate: nearest reference point
            return Y[C.argmin(axis=1)] * self.scale, np.ones(n, dtype=bool)
        T = np.zeros_like(X)
        T[ok] = (P[ok] @ Y) / mass[ok, None]
        return T * self.scale, ok

    def transform(self, img):
        """Normalise one uint8 RGB image to the fitted reference."""
        if self.H_ref is None:
            raise RuntimeError("call fit() first")
        m = so.tissue_mask(img)
        if m.sum() < self.min_tissue:
            return img.copy()
        self.n_images += 1

        od = so.rgb_to_od(img)
        W = so.macenko_stain_matrix(img)
        if np.linalg.cond(W) > self.cond_max:           # unstable stain estimate
            W = self.W_ref
            self.n_fallback += 1
        Ht = so.concentrations(od, W)[m]                # (N, 2)
        od_sum = od[m].sum(axis=-1)

        rng = np.random.default_rng(self.seed)
        pool = np.where(od_sum <= self.dark_od)[0]      # skip very dark pixels for the solve
        if len(pool) < self.min_tissue:
            pool = np.arange(len(Ht))
        idx = rng.choice(pool, size=min(self.n_sub, len(pool)), replace=False)
        Hs = Ht[idx]
        T_sub, ok = self._map(Hs)

        knn = KNeighborsRegressor(n_neighbors=min(self.k, int(ok.sum())), weights="distance")
        knn.fit(Hs[ok], T_sub[ok])
        T_all = np.clip(knn.predict(Ht), 0, None)

        out = img.copy()
        out[m] = so.od_to_rgb(T_all @ self.W_ref.T)     # background stays as is
        return out