"""1-D optimal transport helpers: quantile maps and the gap left by affine maps."""
import numpy as np


def quantile_grid(n=1000):
    """Midpoint grid u_i = (i + 0.5) / n on (0, 1)."""
    return (np.arange(n) + 0.5) / n


def quantile_map_samples(src, tgt):
    """Optimal 1-D map T* = F_t^{-1} o F_s, evaluated at every source sample.

    The k-th smallest source value goes to the k-th smallest target value
    (exact when both samples have the same size).
    """
    src = np.asarray(src, dtype=float)
    tgt = np.asarray(tgt, dtype=float)
    ranks = np.argsort(np.argsort(src))
    u = (ranks + 0.5) / len(src)
    return np.quantile(tgt, u, method="inverted_cdf")


def affine_percentile(qs, qt, u, lo=0.01, hi=0.99):
    """Line a*x + b that sends source quantile lo -> target quantile lo, hi -> hi."""
    s_lo, s_hi = np.interp([lo, hi], u, qs)
    t_lo, t_hi = np.interp([lo, hi], u, qt)
    a = (t_hi - t_lo) / (s_hi - s_lo)
    return a, t_lo - a * s_lo


def scale_p99(qs, qt, u, hi=0.99):
    """Macenko-style pure rescaling: a = ratio of 99th percentiles, b = 0."""
    s_hi, t_hi = np.interp(hi, u, qs), np.interp(hi, u, qt)
    return t_hi / s_hi, 0.0


def affine_moment(qs, qt):
    """Line matching mean and standard deviation (Reinhard-style)."""
    a = qt.std() / qs.std()
    return a, qt.mean() - a * qs.mean()


def w2_sq_affine(qs, qt, a, b):
    """Exact squared Wasserstein-2 distance between (a*X + b) and the target.

    For a > 0 the quantile function of a*X + b is a*q_s + b, and in 1-D
    W2^2 = integral of (difference of quantile functions)^2 du.
    """
    if a <= 0:
        raise ValueError("affine map must be increasing (a > 0)")
    return float(np.mean((a * qs + b - qt) ** 2))


def relative_gap(qs, qt, a, b):
    """W2^2 left by the affine map, divided by the target variance."""
    return w2_sq_affine(qs, qt, a, b) / float(np.var(qt))