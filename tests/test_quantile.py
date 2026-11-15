import numpy as np
from scipy.stats import gamma
from src import quantile as q


def _samples(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.gamma(2.0, 1.0, n), rng.gamma(5.0, 0.5, n)


def test_quantile_map_pushes_source_onto_target_exactly():
    src, tgt = _samples()
    mapped = q.quantile_map_samples(src, tgt)
    assert np.mean((np.sort(mapped) - np.sort(tgt)) ** 2) < 1e-20


def test_quantile_map_is_monotone():
    src, tgt = _samples()
    mapped = q.quantile_map_samples(src, tgt)
    assert np.all(np.diff(mapped[np.argsort(src)]) >= 0)


def test_affine_maps_are_exact_when_target_is_an_affine_copy():
    u = q.quantile_grid(1000)
    qs = gamma.ppf(u, 2.0, scale=0.5)
    qt = 3.0 * qs + 1.0
    for a, b in (q.affine_percentile(qs, qt, u), q.affine_moment(qs, qt)):
        assert q.w2_sq_affine(qs, qt, a, b) < 1e-12


def test_w2_gap_equals_l2_distance_between_affine_and_optimal_map():
    src, tgt = _samples()
    a, b = 1.3, 0.2
    t_star = q.quantile_map_samples(src, tgt)
    l2 = np.mean((a * src + b - t_star) ** 2)
    w2 = np.mean((np.sort(a * src + b) - np.sort(tgt)) ** 2)
    assert np.isclose(l2, w2)