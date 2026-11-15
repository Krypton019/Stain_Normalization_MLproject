import numpy as np
import ot
import pytest
from src import sinkhorn as sk


def _problem(n=30, m=40, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    y = rng.normal(loc=0.5, size=(m, 2))
    C = ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)
    C = C / C.max()                       # keep exp(-C/eps) away from underflow
    a = rng.random(n) + 0.5
    b = rng.random(m) + 0.5
    return a / a.sum(), b / b.sum(), C


def test_balanced_marginals_are_exact():
    a, b, C = _problem()
    P = sk.sinkhorn_balanced(a, b, C, eps=0.05)
    assert np.allclose(P.sum(1), a, atol=1e-8)
    assert np.allclose(P.sum(0), b, atol=1e-8)


def test_balanced_matches_pot():
    a, b, C = _problem()
    P = sk.sinkhorn_balanced(a, b, C, eps=0.05)
    P_ot = ot.sinkhorn(a, b, C, reg=0.05, numItermax=100000, stopThr=1e-12)
    assert np.allclose(P, P_ot, rtol=1e-4, atol=1e-8)


@pytest.mark.parametrize("reg_type", ["kl", "entropy"])
def test_unbalanced_matches_pot(reg_type):
    a, b, C = _problem()
    P = sk.sinkhorn_unbalanced(a, b, C, eps=0.05, rho=1.0, reg_type=reg_type)
    P_ot = ot.unbalanced.sinkhorn_unbalanced(a, b, C, reg=0.05, reg_m=1.0,
                                             reg_type=reg_type,
                                             numItermax=100000, stopThr=1e-12)
    assert np.allclose(P, P_ot, rtol=1e-3, atol=1e-8)


def test_larger_rho_gives_stricter_marginals():
    a, b, C = _problem()
    loose = sk.sinkhorn_unbalanced(a, b, C, eps=0.05, rho=0.1)
    tight = sk.sinkhorn_unbalanced(a, b, C, eps=0.05, rho=100.0)
    assert np.abs(tight.sum(1) - a).max() < np.abs(loose.sum(1) - a).max()