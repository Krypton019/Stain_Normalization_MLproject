"""Plain NumPy Sinkhorn solvers: balanced and unbalanced entropic optimal transport.

a and b must be strictly positive weight vectors; C is the cost matrix (len(a), len(b)).
"""
import numpy as np


def _check(a, b, C):
    a, b, C = (np.asarray(t, dtype=float) for t in (a, b, C))
    if (a <= 0).any() or (b <= 0).any():
        raise ValueError("a and b must be strictly positive")
    if C.shape != (len(a), len(b)):
        raise ValueError("C must have shape (len(a), len(b))")
    return a, b, C


def sinkhorn_balanced(a, b, C, eps, n_iter=10000, tol=1e-10):
    """Entropic OT plan with exact marginals. Updates u = a/(K v), v = b/(K^T u)."""
    a, b, C = _check(a, b, C)
    K = np.exp(-C / eps)
    u, v = np.ones_like(a), np.ones_like(b)
    for _ in range(n_iter):
        u_new = a / (K @ v)
        v = b / (K.T @ u_new)
        done = np.max(np.abs(np.log(u_new) - np.log(u))) < tol
        u = u_new
        if done:
            break
    return u[:, None] * K * v[None, :]


def sinkhorn_unbalanced(a, b, C, eps, rho, reg_type="kl", n_iter=10000, tol=1e-10):
    """Unbalanced entropic OT plan (Chizat et al.).

    Updates u = (a / K v)^fi, v = (b / K^T u)^fi with fi = rho / (rho + eps).
    As rho -> infinity, fi -> 1 and this becomes balanced Sinkhorn.

    reg_type="kl":      penalty eps * KL(P | a (x) b); kernel K = exp(-C/eps) * a b^T.
                        This is the convention in the paper and POT's default.
    reg_type="entropy": penalty eps * sum P (log P - 1); kernel K = exp(-C/eps).
    """
    a, b, C = _check(a, b, C)
    if reg_type not in ("kl", "entropy"):
        raise ValueError("reg_type must be 'kl' or 'entropy'")
    fi = rho / (rho + eps)
    K = np.exp(-C / eps)
    if reg_type == "kl":
        K = K * a[:, None] * b[None, :]
    u, v = np.ones_like(a), np.ones_like(b)
    for _ in range(n_iter):
        u_new = (a / (K @ v)) ** fi
        v = (b / (K.T @ u_new)) ** fi
        done = np.max(np.abs(np.log(u_new) - np.log(u))) < tol
        u = u_new
        if done:
            break
    return u[:, None] * K * v[None, :]