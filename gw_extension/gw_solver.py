"""Gromov-Wasserstein and Unbalanced Fused Gromov-Wasserstein (U-FGW) Solvers.

Implements structure-preserving optimal transport in stain space:
- Feature cost M: distance in stain concentration space H (Hematoxylin, Eosin)
- Intra-domain distance matrices C_s, C_t: spatial/morphological tissue geometry
- Unbalanced marginal relaxation (rho): handles varying tissue densities and staining variations
- Entropic regularization (eps): fast Sinkhorn-type scaling updates
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 1. Standalone Unbalanced Entropic OT Solver (NumPy)
# ─────────────────────────────────────────────────────────────────────────────

def sinkhorn_unbalanced_np(
    a: np.ndarray,
    b: np.ndarray,
    M: np.ndarray,
    eps: float = 0.05,
    rho: float = 1.0,
    n_iter: int = 500,
    tol: float = 1e-7,
) -> np.ndarray:
    """Solve entropic unbalanced optimal transport:

    min_{P >= 0} <M, P> + rho * KL(P 1 || a) + rho * KL(P^T 1 || b) + eps * KL(P || a (x) b)

    Uses the multiplicative Sinkhorn scaling updates (Chizat et al., 2018):
        u <- (a / (K v))^tau
        v <- (b / (K^T u))^tau
    where tau = rho / (rho + eps) and kernel K = exp(-M / eps) * (a (x) b).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    # Normalize cost to avoid numerical overflow in exp(-M / eps)
    M_min = M.min()
    M_scaled = M - M_min
    max_val = np.max(M_scaled)
    if max_val > 50.0 * eps:
        M_scaled = M_scaled * (50.0 * eps / max_val)

    tau = rho / (rho + eps)
    K = np.exp(-M_scaled / eps) * a[:, None] * b[None, :]

    u = np.ones_like(a)
    v = np.ones_like(b)

    for i in range(n_iter):
        Kv = K @ v
        Kv = np.maximum(Kv, 1e-300)
        u_new = (a / Kv) ** tau

        KTu = K.T @ u_new
        KTu = np.maximum(KTu, 1e-300)
        v_new = (b / KTu) ** tau

        err = np.max(np.abs(u_new - u)) / (np.max(np.abs(u)) + 1e-12)
        u, v = u_new, v_new
        if err < tol:
            break

    P = u[:, None] * K * v[None, :]
    return P


# ─────────────────────────────────────────────────────────────────────────────
# 2. Gromov-Wasserstein Gradient Computation
# ─────────────────────────────────────────────────────────────────────────────

def gw_gradient(
    C_s: np.ndarray,
    C_t: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    """Compute the Euclidean gradient of the quadratic Gromov-Wasserstein loss:

    E(P) = sum_{ijkl} (C_s(i,k) - C_t(j,l))^2 P_{ij} P_{kl}

    The gradient with respect to P is:
        grad(P) = 2 * ( (C_s^2 P 1_m) 1_m^T + 1_n (1_n^T P (C_t^2)^T) - 2 C_s P C_t^T )
    """
    p_s = P.sum(axis=1)           # (n,)
    p_t = P.sum(axis=0)           # (m,)

    # Term 1: C_s^2 @ p_s outer 1_m
    term1 = (C_s**2 @ p_s)[:, None] * np.ones_like(p_t)[None, :]

    # Term 2: 1_n outer (C_t^2 @ p_t)
    term2 = np.ones_like(p_s)[:, None] * (C_t**2 @ p_t)[None, :]

    # Term 3: - 2 * C_s @ P @ C_t^T
    term3 = - 2.0 * (C_s @ P @ C_t.T)

    return 2.0 * (term1 + term2 + term3)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Unbalanced Fused Gromov-Wasserstein Solver
# ─────────────────────────────────────────────────────────────────────────────

def unbalanced_fused_gromov_wasserstein(
    M: Optional[np.ndarray],
    C_s: np.ndarray,
    C_t: np.ndarray,
    p_s: Optional[np.ndarray] = None,
    p_t: Optional[np.ndarray] = None,
    alpha: float = 0.3,
    eps: float = 0.05,
    rho: float = 1.0,
    max_iter: int = 50,
    tol: float = 1e-5,
    inner_iter: int = 200,
    damping: float = 0.5,
) -> Tuple[np.ndarray, dict]:
    """Solve the Unbalanced Fused Gromov-Wasserstein problem:

    min_{P >= 0} (1 - alpha) <M, P> + alpha E_{GW}(C_s, C_t, P)
                 + rho * KL(P 1 || p_s) + rho * KL(P^T 1 || p_t)
                 + eps * KL(P || p_s (x) p_t)

    Parameters
    ----------
    M : np.ndarray of shape (n, m) or None
        Linear feature transport cost matrix (e.g. Euclidean distance between stain concentrations).
        If None, pure Gromov-Wasserstein (alpha=1.0) is solved.
    C_s : np.ndarray of shape (n, n)
        Intra-source tissue pairwise distance matrix.
    C_t : np.ndarray of shape (m, m)
        Intra-reference tissue pairwise distance matrix.
    p_s : np.ndarray of shape (n,), optional
        Source marginal distribution. Defaults to uniform 1/n.
    p_t : np.ndarray of shape (m,), optional
        Reference marginal distribution. Defaults to uniform 1/m.
    alpha : float, in [0, 1]
        Trade-off between linear cost M and quadratic GW cost.
        alpha=0: pure Unbalanced OT.
        alpha=1: pure Unbalanced GW.
    eps : float
        Entropic regularization parameter.
    rho : float
        Marginal relaxation parameter for unbalanced transport.
    max_iter : int
        Maximum outer iterations (linearization steps).
    tol : float
        Relative convergence tolerance on transport plan P.
    damping : float in (0, 1]
        Damping factor for the outer update: P <- (1 - damping)*P + damping*P_new.

    Returns
    -------
    P : np.ndarray of shape (n, m)
        Optimal transport coupling matrix.
    info : dict
        Convergence diagnostics (iterations, gradient norm, cost history).
    """
    n, m = C_s.shape[0], C_t.shape[0]

    if p_s is None:
        p_s = np.full(n, 1.0 / n, dtype=np.float64)
    if p_t is None:
        p_t = np.full(m, 1.0 / m, dtype=np.float64)

    # Initial coupling: independent joint distribution
    P = p_s[:, None] * p_t[None, :]

    if M is None or alpha >= 1.0:
        M = np.zeros((n, m), dtype=np.float64)
        alpha = 1.0
    else:
        # Scale M to have similar magnitude to the GW gradient
        M_scale = np.max(M)
        if M_scale > 1e-8:
            M = M / M_scale

    cost_history = []
    converged = False

    for it in range(max_iter):
        # 1. Compute quadratic GW gradient
        G_gw = gw_gradient(C_s, C_t, P)
        max_gw = np.max(np.abs(G_gw))
        if max_gw > 1e-8:
            G_gw = G_gw / max_gw

        # 2. Form linearized surrogate cost matrix
        if alpha == 0.0:
            C_linear = M
        elif alpha == 1.0:
            C_linear = G_gw
        else:
            C_linear = (1.0 - alpha) * M + alpha * G_gw

        # 3. Solve inner unbalanced Sinkhorn problem
        P_sub = sinkhorn_unbalanced_np(
            a=p_s,
            b=p_t,
            M=C_linear,
            eps=eps,
            rho=rho,
            n_iter=inner_iter,
            tol=1e-6,
        )

        # 4. Damped update
        P_new = (1.0 - damping) * P + damping * P_sub

        # 5. Convergence check
        diff = np.linalg.norm(P_new - P) / (np.linalg.norm(P) + 1e-12)
        P = P_new

        cost_history.append(float(diff))
        if diff < tol:
            converged = True
            break

    info = {
        "converged": converged,
        "n_iter": it + 1,
        "final_diff": float(diff),
        "total_mass": float(P.sum()),
    }
    return P, info
