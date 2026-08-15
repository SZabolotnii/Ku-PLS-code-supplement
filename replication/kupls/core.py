"""Ku-PLS — shared machinery.

Sampling of alpha-stable functional predictors, the two competing test
statistics, and their plug-in weighted-chi^2 calibration.

  T_n = n ||Khat (betahat_m - b)||^2      the BCT moment statistic
  S_n = n ||rhohat - rhohat_b||^2         the ECF statistic of Theorem 6.1

Both are calibrated the same way and each by its own asymptotic theory: the
eigenvalues of the relevant plug-in variance operator, with critical values or
p-values obtained from sum_j omega_j Z_j^2.

A note on what the two nulls actually say, because it matters when reading the
empirical results. T_n tests beta = b through the second-moment normal
equation. S_n compares the joint empirical characteristic function against the
restriction rho(s,u) = phi_X(u + s b) phi_eps(s), which encodes eps being
INDEPENDENT of X, not merely mean-independent. With b = 0 that makes S_n a
characteristic-function test of independence in the ICM tradition. On data with
volatility dependence the two can disagree, and the disagreement is a
difference of hypotheses rather than of power.
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------- sampling


def sas(alpha: float, size, rng: np.random.Generator) -> np.ndarray:
    """Symmetric alpha-stable, Chambers-Mallows-Stuck. Exact, not approximate."""
    U = rng.uniform(-np.pi / 2, np.pi / 2, size)
    W = rng.exponential(1.0, size)
    if abs(alpha - 1.0) < 1e-9:
        return np.tan(U)
    a = alpha
    return (
        np.sin(a * U)
        / np.cos(U) ** (1.0 / a)
        * (np.cos(U - a * U) / W) ** ((1.0 - a) / a)
    )


class Design:
    """Fixed across replications: basis, true slope, grid."""

    def __init__(self, p: int = 50, J: int = 20, sigma: float = 0.5):
        self.p, self.J, self.sigma = p, J, sigma
        t = (np.arange(p) + 0.5) / p
        # real Fourier basis, orthonormal w.r.t. the discrete inner product
        phi = np.empty((J, p))
        phi[0] = 1.0
        for j in range(1, J):
            k = (j + 1) // 2
            phi[j] = np.sqrt(2.0) * (np.cos if j % 2 == 1 else np.sin)(
                2 * np.pi * k * t
            )
        self.phi = phi                       # (J, p)
        self.scale = (np.arange(J) + 1.0) ** -1.0     # xi_j scale
        self.beta_c = (np.arange(J) + 1.0) ** -2.0    # slope coefficients

    def draw(self, n: int, alpha: float, rng, eps_law: str = "gauss"):
        xi = sas(alpha, (n, self.J), rng) * self.scale          # (n, J)
        if eps_law == "gauss":
            eps = rng.normal(0.0, self.sigma, n)
        elif eps_law == "cauchy":
            eps = self.sigma * sas(1.0, n, rng)      # E|eps| = inf
        else:
            raise ValueError(eps_law)
        Y = xi @ self.beta_c + eps
        # Work in coefficient coordinates: <u,X> and <b,X> are exact there,
        # and the discrete grid adds nothing but noise.
        return xi, Y, eps


# -------------------------------------------------------- baseline: T_n (BCT)


def cg_pls(K: np.ndarray, r: np.ndarray, m: int) -> np.ndarray:
    """m steps of conjugate gradient on K a = r; a_0 = 0. This IS functional
    PLS in the BCT reframing."""
    a = np.zeros_like(r)
    res = r.copy()
    d = res.copy()
    for _ in range(m):
        Kd = K @ d
        denom = d @ Kd
        if denom <= 0 or not np.isfinite(denom):
            break
        step = (res @ res) / denom
        a = a + step * d
        new = res - step * Kd
        if not np.isfinite(new).all():
            break
        beta_cg = (new @ new) / (res @ res)
        res, d = new, new + beta_cg * d
    return a


def stat_T(xi, Y, b, m):
    """T_n and the eigenvalues of the plug-in variance operator Vhat."""
    n = len(Y)
    K = xi.T @ xi / n
    r = xi.T @ Y / n
    a = cg_pls(K, r, m)
    diff = a - b
    T = n * float(np.sum((K @ diff) ** 2))
    resid = Y - xi @ a
    # Vhat = (1/n) sum eps_k^2 X_k (x) X_k ; eigenvalues via the J x J matrix
    Z = xi * resid[:, None]
    V = Z.T @ Z / n
    w = np.linalg.eigvalsh(V)
    return T, np.clip(w, 0.0, None)


# ------------------------------------------------------- this paper: S_n (ECF)


class Grid:
    """Finite discrete frequency measure pi on the joint (s,u) space.

    pi is discrete, symmetric and finite -- all the manuscript requires
    (pi(U) < inf, 0 in supp pi). Frequencies are placed procedurally, scaled
    by a robust spread of the data so the grid sits where the CF is
    informative rather than at an arbitrary absolute location.
    """

    def __init__(self, n_dirs: int = 4, mags=(0.5, 1.0), s_mags=(0.5, 1.0)):
        self.n_dirs, self.mags, self.s_mags = n_dirs, mags, s_mags

    def build(self, xi, Y, b):
        def mad(v):
            return np.median(np.abs(v - np.median(v))) + 1e-12

        sY = mad(Y)
        pairs = []                       # (s, u) with u a J-vector
        J = xi.shape[1]
        for j in range(self.n_dirs):
            sj = mad(xi[:, j])
            for t in self.mags:
                for sgn in (1.0, -1.0):
                    u = np.zeros(J)
                    u[j] = sgn * t / sj
                    pairs.append((0.0, u))            # marginal in X
                    for sm in self.s_mags:
                        pairs.append((sgn * sm / sY, u))
        for sm in self.s_mags:                        # marginal in Y
            for sgn in (1.0, -1.0):
                pairs.append((sgn * sm / sY, np.zeros(J)))
        self.s = np.array([p[0] for p in pairs])
        self.U = np.array([p[1] for p in pairs])      # (N, J)
        self.w = np.full(len(pairs), 1.0 / len(pairs))
        return self


def stat_S(xi, Y, b, grid: Grid):
    """S_n and the eigenvalues of the plug-in CF variance operator Vhat_phi.

    D = rhohat(s,u) - phihat_X(u + s b) * phihat_eps(s), which is exactly the
    null restriction rho = phi_X(u+sb) phi_eps(s) implied by Y = <b,X> + eps
    with eps independent of X.
    """
    n = len(Y)
    g = grid.build(xi, Y, b)
    s, U, w = g.s, g.U, g.w

    resid = Y - xi @ b                       # = eps exactly under H0
    proj = xi @ U.T                          # (n, N) = <u, X_k>
    argA = np.outer(Y, s) + proj             # s*Y_k + <u,X_k>
    A = np.exp(1j * argA)                    # (n, N)

    Ushift = U + np.outer(s, b)              # u + s b
    B = np.exp(1j * (xi @ Ushift.T))         # (n, N)
    C = np.exp(1j * np.outer(resid, s))      # (n, N)

    Abar, Bbar, Cbar = A.mean(0), B.mean(0), C.mean(0)
    D = Abar - Bbar * Cbar
    S = n * float(np.sum(w * np.abs(D) ** 2))

    # influence function of D (delta method through the product Bbar*Cbar)
    psi = (A - Abar) - Cbar[None, :] * (B - Bbar) - Bbar[None, :] * (C - Cbar)
    sw = np.sqrt(w)[None, :]
    Psi = np.hstack([psi.real * sw, psi.imag * sw])    # (n, 2N)
    Psi -= Psi.mean(0, keepdims=True)
    Vp = Psi.T @ Psi / n
    ev = np.linalg.eigvalsh(Vp)
    return S, np.clip(ev, 0.0, None)


# ------------------------------------------------------------- calibration


def crit(weights, level, rng, ndraw=4000):
    """Critical value of sum_j w_j Z_j^2 from the plug-in spectrum."""
    w = weights[weights > 1e-12]
    if w.size == 0:
        return np.inf
    draws = (rng.chisquare(1.0, (ndraw, w.size)) * w).sum(1)
    return float(np.quantile(draws, 1.0 - level))




def hill(x, k):
    """Hill tail-index estimator on the k largest absolute values."""
    x = np.sort(np.abs(x))[::-1]
    return 1.0 / np.mean(np.log(x[:k] / x[k]))
