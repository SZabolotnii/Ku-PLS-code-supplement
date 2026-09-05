"""A dependent design with the marginal tail index held EXACTLY fixed.

The selector's kappa was calibrated on i.i.d. draws and proved too loose on
serially dependent series (EMPIRICAL-FINDINGS.md section 8).  Recalibrating needs
a design in which dependence can be dialled without moving the marginal tail,
otherwise a change in kappa cannot be attributed to either.

CONSTRUCTION.  A Gaussian AR(1) is mapped through its own CDF to uniforms and
then through the inverse Pareto CDF:

    G_t = phi G_{t-1} + sqrt(1-phi^2) e_t,   e_t ~ N(0,1)   (stationary, unit var)
    U_t = Phi(G_t)                                          (exactly Uniform(0,1))
    R_t = (1 - U_t)^{-1/index}                              (exactly Pareto(index))

so the marginal law of R_t is the SAME Pareto used by the i.i.d. design, for every
phi.  Only the serial dependence changes.  The predictor is X_t = R_t * Z_t with
Z_t an independent Gaussian curve, so ||X_t|| inherits both the Pareto tail and
the persistence.

PERSISTENCE IS MEASURED FROM THE DATA, not chosen.  The matching statistic is the
lag-1 SPEARMAN autocorrelation of ||X_t||, which is invariant to the marginal and
therefore comparable between a real series and this design.  phi is set so the
design reproduces it.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata


def spearman_lag1(x):
    r = rankdata(x)
    return float(np.corrcoef(r[:-1], r[1:])[0, 1])


def pareto_ar1(n, index, phi, rng):
    """Pareto(index) marginal exactly, Gaussian-copula AR(1) dependence."""
    if phi == 0.0:
        U = rng.uniform(size=n)
    else:
        e = rng.standard_normal(n)
        G = np.empty(n)
        G[0] = e[0]
        s = np.sqrt(1 - phi ** 2)
        for t in range(1, n):
            G[t] = phi * G[t - 1] + s * e[t]
        U = norm.cdf(G)
    return (1.0 - U) ** (-1.0 / index)


def phi_for_target(target_rho, index, n, rng, grid=np.linspace(0, 0.95, 20)):
    """The phi whose design reproduces a measured lag-1 Spearman autocorrelation."""
    best, bestd = 0.0, np.inf
    for phi in grid:
        rhos = [spearman_lag1(pareto_ar1(min(n, 4000), index, phi, rng))
                for _ in range(6)]
        d = abs(np.mean(rhos) - target_rho)
        if d < bestd:
            best, bestd = float(phi), d
    return best


class DependentDesign:
    """Same J, scale, beta and error law as twomoments.Design; dependent radial."""

    def __init__(self, J=20, decay=1.0, sigma=0.5, phi=0.0):
        self.J, self.sigma, self.phi = J, sigma, phi
        self.scale = (np.arange(J) + 1.0) ** -decay
        self.beta = (np.arange(J) + 1.0) ** -2.0

    def draw(self, n, index, rng, eps_law="gauss"):
        Z = rng.standard_normal((n, self.J)) * self.scale
        R = pareto_ar1(n, index, self.phi, rng) 
        X = Z * R[:, None]
        eps = rng.normal(0.0, self.sigma, n)
        return X, X @ self.beta + eps
