"""A selection rule for theta, tied to the hypothesis of Theorem 6' itself.

THE PROBLEM.  Theorem 6' holds at any fixed theta with tr V_theta < inf, and the
simulation says theta = 1 wins at tail index 1.2, theta = 0.5 across 1.5-2.5 and
theta = 0 at 4.0 -- while on the commercial-paper data theta = 0.5 rejects nothing
that theta = 1 rejects at p = 0.0013.  The dial is NOT monotone in usefulness, so
"take theta large" is wrong and "take the smallest valid theta" is wrong too.

THE RULE.  The theorem's condition is tr V_theta = E||psi||^2 < inf for
psi_k = w_theta(X_k) eps_k X_k, ||psi_k||^2 = eps_k^2 ||X_k||^{2-2theta}.  That
condition has a classical observable counterpart: the MAXIMUM-TO-SUM RATIO

    R_n(theta) = max_k ||psi_k||^2 / sum_k ||psi_k||^2

converges to 0 almost surely if and only if E||psi||^2 < infinity (O'Brien 1980;
see Embrechts, Klueppelberg & Mikosch 1997, sec. 6.2.6 for the ratio as a moment
diagnostic).  It needs no tail-index estimate, no bandwidth and no bootstrap: it
is a ratio of two sums the statistic already forms.

    theta-hat = min { theta in grid : R_n(theta) <= kappa }

-- the LEAST reweighting whose variance operator has a second moment to converge
to, which is what the theorem asks for and nothing more.  Small theta is preferred
when it is admissible because it keeps the non-centrality A_theta h closest to Kh.

WHAT MUST BE CHECKED, and is, in run_theta_selector.py:
  1. that kappa can be set once, not per data set;
  2. that the rule holds SIZE -- it is a pre-test, so theta-hat is random and a
     pre-test can distort the level of the test that follows it;
  3. that it recovers most of the ORACLE power (the best fixed theta per cell),
     since a rule that is valid but weak is not worth having.
"""
from __future__ import annotations

import numpy as np

GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def psi_norm2(X, resid, theta):
    """||psi_k||^2 = eps_k^2 ||X_k||^{2-2theta}, with the zero-norm convention."""
    nrm = np.linalg.norm(X, axis=1)
    p = 2.0 - 2.0 * theta
    out = np.zeros(len(resid))
    nz = nrm > 0
    out[nz] = resid[nz] ** 2 * nrm[nz] ** p
    if p <= 0:                       # theta >= 1: ||X||^0 = 1, zeros still dropped
        out[nz] = resid[nz] ** 2
    return out


def max_to_sum(X, resid, theta):
    v = psi_norm2(X, resid, theta)
    s = v.sum()
    return float(v.max() / s) if s > 0 else 1.0


def select(X, resid, kappa, grid=GRID):
    """theta-hat = the least theta on the grid whose max-to-sum ratio is <= kappa.

    Falls back to the largest grid point if none qualifies -- which is the honest
    behaviour: no admissible reweighting was found, so take the most aggressive
    one available and let the size table say whether that was enough.
    """
    for th in grid:
        if max_to_sum(X, resid, th) <= kappa:
            return th, False
    return grid[-1], True
