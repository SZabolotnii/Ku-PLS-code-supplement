"""Route A, revised --- functional PLS on the reweighted operator A_theta.

    A_theta = E[ ||X||^{-theta} X (x) X ],   r_theta = E[ Y X / ||X||^theta ]
    A_theta beta = r_theta                    exactly, under (MI) with E|eps| < inf

theta = 0 recovers Babii-Carrasco-Tsafack exactly, so the comparison is internal
to one code path and one CG routine: nothing differs between the baseline and the
proposal except the scalar theta.  That is deliberate.  The paper-1 replication package put
the baseline and the proposal in two separate functions with two separate
calibrations, and it made the two columns hard to trust.

WHAT IS NEW HERE RELATIVE TO ../replication/kupls/core.py, which is left alone
as the record of what the refuted version produced:

 1. The operator is the reweighted covariance, not the ECF kernel K_phi.  There
    is no bridge identity: the operator inverted IS the operator estimated.
 2. The predictor is drawn with a CONTROLLED TAIL INDEX via a Pareto radial part,
    so the honest comparison axis --- index in (2,4) where BCT's fourth moment is
    absent, AND index > 4 where it is present --- can actually be swept.  The paper-1
    simulation swept alpha-stable alpha in [1.2, 2.0], i.e. tail index BELOW 2,
    a regime in which no member of this family has a sqrt(n) CLT either.  That
    sweep could not have shown the effect the paper claimed.
 3. Reported is estimation error, which is proved (Lemmas 1-4 of
    theory/A0-operator-theory.md).  The test is reported as a DIAGNOSTIC and
    labelled as such: the delta method through the CG regulariser with a
    data-driven stopping rule is open (A1.7) and is not announced here.
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------- the design

class Design:
    """Functional predictor in coefficient coordinates, with a controlled tail.

    X = R * Z,  Z ~ N(0, diag(scale^2)),  R ~ Pareto(index) + 1.

    Then E||X||^p < inf iff p < index, exactly, and the eigenvalue decay of the
    covariance is set separately by `decay`.  Separating the two is the point:
    the tail index is the paper's axis and the decay is the ill-posedness.
    """

    def __init__(self, J: int = 20, decay: float = 1.0, sigma: float = 0.5):
        self.J, self.decay, self.sigma = J, decay, sigma
        self.scale = (np.arange(J) + 1.0) ** -decay
        self.beta = (np.arange(J) + 1.0) ** -2.0

    def draw(self, n, index, rng, eps_law="gauss"):
        Z = rng.standard_normal((n, self.J)) * self.scale
        R = rng.pareto(index, n) + 1.0
        X = Z * R[:, None]
        if eps_law == "gauss":
            eps = rng.normal(0.0, self.sigma, n)
        elif eps_law == "t3":                      # E eps^2 < inf, E eps^4 = inf
            eps = self.sigma * rng.standard_t(3.0, n)
        elif eps_law == "cauchy":                  # E|eps| = inf: (MI) undefined
            eps = self.sigma * np.tan(rng.uniform(-np.pi / 2, np.pi / 2, n))
        else:
            raise ValueError(eps_law)
        return X, X @ self.beta + eps


# ------------------------------------------------------------- the estimator

def operator(X, Y, theta):
    """(A_theta, r_theta) --- the empirical pair.  theta = 0 is BCT's (K, r).

    THE ZERO-NORM CONVENTION, and why it is not a detail.  The theory in
    theory/A0-operator-theory.md assumes P(X = 0) = 0, and the money-market
    application violates it on the first contact: the commercial-paper curve is
    quoted to two decimals and is UNCHANGED on a nontrivial share of days, so
    dX_t = 0 exactly.  The spatial sign of the zero vector does not exist.

    Convention adopted: t_theta(0) := 0, i.e. an observation with X_k = 0
    contributes nothing.  It is the continuous extension for theta < 1 (since
    ||t_theta(x)|| = ||x||^{1-theta} -> 0), it is the standard convention in the
    spatial-sign literature at theta = 1, and it is what the estimand then means:
    A_theta and r_theta are expectations over the conditional law given X != 0,
    and identification is on that law.  At theta = 0 it changes nothing, because
    a zero observation already contributes zero.

    This must be DECLARED in the paper, not implemented silently: it drops
    observations, and the count belongs in the data section.
    """
    n = len(Y)
    if theta:
        nrm = np.linalg.norm(X, axis=1)
        w = np.zeros(n)
        nz = nrm > 0
        w[nz] = nrm[nz] ** (-theta)
    else:
        w = np.ones(n)
    return (X * w[:, None]).T @ X / n, (X * (w * Y)[:, None]).sum(0) / n


def cg(A, r, m):
    """m steps of conjugate gradient on A a = r from a_0 = 0.

    This is functional PLS in the BCT reframing and is byte-for-byte the routine
    of lean/replication/kupls/core.py::cg_pls, so no comparison in this file can
    be an artefact of a different solver.
    """
    a = np.zeros_like(r)
    res = r.copy()
    d = res.copy()
    for _ in range(m):
        Ad = A @ d
        den = d @ Ad
        if den <= 0 or not np.isfinite(den):
            break
        step = (res @ res) / den
        a = a + step * d
        new = res - step * Ad
        if not np.isfinite(new).all():
            break
        rr = res @ res
        if rr <= 0:                      # exact convergence; further steps are 0/0
            return a
        res, d = new, new + (new @ new) / rr * d
    return a


def fit(X, Y, theta, m):
    A, r = operator(X, Y, theta)
    return cg(A, r, m)


def discrepancy_stop_operator_scale(X, Y, theta, m_max=15, kappa=1.0):
    """SUPERSEDED.  Kept only so the numbers it produced remain reproducible.

    Stops when ||rhat - Ahat betahat_m|| <= kappa * sqrt(tr Ahat_theta / n), i.e.
    at the scale of the OPERATOR, tr A_theta = E||X||^{2-theta}.  The discrepancy
    principle is supposed to stop at the noise level of the DATA.  See gates/
    a1_6_discrepancy.py and theory/A0-operator-theory.md section 9.4.  Every
    number in this repository dated before that section was produced by this rule;
    do not delete it, and do not use it for anything new.
    """
    A, r = operator(X, Y, theta)
    n = len(Y)
    tau = kappa * np.sqrt(np.trace(A) / n)
    a = np.zeros_like(r)
    res = r.copy()
    d = res.copy()
    for m in range(1, m_max + 1):
        Ad = A @ d
        den = d @ Ad
        if den <= 0 or not np.isfinite(den):
            return a, m - 1
        step = (res @ res) / den
        a = a + step * d
        new = res - step * Ad
        if not np.isfinite(new).all():
            return a, m - 1
        rr = res @ res
        if rr <= 0:
            return a, m
        res, d = new, new + (new @ new) / rr * d
        if np.linalg.norm(res) <= tau:
            return a, m
    return a, m_max


def discrepancy_stop(X, Y, theta, tau=1.0, m_max=25):
    """THE estimation stopping rule: discrepancy at the DATA noise level.

        mhat = min{ m : ||rhat - Ahat betahat_m|| <= tau * sqrt(tr Vhat_theta(m)/n) }
        tr Vhat_theta(m) = n^{-1} sum_k (Y_k - <betahat_m, X_k>)^2 ||X_k||^{2-2theta}

    tr V_theta = E[eps^2 ||X||^{2-2theta}] is the trace of the SAME operator whose
    eigenvalues are the weights in Theorem 6', so estimation and inference are
    governed by one moment condition rather than two, and the rule is estimable
    online from the current residuals.  The scope condition tr V_theta < inf is
    exactly the one Theorem 6' needs and the theta-selector enforces; where it
    fails (theta = 0 below tail index 2) there is nothing to estimate the
    threshold from, and that is the correct behaviour, not a defect.

    The zero-norm convention applies to the weight ||X_k||^{2-2theta} as well:
    at theta > 1 the exponent is negative and X_k = 0 contributes 0.
    """
    n = len(Y)
    A, r = operator(X, Y, theta)
    p = 2.0 - 2.0 * theta
    if p == 0:
        wv = np.ones(n)
    else:
        nrm = np.linalg.norm(X, axis=1)
        nz = nrm > 0
        wv = np.zeros(n)
        wv[nz] = nrm[nz] ** p

    a = np.zeros_like(r)
    res = r.copy()
    d = res.copy()
    for m in range(1, m_max + 1):
        Ad = A @ d
        den = d @ Ad
        if den <= 0 or not np.isfinite(den):
            return a, m - 1
        step = (res @ res) / den
        a = a + step * d
        new = res - step * Ad
        if not np.isfinite(new).all():
            return a, m - 1
        rr = res @ res
        if rr <= 0:
            return a, m
        res, d = new, new + (new @ new) / rr * d
        e = Y - X @ a
        trV = float(np.mean(e ** 2 * wv))
        if np.linalg.norm(res) <= tau * np.sqrt(trV / n):
            return a, m
    return a, m_max


# ------------------------------------------------- diagnostic test (NOT proved)

def wald_diagnostic(X, Y, b, theta, m, rng, level=0.05, ndraw=3000):
    """n ||Ahat(betahat_m - b)||^2 against a plug-in weighted-chi^2.

    LABELLED A DIAGNOSTIC ON PURPOSE.  The plug-in spectrum is that of
    Vhat_theta = Cov(w(X) eps X); the delta method through the CG regulariser
    with a data-driven m is NOT proved (plan A1.7).  This function measures what
    the plug-in calibration does; it does not claim a limit law.
    """
    A, r = operator(X, Y, theta)
    a = cg(A, r, m)
    stat = len(Y) * float(np.sum((A @ (a - b)) ** 2))
    w = np.linalg.norm(X, axis=1) ** (-theta) if theta else np.ones(len(Y))
    Zt = X * (w * (Y - X @ a))[:, None]
    V = Zt.T @ Zt / len(Y)
    ev = np.clip(np.linalg.eigvalsh(V), 0.0, None)
    ev = ev[ev > 1e-12]
    if ev.size == 0:
        return stat, np.inf
    draws = (rng.chisquare(1.0, (ndraw, ev.size)) * ev).sum(1)
    return stat, float(np.quantile(draws, 1.0 - level))
