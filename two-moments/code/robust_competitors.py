#!/usr/bin/env python3
"""Robust competitors on the design of section 7 --- pre-registered.

The protocol is ROBUST-COMPARISON-PREREG-2026-11-18.md, written before any run;
this script implements it and nothing it does not fix, with ONE addition the
protocol did not foresee and which is labelled wherever it appears:

  The protocol names `wald_diagnostic` at m = 3 as the reference test.  That
  function builds Vhat from ESTIMATED residuals, the variant gates/a1_7_power.py
  ("CORRECTION 1") withdrew; the paper's tab:power uses the overfitting rule (O)
  with a_n = 1/(sqrt(n) log n) and the null-imposed residuals Y - <b, X>
  (gates/a1_7_power.py::run_stat).  Both are reported, on the SAME draws, so the
  registered analysis is kept and the paper's own procedure sits beside it.

E1  estimation.  Median relative slope error ||betahat - beta|| / ||beta||,
    n = 1000, R = 200, tail index in {1.2, 1.5, 1.8, 2.5, 3.0}.  Reweighted CG at
    theta in {0, 1, 2} and PRM-FPLS, each at its own oracle-best m in 1..10.

E2  the test.  Size at nominal 5% (R = 1000) and raw power under the local
    alternative of tab:power (h = c beta/||beta||, c = 0.716), n = 1000, tail
    index in {1.2, 1.5, 1.8, 3.0}.  Competitor: robust Wald test on a truncated
    principal-component expansion (the structure of Zhang et al. 2023): Huber
    M-regression (k = 1.345, MAD scale) of Y - <X, b> on the first K = 3
    empirical principal-component scores, sandwich covariance, chi^2_3.

RE-IMPLEMENTATIONS OF STRUCTURE, NOT THE AUTHORS' CODE.  PRM follows Serneels,
Croux, Filzmoser & Van Espen (2005): iteratively reweighted PLS1 with Fair
weights (c = 4) on MAD-standardised residuals and on score-space distances from
the coordinatewise median.  The model has no intercept and the design is
symmetric about zero, so neither PRM nor the Huber regression centres or fits
an intercept --- the same convention as the CG estimators it is compared with.

Data and critical-value draws use separate generators, so every method sees
exactly the same (X, Y) in every replication.

    python code/robust_competitors.py            # full protocol
    python code/robust_competitors.py --quick    # R/10, smoke test only
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gates"))
from twomoments import Design, fit, wald_diagnostic  # noqa: E402
from a1_7_power import run_stat, crit  # noqa: E402

OUT = HERE.parent / "results"
SEED_DATA = 20261118
SEED_CRIT = 20261119
N = 1000
C = 0.716
LEVEL = 0.05
M_GRID = range(1, 11)


# ------------------------------------------------------------------ PRM-FPLS

def _fair(z, c=4.0):
    return 1.0 / (1.0 + np.abs(z / c)) ** 2


def _mad(v):
    s = 1.4826 * np.median(np.abs(v - np.median(v)))
    return s if s > 0 else 1.0


def _pls1(X, y, m):
    """NIPALS PLS1 without centring; returns the coefficient vector for m comps."""
    Xr, yr = X.copy(), y.copy()
    W, P, q = [], [], []
    for _ in range(m):
        w = Xr.T @ yr
        nw = np.linalg.norm(w)
        if nw == 0:
            break
        w /= nw
        t = Xr @ w
        tt = t @ t
        if tt == 0:
            break
        p = Xr.T @ t / tt
        qq = yr @ t / tt
        Xr = Xr - np.outer(t, p)
        yr = yr - qq * t
        W.append(w); P.append(p); q.append(qq)
    if not W:
        return np.zeros(X.shape[1]), np.zeros((X.shape[1], 0))
    W, P, q = np.array(W).T, np.array(P).T, np.array(q)
    Rm = W @ np.linalg.inv(P.T @ W)          # X @ Rm = scores
    return Rm @ q, Rm


def prm(X, y, m, max_iter=30, tol=1e-2):
    """Partial robust M-regression (Serneels et al. 2005), Fair weights."""
    d = np.linalg.norm(X - np.median(X, axis=0), axis=1)
    wr = _fair((y - np.median(y)) / _mad(y))
    wx = _fair(d / np.median(d))
    w = wr * wx
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        sw = np.sqrt(w)
        new, Rm = _pls1(X * sw[:, None], y * sw, m)
        res = y - X @ new
        T = X @ Rm
        dt = np.linalg.norm(T - np.median(T, axis=0), axis=1)
        med = np.median(dt)
        wr = _fair(res / _mad(res))
        wt = _fair(dt / med) if med > 0 else np.ones_like(dt)
        w = np.clip(wr * wt, 1e-8, None)
        step = np.linalg.norm(new - beta) / max(np.linalg.norm(new), 1e-12)
        beta = new
        if step < tol:
            break
    return beta


# ------------------------------------------------------- robust Wald (Zhang)

def huber_wald(X, Y, b, K=3, k=1.345, max_iter=50):
    """Huber regression of Y - <X,b> on K PC scores; sandwich Wald vs chi^2_K."""
    n = len(Y)
    S = X.T @ X / n
    vals, vecs = np.linalg.eigh(S)
    Z = X @ vecs[:, ::-1][:, :K]             # first K principal-component scores
    e = Y - X @ b
    g = np.linalg.lstsq(Z, e, rcond=None)[0]
    for _ in range(max_iter):
        r = e - Z @ g
        s = _mad(r)
        u = r / s
        wts = np.where(np.abs(u) <= k, 1.0, k / np.abs(u))
        gn = np.linalg.solve(Z.T @ (Z * wts[:, None]), Z.T @ (wts * e))
        if np.linalg.norm(gn - g) <= 1e-8 * (1 + np.linalg.norm(g)):
            g = gn
            break
        g = gn
    r = e - Z @ g
    s = _mad(r)
    u = r / s
    psi = np.clip(u, -k, k)
    dpsi = (np.abs(u) <= k).astype(float)
    A = (Z * dpsi[:, None]).T @ Z / (n * s)
    B = (Z * (psi ** 2)[:, None]).T @ Z / n
    Ai = np.linalg.inv(A)
    V = Ai @ B @ Ai / n
    return float(g @ np.linalg.solve(V, g))


CHI2_3_95 = 7.814727903251178            # scipy.stats.chi2.ppf(0.95, 3)


# ------------------------------------------------------------------- E1, E2

def e1(design, R, rng):
    out = {}
    for index in (1.2, 1.5, 1.8, 2.5, 3.0):
        errs = {k: [] for k in ("theta0", "theta1", "theta2", "prm")}
        nb = np.linalg.norm(design.beta)
        for _ in range(R):
            X, Y = design.draw(N, index, rng)
            for th, key in ((0.0, "theta0"), (1.0, "theta1"), (2.0, "theta2")):
                errs[key].append(min(np.linalg.norm(fit(X, Y, th, m) - design.beta)
                                     for m in M_GRID) / nb)
            errs["prm"].append(min(np.linalg.norm(prm(X, Y, m) - design.beta)
                                   for m in M_GRID) / nb)
        out[str(index)] = {k: float(np.median(v)) for k, v in errs.items()}
        print(f"E1 index {index:3.1f}  " + "  ".join(
            f"{k}={v:.4f}" for k, v in out[str(index)].items()), flush=True)
    return out


def e2(design, R, rng, crng):
    b = design.beta
    h = C * b / np.linalg.norm(b)
    a_n = 1.0 / (np.sqrt(N) * np.log(N))
    keys = ("robust_wald", "O_theta0", "O_theta1", "m3_theta0", "m3_theta1")
    out = {}
    for index in (1.2, 1.5, 1.8, 3.0):
        rej = {"H0": dict.fromkeys(keys, 0), "H1": dict.fromkeys(keys, 0)}
        for _ in range(R):
            for hyp in ("H0", "H1"):
                X, Y = design.draw(N, index, rng)
                if hyp == "H1":
                    Y = Y + X @ h / np.sqrt(N)
                rej[hyp]["robust_wald"] += huber_wald(X, Y, b) > CHI2_3_95
                for th, tag in ((0.0, "theta0"), (1.0, "theta1")):
                    st, ev = run_stat(X, Y, b, th, a_n)
                    rej[hyp][f"O_{tag}"] += st > crit(ev, crng)
                    st, cv = wald_diagnostic(X, Y, b, th, 3, crng)
                    rej[hyp][f"m3_{tag}"] += st > cv
        out[str(index)] = {k: {"size": rej["H0"][k] / R, "power": rej["H1"][k] / R}
                           for k in keys}
        print(f"E2 index {index:3.1f}  " + "  ".join(
            f"{k}: {v['size']:.3f}/{v['power']:.3f}"
            for k, v in out[str(index)].items()), flush=True)
    return out


def calibration_check(design, R, rng):
    """ADDED AFTER THE RUN, NOT PRE-REGISTERED: is the robust Wald code itself
    calibrated where its derivation applies?  Light-tailed predictors (tail index
    50 and 8 in the same design) and a Gaussian predictor with the same scales.
    A size near 5% here locates any distortion in E2 in the predictor's tail, not
    in the implementation."""
    out = {}
    for label in ("index50", "index8", "gaussian"):
        rej = 0
        for _ in range(R):
            if label == "gaussian":
                X = rng.standard_normal((N, design.J)) * design.scale
                Y = X @ design.beta + rng.normal(0.0, design.sigma, N)
            else:
                X, Y = design.draw(N, 50.0 if label == "index50" else 8.0, rng)
            rej += huber_wald(X, Y, design.beta) > CHI2_3_95
        out[label] = rej / R
    print("calibration check (robust Wald size, H0): " +
          "  ".join(f"{k}={v:.3f}" for k, v in out.items()), flush=True)
    return out


def main():
    quick = "--quick" in sys.argv
    R1, R2 = (20, 100) if quick else (200, 1000)
    design = Design(J=20, decay=1.0)
    rng = np.random.default_rng(SEED_DATA)
    crng = np.random.default_rng(SEED_CRIT)
    t0 = time.time()
    res = {"protocol": "ROBUST-COMPARISON-PREREG-2026-11-18.md", "n": N,
           "R_E1": R1, "R_E2": R2, "c": C, "level": LEVEL,
           "se_E2": float(np.sqrt(LEVEL * (1 - LEVEL) / R2)),
           "seeds": [SEED_DATA, SEED_CRIT],
           "E1_median_rel_error_oracle_m": e1(design, R1, rng),
           "E2_size_power": e2(design, R2, rng, crng)}
    res["calibration_check_not_preregistered"] = calibration_check(
        design, 40 if quick else 1000, np.random.default_rng(20261120))
    res["seconds"] = round(time.time() - t0, 1)
    name = "robust_competitors_quick.json" if quick else "robust_competitors.json"
    OUT.mkdir(exist_ok=True)
    with open(OUT / name, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote results/{name}  ({res['seconds']} s)")   # relative: no local path in shipped output


if __name__ == "__main__":
    main()
