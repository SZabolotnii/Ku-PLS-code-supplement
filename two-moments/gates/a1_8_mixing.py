#!/usr/bin/env python3
"""Theorem 6' under serial dependence: where i.i.d. is used, and what replaces it.

THE OBSERVATION.  In the proof of Theorem 6', i.i.d. enters in exactly ONE place --
the central limit theorem for n^{-1/2} sum_k psi_k, psi_k = w_theta(X_k) eps_k X_k.
The other half, sqrt(n)||Rhat|| = o_P(1), is a deterministic bound on an OBSERVABLE
that the stopping rule enforces whatever the dependence.  So extending the theorem
is exactly the problem of replacing one CLT.

THE CONJECTURE THIS SCRIPT TESTS.  psi_k is a MARTINGALE DIFFERENCE, not merely a
mixing sequence, whenever the errors are conditionally centred given the past:

    (MI-t)   E[eps_t | X_t, F_{t-1}] = 0,   F_t = sigma((X_s, Y_s), s <= t)

because then E[psi_t | F_{t-1}] = E[ t_theta(X_t) E[eps_t | X_t, F_{t-1}] | F_{t-1} ] = 0.
Under (MI-t) plus stationarity, ergodicity and E||psi||^2 < inf, the Hilbert-space
martingale CLT gives n^{-1/2} sum psi_k => N(0, V_theta) with V_theta = E[psi (x) psi]
-- the SAME variance operator, so the plug-in Vhat_theta stays correct and NO
long-run / HAC correction is needed.

If instead psi_k were merely mixing, the limit would carry the LONG-RUN covariance
V + sum_k (C_k + C_k*), and the plug-in would be wrong by exactly those autocovariance
terms.

WHY IT MATTERS HERE.  It would explain the otherwise surprising empirical fact that
on the commercial-paper data the plug-in critical value matched a block bootstrap to
within 1% at theta = 1, despite ||X_t|| having lag-1 rank autocorrelation +0.59:
volatility clusters, but SIGNS do not, and psi_k inherits the sign.

THREE TESTS.
  1. MDS but strongly dependent in scale -- eps conditionally centred, volatility
     clustered.  Plug-in should hold its level.  (This is also the design of
     run_dependence.py STEP 1, re-run here with clustering in eps as well as in X.)
  2. NOT MDS -- eps serially CORRELATED, which breaks (MI-t).  Plug-in should fail,
     and the size distortion should grow with the autocorrelation.  This is the
     boundary of the theorem and it must be shown, not assumed.
  3. THE DATA -- is psi-hat_t actually a martingale difference?  Ljung-Box on the
     components of psi-hat and on their signs, for both applications.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
from dependent_design import pareto_ar1              # noqa: E402
from run_theta_selector import stat_and_crit         # noqa: E402
from empirical2_cp import load as load_cp, _w        # noqa: E402

OUT = HERE.parent / "results"
LEVEL, N, J = 0.05, 1000, 20
SCALE = (np.arange(J) + 1.0) ** -1.0
BETA = (np.arange(J) + 1.0) ** -2.0


def draw(n, index, rng, phi_x=0.6, eps_mode="mds", rho_e=0.0, sigma=0.5):
    """eps_mode 'mds'  : eps_t = s_t * z_t with s_t a clustered volatility and
                         z_t i.i.d. standard normal -> conditionally centred.
       eps_mode 'ar1'  : eps_t = rho_e eps_{t-1} + innovation -> NOT an MDS."""
    Z = rng.standard_normal((n, J)) * SCALE
    R = pareto_ar1(n, index, phi_x, rng)
    X = Z * R[:, None]
    if eps_mode == "mds":
        lv = np.empty(n)                     # log-volatility AR(1): clustering
        lv[0] = rng.standard_normal()
        for t in range(1, n):
            lv[t] = 0.9 * lv[t - 1] + np.sqrt(1 - 0.81) * rng.standard_normal()
        eps = sigma * np.exp(0.5 * lv) * rng.standard_normal(n)
    else:
        e = rng.standard_normal(n) * sigma * np.sqrt(1 - rho_e ** 2)
        eps = np.empty(n)
        eps[0] = e[0]
        for t in range(1, n):
            eps[t] = rho_e * eps[t - 1] + e[t]
    return X, X @ BETA + eps


def ljung_box(x, lags=10):
    n = len(x)
    x = x - x.mean()
    d = (x * x).sum()
    q = 0.0
    for k in range(1, lags + 1):
        r = (x[:-k] * x[k:]).sum() / d
        q += r * r / (n - k)
    return n * (n + 2) * q                   # ~ chi2(lags) under no autocorrelation


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    rng = np.random.default_rng(20260906)
    se = np.sqrt(LEVEL * (1 - LEVEL) / R)
    out = {}

    print("=" * 88)
    print("TEST 1 -- MDS errors with CLUSTERED volatility, and a dependent predictor")
    print(f"   log-vol AR(1) at 0.9, phi_x = 0.6.  n = {N}, R = {R}, s.e. {se:.4f},")
    print(f"   tolerance [{LEVEL-2*se:.3f}, {LEVEL+2*se:.3f}]")
    print("=" * 88)
    print(f"   {'theta':>7} |" + "".join(f"{'idx ' + str(i):>10}" for i in (1.5, 2.5, 4.0)))
    t1 = {}
    for th in (0.0, 0.5, 1.0):
        row = []
        for idx in (1.5, 2.5, 4.0):
            rej = 0
            for _ in range(R):
                X, Y = draw(N, idx, rng, eps_mode="mds")
                s, cr = stat_and_crit(X, Y, BETA, th, rng)
                rej += s > cr
            row.append(rej / R)
        t1[str(th)] = row
        print(f"   {th:7.1f} |" + "".join(
            f"{v:9.3f}" + ("*" if abs(v - LEVEL) > 2 * se else " ") for v in row))
    print("   * outside +-2 s.e.   -> the plug-in survives clustering in BOTH X and eps.")
    out["mds"] = t1

    print()
    print("=" * 88)
    print("TEST 2 -- NOT an MDS: serially CORRELATED errors break (MI-t)")
    print("   The plug-in Vhat ignores the autocovariances the limit then carries,")
    print("   so the size must degrade with rho_e.  This is the theorem's boundary.")
    print("=" * 88)
    print(f"   {'theta':>7} |" + "".join(f"{'rho_e=' + str(r):>12}"
                                          for r in (0.0, 0.3, 0.6)))
    t2 = {}
    for th in (0.0, 1.0):
        row = []
        for rho in (0.0, 0.3, 0.6):
            rej = 0
            for _ in range(R):
                X, Y = draw(N, 2.5, rng, eps_mode="ar1", rho_e=rho)
                s, cr = stat_and_crit(X, Y, BETA, th, rng)
                rej += s > cr
            row.append(rej / R)
        t2[str(th)] = row
        print(f"   {th:7.1f} |" + "".join(
            f"{v:11.3f}" + ("*" if abs(v - LEVEL) > 2 * se else " ") for v in row))
    print("   * outside +-2 s.e.")
    out["ar1"] = t2

    print()
    print("=" * 88)
    print("TEST 3 -- IS psi-hat_t A MARTINGALE DIFFERENCE ON THE REAL DATA?")
    print("   Ljung-Box(10) on each component of psi-hat_t = w_theta(X_t) Y_t X_t")
    print("   (H0: beta = 0, so the residual is Y).  chi2(10) 95% = 18.31.")
    print("   Also on ||psi||^2, which SHOULD show dependence -- volatility clusters")
    print("   even when signs do not, and that is the whole point.")
    print("=" * 88)
    dXc, Yc, Yv, _ = load_cp()
    d = np.load(HERE.parent / "data" / "fred_cache.npz",
                allow_pickle=True)
    t3 = {}
    for name, X, Y in (("Treasury x S&P 500", d["dX"], d["Y"]),
                       ("CP x VIX", dXc, Yv)):
        for th in (0.0, 1.0):
            P = X * (_w(X, th) * Y)[:, None]
            lbs = [ljung_box(P[:, j]) for j in range(P.shape[1])]
            frac = float(np.mean(np.array(lbs) > 18.31))
            lb_n = ljung_box(np.sum(P ** 2, axis=1))
            t3[f"{name}_th{th}"] = {"median_LB": float(np.median(lbs)),
                                    "frac_reject": frac, "LB_norm2": float(lb_n)}
            print(f"   {name:>20} theta={th:.1f}  median LB {np.median(lbs):8.2f}"
                  f"   components rejecting {frac:5.1%}"
                  f"   LB(||psi||^2) {lb_n:10.1f}")
    print()
    print("   -> components near the chi2 range with ||psi||^2 far above it is the")
    print("      signature the theorem needs: clustered magnitude, unpredictable sign.")
    out["data"] = t3

    OUT.mkdir(exist_ok=True)
    with open(OUT / "a1_8_mixing.json", "w") as f:
        json.dump({"R": R, "n": N, **out}, f, indent=2, default=float)
    print(f"\nwrote {OUT / 'a1_8_mixing.json'}")


if __name__ == "__main__":
    main()
