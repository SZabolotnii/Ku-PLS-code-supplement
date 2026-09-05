#!/usr/bin/env python3
"""Redo of the two tests in a1_8_mixing.py that could not answer their questions.

WHAT WENT WRONG THE FIRST TIME, both times my own design's fault.

  TEST 2 could not show the non-MDS failure.  It made eps serially correlated
  while leaving the DIRECTION of X serially independent.  Then the autocovariance
  of psi is C_k = E[eps_0 eps_k] * E[w w X_0 (x) X_k], and the second factor
  vanishes because E[Z] = 0 with directions independent across t.  Serial
  correlation in eps alone cannot make psi non-MDS.  A genuine violation needs
  persistence in the DIRECTION as well, so that E[t_theta(X_t) | F_{t-1}] != 0.
  Reporting the original as "the plug-in is robust to non-MDS errors" would have
  been the earlier simulation's mistake exactly -- a sweep in a regime where the effect
  it was cited for cannot exist.

  TEST 3 used Ljung-Box on psi-hat, whose chi^2(10) calibration assumes a finite
  FOURTH moment.  With ||X|| at index 1.6 and |Y| at 2.5, psi has none, so the
  statistic is inflated by the tails and says nothing about serial correlation.
  The martingale-difference question is about SIGNS, and the sign process is
  bounded -- so test the signs, where the chi^2 calibration is valid.

REDONE HERE:
  A. non-MDS by construction: direction AR(1) at rho_d AND error AR(1) at rho_e,
     so C_1 != 0 and the plug-in must degrade.  If it still does not, the MDS
     route is not what is keeping the test calibrated and the conjecture is wrong.
  B. sign-based Ljung-Box on the real data, plus the same on ||psi||^2 for
     contrast: the theorem needs clustered magnitude with unpredictable sign.
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
from a1_8_mixing import ljung_box                    # noqa: E402

OUT = HERE.parent / "results"
LEVEL, N, J = 0.05, 1000, 20
SCALE = (np.arange(J) + 1.0) ** -1.0
BETA = (np.arange(J) + 1.0) ** -2.0


def draw_nonmds(n, index, rng, rho_d, rho_e, sigma=0.5):
    """Persistent DIRECTION and persistent error -- the combination is what makes
    psi_t = w(X_t) eps_t X_t fail to be a martingale difference."""
    eta = rng.standard_normal((n, J))
    Z = np.empty((n, J))
    Z[0] = eta[0]
    s = np.sqrt(1 - rho_d ** 2)
    for t in range(1, n):
        Z[t] = rho_d * Z[t - 1] + s * eta[t]
    Z *= SCALE
    R = pareto_ar1(n, index, 0.6, rng)
    X = Z * R[:, None]
    e = rng.standard_normal(n) * sigma * np.sqrt(1 - rho_e ** 2)
    eps = np.empty(n)
    eps[0] = e[0]
    for t in range(1, n):
        eps[t] = rho_e * eps[t - 1] + e[t]
    return X, X @ BETA + eps


def psi_autocorr1(X, eps, theta):
    """Lag-1 autocorrelation of psi, averaged over components -- the quantity the
    long-run covariance would carry and the plug-in ignores."""
    P = X * (_w(X, theta) * eps)[:, None]
    P = P - P.mean(0)
    num = (P[:-1] * P[1:]).sum(0)
    den = (P * P).sum(0)
    return float(np.mean(num / den))


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 700
    rng = np.random.default_rng(20260907)
    se = np.sqrt(LEVEL * (1 - LEVEL) / R)
    out = {}

    print("=" * 90)
    print("A -- NON-MDS BY CONSTRUCTION: persistent direction AND persistent error")
    print(f"    n = {N}, R = {R}, index 2.5, s.e. {se:.4f}, "
          f"tolerance [{LEVEL-2*se:.3f}, {LEVEL+2*se:.3f}]")
    print("    'psi acf1' is the realised lag-1 autocorrelation of psi -- the term")
    print("    the plug-in Vhat omits.  If it is ~0 the design still cannot answer.")
    print("=" * 90)
    print(f"   {'rho_d':>7} {'rho_e':>7} |" + "".join(
        f"{'size th=' + str(t):>14}" for t in (0.0, 1.0)) + f"{'psi acf1':>11}")
    rowsA = []
    for rho_d, rho_e in ((0.0, 0.0), (0.6, 0.0), (0.0, 0.6),
                         (0.6, 0.6), (0.85, 0.85)):
        sizes, acf = [], []
        for th in (0.0, 1.0):
            rej = 0
            for _ in range(R):
                X, Y = draw_nonmds(N, 2.5, rng, rho_d, rho_e)
                s, cr = stat_and_crit(X, Y, BETA, th, rng)
                rej += s > cr
                if th == 1.0 and len(acf) < 200:
                    acf.append(psi_autocorr1(X, Y - X @ BETA, 1.0))
            sizes.append(rej / R)
        a = float(np.mean(acf))
        rowsA.append({"rho_d": rho_d, "rho_e": rho_e, "size": sizes, "acf1": a})
        print(f"   {rho_d:7.2f} {rho_e:7.2f} |" + "".join(
            f"{v:13.3f}" + ("*" if abs(v - LEVEL) > 2 * se else " ") for v in sizes)
            + f"{a:11.4f}")
    print("   * outside +-2 s.e.")
    out["nonmds"] = rowsA

    print()
    print("=" * 90)
    print("B -- IS THE SIGN OF psi PREDICTABLE ON THE REAL DATA?")
    print("    Ljung-Box(10) on sign(psi_j,t): the sign process is BOUNDED, so the")
    print("    chi^2(10) calibration is valid where it was not for psi itself.")
    print("    chi^2(10) 95% = 18.31.  Contrast with ||psi||^2, which should and does")
    print("    show dependence -- magnitude clusters, and that is not the issue.")
    print("=" * 90)
    dXc, Yc, Yv, _ = load_cp()
    d = np.load(HERE.parent / "data" / "fred_cache.npz",
                allow_pickle=True)
    rowsB = {}
    print(f"   {'application':>22} {'theta':>6} {'median LB(sign)':>17} "
          f"{'frac rej':>10} {'LB(||psi||^2)':>15}")
    for name, X, Y in (("Treasury x S&P 500", d["dX"], d["Y"]),
                       ("CP x VIX", dXc, Yv)):
        for th in (0.0, 1.0):
            P = X * (_w(X, th) * Y)[:, None]
            lbs = np.array([ljung_box(np.sign(P[:, j])) for j in range(P.shape[1])])
            lb_n = ljung_box(np.sum(P ** 2, axis=1))
            rowsB[f"{name}_th{th}"] = {"median_LB_sign": float(np.median(lbs)),
                                       "frac_reject": float((lbs > 18.31).mean()),
                                       "LB_norm2": float(lb_n)}
            print(f"   {name:>22} {th:6.1f} {np.median(lbs):17.2f} "
                  f"{(lbs > 18.31).mean():9.1%} {lb_n:15.1f}")
    out["signs"] = rowsB

    OUT.mkdir(exist_ok=True)
    with open(OUT / "a1_8b_mds_boundary.json", "w") as f:
        json.dump({"R": R, "n": N, **out}, f, indent=2, default=float)
    print(f"\nwrote {OUT / 'a1_8b_mds_boundary.json'}")


if __name__ == "__main__":
    main()
