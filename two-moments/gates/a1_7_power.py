#!/usr/bin/env python3
"""A1.7 --- where the payoff actually is: POWER below tail index 2.

Two corrections to this repo's own earlier numbers drive this script.

CORRECTION 1.  `code/twomoments.py::wald_diagnostic` builds Vhat from the
ESTIMATED residuals Y - <betahat_m, X>.  Under a simple null the exact residuals
Y - <b, X> are available and are the errors themselves (Theorem 6' section 2), so
the estimated-residual version is a strictly worse estimator of V that carries the
regularisation bias into the weights.  The A2 pilot's headline -- BCT at size
0.092 / 0.064 at tail index 2.5 / 3.0 -- was produced by THAT estimator, not by a
failure of the fourth moment.  With the null-imposed residuals BCT is correctly
sized throughout the (2,4) window.  **That earlier number is withdrawn.**

CORRECTION 2.  The failure of BCT's test below index 2 is not over-rejection, it
is severe CONSERVATISM: size 0.015 against a nominal 0.05 at index 1.2, because
the plug-in weights diverge with the statistic and the critical value outruns it.
tr Vhat for theta = 0 reads 97.0 / 14.4 / 4.70 / 1.57 across index 1.2 / 1.5 / 1.8
/ 2.5 -- growing without settling, the signature of an infinite population trace
(tr V_0 = sigma^2 E||X||^2).  For theta = 1 it reads 0.249 / 0.251 / 0.250 / 0.249,
i.e. sigma^2 = 0.25 exactly, as tr V_1 = E eps^2 predicts.

A conservative test is not a safe test; it is a test that has quietly lost its
power.  So the payoff of theta is a POWER claim, and this script measures it.

DESIGN.  H1: beta = b + h/sqrt(n), h = c*v for a fixed direction v.  The scale c
is CALIBRATED so that the reference (theta = 0) test has power near 0.5 at the
lightest tail index, and then held fixed across the sweep -- otherwise every cell
reads 1.000 or 0.050 and the table says nothing (which is what a first attempt
produced).  Power is reported BOTH raw (the honest operating characteristic of a
test used with its own plug-in critical value) and SIZE-CORRECTED (the empirical
null quantile), because a conservative test's raw power understates its
discriminating ability and an over-sized one's overstates it.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "code"))
from twomoments import Design, operator, cg  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
SEED = 20260824
THETAS = [0.0, 0.5, 1.0, 2.0]
INDICES = [1.2, 1.5, 1.8, 2.5, 4.0]
LEVEL = 0.05
N = 1000


def run_stat(X, Y, b, theta, a_n, m_max=40):
    n = len(Y)
    A, r = operator(X, Y, theta)
    for m in range(1, m_max + 1):
        bh = cg(A, r, m)
        if np.linalg.norm(r - A @ bh) <= a_n:
            break
    stat = n * float(np.sum((A @ (bh - b)) ** 2))
    w = np.linalg.norm(X, axis=1) ** (-theta) if theta else np.ones(n)
    Z = X * (w * (Y - X @ b))[:, None]        # exact errors under H0
    ev = np.clip(np.linalg.eigvalsh(Z.T @ Z / n), 0.0, None)
    return stat, ev[ev > 1e-14]


def crit(ev, rng, ndraw=3000):
    if ev.size == 0:
        return np.inf
    return float(np.quantile((rng.chisquare(1.0, (ndraw, ev.size)) * ev).sum(1),
                             1.0 - LEVEL))


def cell(design, index, theta, c, v, R, rng, a_n):
    b = design.beta
    h = c * v
    s0, s1, rej0, rej1 = [], [], 0, 0
    for _ in range(R):
        X, Y = design.draw(N, index, rng)          # H0
        st, ev = run_stat(X, Y, b, theta, a_n)
        s0.append(st); rej0 += st > crit(ev, rng)
        X, Y = design.draw(N, index, rng)          # H1, local
        Y = Y + X @ h / np.sqrt(N)
        st, ev = run_stat(X, Y, b, theta, a_n)
        s1.append(st); rej1 += st > crit(ev, rng)
    c0 = float(np.quantile(s0, 1 - LEVEL))
    return dict(size=rej0 / R, power_raw=rej1 / R,
                power_sc=float(np.mean(np.array(s1) > c0)))


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    rng = np.random.default_rng(SEED)
    design = Design(J=20, decay=1.0)
    a_n = 1.0 / (np.sqrt(N) * np.log(N))
    se = np.sqrt(LEVEL * (1 - LEVEL) / R)

    # direction: beta itself, normalised -- the economically meaningful departure
    v = design.beta / np.linalg.norm(design.beta)

    # ---- calibrate c once, at the LIGHTEST tail, against the reference test
    print("calibrating the local departure so the table is informative ...", flush=True)
    lo, hi = 0.5, 40.0
    for _ in range(7):
        mid = np.sqrt(lo * hi)
        p = cell(design, 4.0, 0.0, mid, v, 150, rng, a_n)["power_sc"]
        print(f"   c = {mid:6.2f} -> theta=0 size-corrected power at index 4.0 = {p:.3f}",
              flush=True)
        if p < 0.5:
            lo = mid
        else:
            hi = mid
    c = np.sqrt(lo * hi)
    print(f"   fixed c = {c:.3f}\n")

    print("=" * 96)
    print(f"LOCAL POWER, H1: beta = b + c*v/sqrt(n),  c = {c:.2f}, n = {N}, R = {R}, "
          f"nominal {LEVEL:.0%}")
    print(f"s.e. on size = {se:.4f}; tolerance [{LEVEL-2*se:.3f}, {LEVEL+2*se:.3f}]")
    print("=" * 96)
    res = {}
    for label, key in (("size", "size"), ("power, raw", "power_raw"),
                       ("power, size-corrected", "power_sc")):
        print(f"\n   {label}")
        print(f"   {'theta':>9} |" + "".join(f"{'idx ' + str(i):>10}" for i in INDICES))
        print("   " + "-" * (11 + 10 * len(INDICES)))
        for th in THETAS:
            row = []
            for index in INDICES:
                if (th, index) not in res:
                    res[(th, index)] = cell(design, index, th, c, v, R, rng, a_n)
                row.append(res[(th, index)][key])
            tag = "(BCT)" if th == 0 else "(sgn)" if th == 2 else "     "
            if key == "size":
                mk = "".join(f"{x:9.3f}" + ("*" if abs(x - LEVEL) > 2 * se else " ")
                             for x in row)
            else:
                mk = "".join(f"{x:10.3f}" for x in row)
            print(f"   {th:6.1f}{tag}|{mk}")
    print("\n   * = size outside +-2 s.e. of nominal")
    print()
    print("   READ: the size row says whether a test is USABLE at that tail index.")
    print("   The raw-power row is what a practitioner actually gets from it.  The")
    print("   size-corrected row separates 'lost power because it is conservative'")
    print("   from 'lost power because the statistic discriminates worse'.")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "a1_7_power.json", "w") as f:
        json.dump({"R": R, "c": float(c), "n": N, "level": LEVEL,
                   "rows": {f"th{t}_idx{i}": v for (t, i), v in res.items()}},
                  f, indent=2, default=float)
    print(f"\nwrote {OUT / 'a1_7_power.json'}")


if __name__ == "__main__":
    main()
