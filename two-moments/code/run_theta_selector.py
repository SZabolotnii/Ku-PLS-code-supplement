#!/usr/bin/env python3
"""Does the theta selector work?  Calibrate kappa once, then size and power.

Three questions in order, and the rule fails if any of them fails.

  1. CAN kappa BE SET ONCE?  The ratio's scale depends on n, so kappa is
     calibrated at a LIGHT tail (index 4, where theta = 0 is admissible and
     preferred) and then held fixed everywhere.  If the value that makes the rule
     pick theta = 0 at index 4 also behaves at index 1.2, kappa is a constant of
     the method rather than a tuning knob.
  2. DOES IT HOLD SIZE?  theta-hat is random and selected on the same data, so
     this is a pre-test: it can distort the level of the test that follows.
  3. DOES IT RECOVER ORACLE POWER?  Compared against the best FIXED theta per
     cell, which no practitioner can know.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from twomoments import Design, operator, cg          # noqa: E402
from theta_selector import select, max_to_sum, GRID  # noqa: E402

OUT = HERE.parent / "results"
SEED = 20260829
LEVEL = 0.05
N = 1000
INDICES = [1.2, 1.5, 1.8, 2.5, 4.0]


def stat_and_crit(X, Y, b, theta, rng, m_max=40, ndraw=3000):
    n = len(Y)
    A, r = operator(X, Y, theta)
    a_n = 1.0 / (np.sqrt(n) * np.log(n))
    for m in range(1, m_max + 1):
        bh = cg(A, r, m)
        if np.linalg.norm(r - A @ bh) <= a_n:
            break
    stat = n * float(np.sum((A @ (bh - b)) ** 2))
    nrm = np.linalg.norm(X, axis=1)
    w = np.zeros(n)
    nz = nrm > 0
    w[nz] = nrm[nz] ** (-theta) if theta else 1.0
    if not theta:
        w = np.ones(n)
    Z = X * (w * (Y - X @ b))[:, None]
    ev = np.clip(np.linalg.eigvalsh(Z.T @ Z / n), 0.0, None)
    ev = ev[ev > 1e-14]
    if ev.size == 0:
        return stat, np.inf
    return stat, float(np.quantile(
        (rng.chisquare(1.0, (ndraw, ev.size)) * ev).sum(1), 1 - LEVEL))


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    rng = np.random.default_rng(SEED)
    design = Design(J=20, decay=1.0)
    b = design.beta
    v = b / np.linalg.norm(b)
    c = 0.716                                   # the calibrated departure of A1.7
    se = np.sqrt(LEVEL * (1 - LEVEL) / R)

    # ---- 1. calibrate kappa once, at the lightest tail
    print("1. CALIBRATING kappa ONCE, at tail index 4.0 where theta = 0 is")
    print("   admissible and preferred.  Distribution of the max-to-sum ratio")
    print("   R_n(0) under H0:")
    rs = []
    for _ in range(400):
        X, Y = design.draw(N, 4.0, rng)
        rs.append(max_to_sum(X, Y - X @ b, 0.0))
    q = np.quantile(rs, [.5, .9, .95, .99])
    print(f"   median {q[0]:.4f}   p90 {q[1]:.4f}   p95 {q[2]:.4f}   p99 {q[3]:.4f}")
    kappa = float(np.quantile(rs, .95))
    print(f"   kappa := the 95th percentile there = {kappa:.4f}\n")
    print("   what R_n(0) looks like at heavier tails (median), for contrast:")
    for idx in INDICES:
        m = np.median([max_to_sum(*design.draw(N, idx, rng)[:1],
                                  resid=None, theta=None) if False else
                       max_to_sum(X_, Y_ - X_ @ b, 0.0)
                       for X_, Y_ in (design.draw(N, idx, rng) for _ in range(120))])
        print(f"     index {idx:4.1f}:  R_n(0) = {m:.4f}"
              + ("   <= kappa" if m <= kappa else "   > kappa  -> reweight"))

    # ---- 2 & 3. size and power under the rule
    print()
    print("2+3. SIZE AND POWER UNDER THE RULE, against fixed theta and the oracle")
    print(f"     n = {N}, R = {R}, nominal {LEVEL:.0%}, s.e. {se:.4f}, "
          f"tolerance [{LEVEL-2*se:.3f}, {LEVEL+2*se:.3f}]")
    print(f"     grid {GRID}, kappa = {kappa:.4f}, local departure c = {c}")
    rows = {}
    print(f"\n   {'index':>7} {'method':>14} {'size':>9} {'power':>9} "
          f"{'median th-hat':>15} {'fallback':>10}")
    for idx in INDICES:
        # fixed thetas
        fixed = {}
        for th in (0.0, 0.5, 1.0):
            rej0 = rej1 = 0
            for _ in range(R):
                X, Y = design.draw(N, idx, rng)
                s, cr = stat_and_crit(X, Y, b, th, rng)
                rej0 += s > cr
                X, Y = design.draw(N, idx, rng)
                Y = Y + X @ (c * v) / np.sqrt(N)
                s, cr = stat_and_crit(X, Y, b, th, rng)
                rej1 += s > cr
            fixed[th] = (rej0 / R, rej1 / R)
        # the rule
        rej0 = rej1 = 0
        ths, fbs = [], 0
        for _ in range(R):
            X, Y = design.draw(N, idx, rng)
            th, fb = select(X, Y - X @ b, kappa)
            ths.append(th); fbs += fb
            s, cr = stat_and_crit(X, Y, b, th, rng)
            rej0 += s > cr
            X, Y = design.draw(N, idx, rng)
            Y = Y + X @ (c * v) / np.sqrt(N)
            th2, _ = select(X, Y - X @ b, kappa)
            s, cr = stat_and_crit(X, Y, b, th2, rng)
            rej1 += s > cr
        rule = (rej0 / R, rej1 / R)
        valid = {t: p for t, p in fixed.items() if abs(p[0] - LEVEL) <= 2 * se}
        oracle = max(valid.values(), key=lambda z: z[1])[1] if valid else float("nan")
        rows[str(idx)] = {"fixed": {str(k): v_ for k, v_ in fixed.items()},
                          "rule": rule, "oracle": oracle,
                          "median_theta": float(np.median(ths)),
                          "fallback_rate": fbs / R}
        for th in (0.0, 0.5, 1.0):
            f = "*" if abs(fixed[th][0] - LEVEL) > 2 * se else " "
            print(f"   {idx:7.1f} {'fixed th=' + str(th):>14} "
                  f"{fixed[th][0]:8.3f}{f} {fixed[th][1]:9.3f}")
        f = "*" if abs(rule[0] - LEVEL) > 2 * se else " "
        print(f"   {idx:7.1f} {'RULE':>14} {rule[0]:8.3f}{f} {rule[1]:9.3f} "
              f"{np.median(ths):15.2f} {fbs / R:9.2f}")
        print(f"   {idx:7.1f} {'oracle (size-':>14} {'valid':>9} {oracle:9.3f}"
              f"   <- best fixed theta that held size")
        print()
    print("   * = size outside +-2 s.e. of nominal")
    OUT.mkdir(exist_ok=True)
    with open(OUT / "theta_selector.json", "w") as f:
        json.dump({"kappa": kappa, "R": R, "n": N, "grid": list(GRID),
                   "rows": rows}, f, indent=2, default=float)
    print(f"\nwrote {OUT / 'theta_selector.json'}")


if __name__ == "__main__":
    main()
