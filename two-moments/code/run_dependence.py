#!/usr/bin/env python3
"""Serial dependence: does the TEST survive it, and what does kappa become?

PROCEDURE FIXED IN ADVANCE, before any of it was run:
  STEP 1  Size of T_n^theta at FIXED theta under the dependent design, phi in
          {0, 0.3, 0.6} (0.3 and 0.6 are the values matched to the Treasury and
          commercial-paper series by their lag-1 Spearman autocorrelation of
          ||X_t||: 0.303 and 0.591).  Theorem 6' assumes i.i.d.; if the test
          loses its level under realistic persistence then no selection rule can
          rescue it and the honest recommendation becomes the block bootstrap.
          THIS QUESTION IS PRIOR TO THE SELECTOR AND IS ASKED FIRST.
  STEP 2  kappa recalibrated as the 95th percentile of R_n(0) at tail index 4.0
          -- where theta = 0 is admissible and preferred -- under the dependent
          design, at the matched phi and the matched n.  The rule's definition is
          unchanged; only the constant moves.
  STEP 3  Size and power of the RULE under dependence, with the recalibrated
          kappa, across the index sweep.

kappa is NOT permitted to be chosen by what it does to the applications.  Step 2
fixes it from a light-tailed null, exactly as the i.i.d. version was fixed, and
step 3 must pass before it is used anywhere.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dependent_design import DependentDesign          # noqa: E402
from theta_selector import select, max_to_sum, GRID   # noqa: E402
from run_theta_selector import stat_and_crit          # noqa: E402

OUT = HERE.parent / "results"
LEVEL, N = 0.05, 1000
PHIS = [0.0, 0.3, 0.6]
INDICES = [1.5, 2.5, 4.0]


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    rng = np.random.default_rng(20260902)
    se = np.sqrt(LEVEL * (1 - LEVEL) / R)
    c = 0.716
    out = {}

    print("STEP 1 -- SIZE OF THE TEST AT FIXED theta UNDER SERIAL DEPENDENCE")
    print(f"   n = {N}, R = {R}, nominal {LEVEL:.0%}, s.e. {se:.4f}, "
          f"tolerance [{LEVEL-2*se:.3f}, {LEVEL+2*se:.3f}]")
    print("   phi = 0.3 matches the Treasury curve, phi = 0.6 the CP curve.\n")
    print(f"   {'theta':>7} {'index':>7} |" + "".join(f"{'phi=' + str(p):>11}"
                                                      for p in PHIS))
    print("   " + "-" * 55)
    size = {}
    for th in (0.0, 0.5, 1.0):
        for idx in INDICES:
            row = []
            for phi in PHIS:
                d = DependentDesign(phi=phi)
                rej = 0
                for _ in range(R):
                    X, Y = d.draw(N, idx, rng)
                    s, cr = stat_and_crit(X, Y, d.beta, th, rng)
                    rej += s > cr
                row.append(rej / R)
            size[f"th{th}_idx{idx}"] = row
            marks = "".join(f"{v:10.3f}" + ("*" if abs(v - LEVEL) > 2 * se else " ")
                            for v in row)
            print(f"   {th:7.1f} {idx:7.1f} |{marks}")
    print("   * = outside +-2 s.e. of nominal")
    out["step1_size"] = size

    print()
    print("STEP 2 -- kappa RECALIBRATED UNDER DEPENDENCE, at index 4.0")
    print(f"   {'phi':>7} {'n':>7} {'median R_n(0)':>15} {'kappa = p95':>13}")
    kappas = {}
    for phi, n in ((0.0, 1000), (0.3, 2494), (0.6, 4396)):
        d = DependentDesign(phi=phi)
        rs = []
        for _ in range(400):
            X, Y = d.draw(n, 4.0, rng)
            rs.append(max_to_sum(X, Y - X @ d.beta, 0.0))
        k = float(np.quantile(rs, .95))
        kappas[str(phi)] = {"n": n, "median": float(np.median(rs)), "kappa": k}
        print(f"   {phi:7.2f} {n:7d} {np.median(rs):15.4f} {k:13.4f}")
    print("   (the i.i.d. value used before was 0.1451 at n = 1000)")
    out["step2_kappa"] = kappas

    print()
    print("STEP 3 -- SIZE AND POWER OF THE RULE UNDER DEPENDENCE (phi = 0.6),")
    print(f"   with kappa = {kappas['0.6']['kappa']:.4f} from step 2")
    kap = kappas["0.6"]["kappa"]
    d = DependentDesign(phi=0.6)
    print(f"\n   {'index':>7} {'size':>9} {'power':>9} {'median th-hat':>15} "
          f"{'th-hat mean':>13}")
    rule = {}
    for idx in [1.2, 1.5, 1.8, 2.5, 4.0]:
        rej0 = rej1 = 0
        ths = []
        for _ in range(R):
            X, Y = d.draw(N, idx, rng)
            th, _ = select(X, Y - X @ d.beta, kap)
            ths.append(th)
            s, cr = stat_and_crit(X, Y, d.beta, th, rng)
            rej0 += s > cr
            X, Y = d.draw(N, idx, rng)
            Y = Y + X @ (c * d.beta / np.linalg.norm(d.beta)) / np.sqrt(N)
            th2, _ = select(X, Y - X @ d.beta, kap)
            s, cr = stat_and_crit(X, Y, d.beta, th2, rng)
            rej1 += s > cr
        f = "*" if abs(rej0 / R - LEVEL) > 2 * se else " "
        print(f"   {idx:7.1f} {rej0/R:8.3f}{f} {rej1/R:9.3f} "
              f"{np.median(ths):15.2f} {np.mean(ths):13.3f}")
        rule[str(idx)] = {"size": rej0 / R, "power": rej1 / R,
                          "median_theta": float(np.median(ths))}
    print("   * = outside +-2 s.e. of nominal")
    out["step3_rule"] = rule

    OUT.mkdir(exist_ok=True)
    with open(OUT / "dependence.json", "w") as f:
        json.dump({"R": R, "n": N, "level": LEVEL, **out}, f, indent=2, default=float)
    print(f"\nwrote {OUT / 'dependence.json'}")


if __name__ == "__main__":
    main()
