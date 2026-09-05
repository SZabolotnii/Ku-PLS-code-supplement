#!/usr/bin/env python3
"""A2 pilot, part 2 --- the half where the moment wall actually binds.

The estimation pilot found essentially no gain: at tail index 2.5, where BCT's
E||X||^4 < inf fails outright, theta = 1 improves the MEDIAN slope error by 12%
and by index 3.5 by nothing.  Before concluding anything from that, this script
asks whether the estimation comparison was even capable of showing an effect:

  PART 1 -- BIAS/VARIANCE DECOMPOSITION.  With adaptive early stopping at m ~ 3
  and a slope decaying like j^-2, the error may be dominated by regularisation
  BIAS, which is a property of the stopping rule and identical across theta.  If
  so the estimation table is uninformative about moments by construction and must
  not be reported as evidence either way.  Measured: ||E betahat - beta|| (bias)
  against the spread of betahat around its own mean (variance), per theta.

  PART 2 -- SIZE OF THE PLUG-IN WEIGHTED-CHI^2.  This is where the fourth moment
  is load-bearing: BCT's weights are the eigenvalues of V = E[eps^2 X (x) X],
  which does not exist when E||X||^4 = inf.  If the theta = 1 column holds its
  nominal level across the (2,4) range while theta = 0 does not, the paper is an
  INFERENCE paper and A1.7 must be proved.  If neither holds it, the route is a
  validity-only paper with nothing to be valid about, and section 6's kill
  criteria apply.

Nothing here is a limit-law claim.  The calibration is plug-in and its theory is
open (A1.7).  This measures what it does.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from twomoments import Design, operator, cg, wald_diagnostic  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
SEED = 20260821
THETAS = [0.0, 0.5, 1.0, 1.5, 2.0]
INDICES = [2.5, 3.0, 3.5, 5.0, 8.0]


def part1(R, n, rng):
    print("=" * 88)
    print("PART 1 -- is the estimation comparison capable of showing a moment effect?")
    print("   ||mean(betahat) - beta||  is regularisation BIAS (a property of m).")
    print("   mean ||betahat - mean(betahat)||  is the VARIANCE the moments govern.")
    print("=" * 88)
    design = Design(J=20, decay=1.0)
    beta, nb = design.beta, np.linalg.norm(design.beta)
    for m in (3, 8):
        print(f"\n   fixed m = {m}")
        print(f"   {'theta':>7} {'index':>7} {'bias/||b||':>12} {'sd/||b||':>10} "
              f"{'bias share':>12}")
        for index in (2.5, 8.0):
            for th in THETAS:
                B = np.empty((R, design.J))
                for j in range(R):
                    X, Y = design.draw(n, index, rng)
                    A, r = operator(X, Y, th)
                    B[j] = cg(A, r, m)
                mu = B.mean(0)
                bias = np.linalg.norm(mu - beta) / nb
                sd = float(np.mean(np.linalg.norm(B - mu, axis=1))) / nb
                print(f"   {th:7.1f} {index:7.1f} {bias:12.4f} {sd:10.4f} "
                      f"{bias**2 / (bias**2 + sd**2):12.1%}")
    print("\n   -> a bias share near 100% means the estimation table was measuring")
    print("      the stopping rule, not the moment condition.")


def part2(R, n, rng, level=0.05):
    print()
    print("=" * 88)
    print(f"PART 2 -- empirical size of the plug-in weighted-chi^2 at nominal "
          f"{level:.0%}  (n = {n}, R = {R})")
    print("   H0 is TRUE in every cell: the test is run at b = beta.")
    print("   BCT is theta = 0.  Its variance operator V = E[eps^2 X (x) X] exists")
    print("   only in the last two columns.")
    print("=" * 88)
    design = Design(J=20, decay=1.0)
    hdr = f"{'theta':>11} |" + "".join(f"{'idx ' + str(i):>10}" for i in INDICES)
    print(hdr)
    print("-" * len(hdr))
    table = {}
    for th in THETAS:
        row = []
        for index in INDICES:
            rej = 0
            for _ in range(R):
                X, Y = design.draw(n, index, rng)
                s, c = wald_diagnostic(X, Y, design.beta, th, m=3, rng=rng,
                                       level=level)
                rej += s > c
            row.append(rej / R)
        table[th] = row
        tag = "  (BCT)" if th == 0 else " (sign)" if th == 2 else "       "
        print(f"{th:4.1f}{tag}    |" + "".join(f"{v:10.3f}" for v in row))
    print()
    print("   a well-calibrated column sits near 0.050; the Monte Carlo standard")
    print(f"   error at R = {R} is {np.sqrt(level*(1-level)/R):.3f}, so read "
          f"anything outside "
          f"[{level - 2*np.sqrt(level*(1-level)/R):.3f}, "
          f"{level + 2*np.sqrt(level*(1-level)/R):.3f}] as a real distortion.")
    return table


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = np.random.default_rng(SEED)
    part1(R, 1000, rng)
    tab = part2(R, 1000, rng)
    OUT.mkdir(exist_ok=True)
    with open(OUT / "a2_pilot_inference.json", "w") as f:
        json.dump({"R": R, "n": 1000, "indices": INDICES,
                   "size": {str(k): v for k, v in tab.items()}}, f, indent=2)
    print(f"\nwrote {OUT / 'a2_pilot_inference.json'}")


if __name__ == "__main__":
    main()
