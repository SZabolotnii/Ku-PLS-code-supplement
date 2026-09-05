#!/usr/bin/env python3
"""A2 pilot --- is the (2,4) gap non-empty in practice, and where does BCT win?

The plan's A2: 're-run the size/power study at tail indices spanning (2,4) AND
> 4, which is now the honest comparison axis.  Report where BCT wins (index > 4)
as prominently as where it fails.'  This is the estimation half; the test is
reported as a diagnostic only.

Metric: relative slope error ||betahat_m - beta|| / ||beta||, MEDIAN across
replications, because at index < 4 the theta = 0 column has no finite mean and a
mean would be reporting the failure as a number rather than as a fact.  The
interquartile range is reported alongside, and the 90th percentile, since the
whole claim is about the tail of the estimator's own distribution.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from twomoments import Design, fit, discrepancy_stop, wald_diagnostic  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
SEED = 20260821
THETAS = [0.0, 0.5, 1.0, 1.5, 2.0]
INDICES = [2.5, 3.0, 3.5, 5.0, 8.0]


def cell(design, index, n, R, rng, eps_law, m):
    beta = design.beta
    nb = np.linalg.norm(beta)
    err = {th: np.empty(R) for th in THETAS}
    mstop = {th: np.empty(R) for th in THETAS}
    for j in range(R):
        X, Y = design.draw(n, index, rng, eps_law)
        for th in THETAS:
            a, ms = discrepancy_stop(X, Y, th)
            err[th][j] = np.linalg.norm(a - beta) / nb
            mstop[th][j] = ms
    return {th: dict(med=float(np.median(err[th])),
                     iqr=float(np.subtract(*np.percentile(err[th], [75, 25]))),
                     p90=float(np.percentile(err[th], 90)),
                     m=float(np.median(mstop[th]))) for th in THETAS}


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    n = 1000
    rng = np.random.default_rng(SEED)
    design = Design(J=20, decay=1.0)
    out = {}

    for eps_law in ("gauss", "t3"):
        print("=" * 92)
        print(f"relative slope error  ||betahat - beta|| / ||beta||   "
              f"(n = {n}, R = {R}, eps ~ {eps_law}, adaptive early stopping)")
        print(f"BCT is theta = 0.  Its Assumption 1 (E||X||^4 < inf) HOLDS only "
              f"in the last two columns.")
        print("=" * 92)
        hdr = f"{'theta':>7} |" + "".join(f"{'idx ' + str(i):>16}" for i in INDICES)
        print(hdr)
        print(f"{'':>7} |" + "".join(f"{'median (p90)':>16}" for _ in INDICES))
        print("-" * len(hdr))
        rows = {th: {} for th in THETAS}
        for index in INDICES:
            res = cell(design, index, n, R, rng, eps_law, m=3)
            for th in THETAS:
                rows[th][index] = res[th]
        for th in THETAS:
            tag = "  (BCT)" if th == 0 else " (sign)" if th == 2 else "       "
            line = f"{th:5.1f}{tag}|"
            for index in INDICES:
                d = rows[th][index]
                line += f"{d['med']:9.3f} ({d['p90']:4.2f})"
            print(line)
        print()
        best = {}
        for index in INDICES:
            b = min(THETAS, key=lambda t: rows[t][index]["med"])
            best[index] = b
            gain = rows[0.0][index]["med"] / rows[b][index]["med"]
            print(f"   tail index {index:<4}: best theta = {b:.1f}"
                  f"   median error vs BCT = {1/gain:5.2f}x"
                  f"   (BCT {rows[0.0][index]['med']:.3f} -> {rows[b][index]['med']:.3f})")
        print()
        out[eps_law] = {"rows": {str(t): {str(i): rows[t][i] for i in INDICES}
                                 for t in THETAS}, "best": {str(k): v for k, v in best.items()}}

    OUT.mkdir(exist_ok=True)
    with open(OUT / "a2_pilot_estimation.json", "w") as f:
        json.dump({"n": n, "R": R, "seed": SEED, "result": out}, f, indent=2)
    print(f"wrote {OUT / 'a2_pilot_estimation.json'}")


if __name__ == "__main__":
    main()
