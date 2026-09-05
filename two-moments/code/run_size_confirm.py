#!/usr/bin/env python3
"""The load-bearing number of Route A, run at a sample size that can carry it.

The pilot at R = 400 put BCT's empirical size at 0.083 (index 2.5) and 0.077
(index 3.0) against a nominal 0.05, with theta = 1 the only column inside
tolerance everywhere.  Monte Carlo standard error there was 0.011, so the
headline cells were about three standard errors from nominal --- suggestive, not
established.  This re-runs the same design at R = 2500 (s.e. 0.004) and at two
sample sizes, because the claim is that BCT's distortion does NOT improve with n
(there being no limit to approach) while theta = 1's is already correct.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from twomoments import Design, wald_diagnostic  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
SEED = 20260822
THETAS = [0.0, 1.0, 2.0]
INDICES = [2.5, 3.0, 3.5, 5.0, 8.0]


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    se = np.sqrt(.05 * .95 / R)
    rng = np.random.default_rng(SEED)
    design = Design(J=20, decay=1.0)
    out = {}
    for n in (1000, 4000):
        print("=" * 78)
        print(f"empirical size, nominal 5%   n = {n}, R = {R}, s.e. = {se:.4f}")
        print(f"tolerance +-2 s.e. = [{.05-2*se:.4f}, {.05+2*se:.4f}]")
        print("=" * 78)
        print(f"{'theta':>12} |" + "".join(f"{'idx ' + str(i):>10}" for i in INDICES))
        print("-" * 66)
        for th in THETAS:
            row = []
            for index in INDICES:
                rej = 0
                for _ in range(R):
                    X, Y = design.draw(n, index, rng)
                    s, c = wald_diagnostic(X, Y, design.beta, th, 3, rng)
                    rej += s > c
                row.append(rej / R)
            out[f"n{n}_th{th}"] = row
            tag = " (BCT)" if th == 0 else "(sign)" if th == 2 else "      "
            marks = "".join(
                f"{v:9.3f}" + ("*" if abs(v - .05) > 2 * se else " ") for v in row)
            print(f"{th:5.1f}{tag}     |{marks}")
        print("   * = outside +-2 s.e. of nominal\n")
    OUT.mkdir(exist_ok=True)
    with open(OUT / "a2_size_confirm.json", "w") as f:
        json.dump({"R": R, "se": se, "indices": INDICES, "size": out}, f, indent=2)
    print(f"wrote {OUT / 'a2_size_confirm.json'}")


if __name__ == "__main__":
    main()
