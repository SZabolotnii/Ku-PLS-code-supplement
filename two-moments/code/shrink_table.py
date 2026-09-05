#!/usr/bin/env python3
"""Table A.2: the shrinkage factors d_j(theta) on the population diagonal.

WHY THIS EXISTS.  Like Table 11, this table was produced by a driver that was not
kept, so nothing in the repository regenerated it.  It is the second of the two
gaps in the replication package.

WHAT IS COMPUTED.

    d_j(theta) = E[ ||Z||^{-theta} Z_j^2 ] / ( sigma_j^2 E||Z||^{-theta} )

for Z ~ N(0, diag(sigma^2)) on J = 200 coordinates with sigma_j = 1/j.  Writing
Z_j = sigma_j * G_j with G standard normal, the sigma_j^2 cancels and d_j is a
WEIGHTED MEAN of G_j^2 with weights w = ||Z||^{-theta}:

    d_j(theta) = sum_i w_i G_ij^2 / sum_i w_i.

That form is used here, not the two expectations separately: numerator and
denominator then share their draws, the ratio's Monte Carlo error is much smaller
than either factor's, and d_j(0) = 1 is recovered as an average of G_j^2 rather
than as a quotient of two independent averages.

THE LAST DIGIT IS NOISE, AND THE SCRIPT SAYS SO.  At the paper's 4,000,000 draws
the standard error of each d_j is a few units in the third decimal -- the printed
table's d_1(0) = 1.002 is that noise, since d_j(0) = 1 exactly.  The delta-method
standard error is computed alongside every cell and printed, so a reader can see
which digits are real.  The seed is fixed, so the table is reproducible.

THE TAIL SLOPE is a log-log least-squares regression of d_j on j over
j in [100, 200], the column that carries the claim: the reweighting shrinks the
top of the spectrum and leaves the decay EXPONENT alone, which is what preserves
the complexity class.

Writes results/shrink_table.{json,out}.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "results"

SEED = 20260905
J = 200
N = 4_000_000
CHUNK = 50_000
THETAS = (0.0, 0.5, 1.0, 1.5, 2.0)
COLS = (1, 2, 5, 10, 25, 50, 100, 200)      # the j printed in the table
SLOPE_RANGE = (100, 200)

lines: list[str] = []


def P(s=""):
    print(s)
    lines.append(s)


def run():
    rng = np.random.default_rng(SEED)
    sig2 = (1.0 / np.arange(1, J + 1)) ** 2

    # per theta: sum w, sum w*G^2, and the three sums the delta-method SE needs
    s_w = {t: 0.0 for t in THETAS}
    s_wg = {t: np.zeros(J) for t in THETAS}
    s_w2 = {t: 0.0 for t in THETAS}
    s_w2g = {t: np.zeros(J) for t in THETAS}
    s_w2g2 = {t: np.zeros(J) for t in THETAS}

    done = 0
    while done < N:
        m = min(CHUNK, N - done)
        G2 = rng.standard_normal((m, J)) ** 2
        nrm2 = G2 @ sig2                      # ||Z||^2
        for t in THETAS:
            w = 1.0 if t == 0.0 else nrm2 ** (-t / 2.0)
            if t == 0.0:
                s_w[t] += m
                s_wg[t] += G2.sum(0)
                s_w2[t] += m
                s_w2g[t] += G2.sum(0)
                s_w2g2[t] += (G2 ** 2).sum(0)
            else:
                s_w[t] += w.sum()
                s_wg[t] += w @ G2
                w2 = w * w
                s_w2[t] += w2.sum()
                s_w2g[t] += w2 @ G2
                s_w2g2[t] += w2 @ (G2 ** 2)
        done += m

    out = {}
    for t in THETAS:
        d = s_wg[t] / s_w[t]
        # sum_i (w_i (G_ij^2 - d_j))^2 = sum w^2 G^4 - 2 d sum w^2 G^2 + d^2 sum w^2
        var = s_w2g2[t] - 2.0 * d * s_w2g[t] + (d ** 2) * s_w2[t]
        se = np.sqrt(np.clip(var, 0.0, None)) / s_w[t]
        j = np.arange(1, J + 1)
        sel = (j >= SLOPE_RANGE[0]) & (j <= SLOPE_RANGE[1])
        slope = float(np.polyfit(np.log(j[sel]), np.log(d[sel]), 1)[0])
        out[t] = {"d": d, "se": se, "slope": slope}
    return out


def main():
    res = run()

    P("=" * 96)
    P("SHRINKAGE FACTORS ON THE POPULATION DIAGONAL  (tab:shrink)")
    P(f"   d_j(theta) = E[||Z||^-theta Z_j^2] / (sigma_j^2 E||Z||^-theta),"
      f"  J = {J}, sigma_j = 1/j")
    P(f"   {N:,} draws, seed {SEED}.  d_j(0) = 1 exactly, so the theta = 0 row is")
    P("   the Monte Carlo noise floor of every other row.")
    P("=" * 96)
    head = f"   {'theta':>6}" + "".join(f"{'d_' + str(c):>9}" for c in COLS) \
        + f"{'tail slope':>13}"
    P(head)
    for t in THETAS:
        d, s = res[t]["d"], res[t]["slope"]
        P(f"   {t:6.1f}" + "".join(f"{d[c - 1]:9.3f}" for c in COLS)
          + f"{s:13.4f}")

    P()
    P("   standard error of each cell (delta method, same draws)")
    P(f"   {'theta':>6}" + "".join(f"{'d_' + str(c):>9}" for c in COLS))
    for t in THETAS:
        se = res[t]["se"]
        P(f"   {t:6.1f}" + "".join(f"{se[c - 1]:9.4f}" for c in COLS))

    worst = max(float(res[t]["se"][c - 1]) for t in THETAS for c in COLS)
    P()
    P(f"   Largest standard error over the printed cells: {worst:.4f}.  The third")
    P("   decimal is therefore not resolved, and the table should not be read as")
    P("   though it were.  What the table carries is the SHAPE: d_j well below 1 at")
    P("   small j, indistinguishable from 1 by j = 50, and a tail slope of zero.")

    P()
    P(f"   tail slope = OLS of log d_j on log j over j in [{SLOPE_RANGE[0]}, "
      f"{SLOPE_RANGE[1]}]")
    mx = max(abs(res[t]["slope"]) for t in THETAS)
    P(f"   largest magnitude over the five rows: {mx:.5f}")

    OUT.mkdir(exist_ok=True)
    (OUT / "shrink_table.out").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(OUT / "shrink_table.json", "w") as f:
        json.dump({"seed": SEED, "draws": N, "J": J, "sigma": "1/j",
                   "columns": list(COLS), "slope_range": list(SLOPE_RANGE),
                   "rows": {f"{t:.1f}": {
                       "d": [float(res[t]["d"][c - 1]) for c in COLS],
                       "se": [float(res[t]["se"][c - 1]) for c in COLS],
                       "tail_slope": res[t]["slope"]} for t in THETAS},
                   "max_se_printed": worst},
                  f, indent=2)
    print(f"\nwrote {OUT / 'shrink_table.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
