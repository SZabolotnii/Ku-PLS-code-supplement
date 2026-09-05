#!/usr/bin/env python3
r"""Repair B7: is 26.95 a property of the data, or of the block length?

WHAT IS AT STAKE.  The number 26.95 -- the baseline's asymptotic 95th percentile divided
by the stationary bootstrap's -- opens the abstract, and the caption of tab:blockboot
calls the bootstrap value "the one the null actually requires".  That phrasing asserts
bootstrap validity as truth.  A PaperMentor review objected (2026-09-05, critical #13),
and the body already carries the honest version of the caveat, so the caption and the
abstract are the parts out of step.

But the reviewer also asked for something the paper does not have: SENSITIVITY TO THE
MEAN BLOCK LENGTH.  The shipped run uses mb = n^{1/3} = 16, a rule of thumb.  If 26.95
moved materially with mb, the headline would be an artefact of one arbitrary choice.
That is a question about the data, not about wording, so it is measured here.

THE CHECKS, stated before they are run:

  A  the ratio across a grid of mean block lengths spanning an order of magnitude,
     mb in {2, 4, 8, 16, 32, 64, 128}.  Prediction: the ratio is LARGE at every mb.
     A ratio that collapses towards one as mb grows would mean the bootstrap null is
     absorbing the signal rather than measuring the calibration, and the headline
     would have to be withdrawn.
  B  the same for theta = 1, where the paper claims the plug-in critical value is right
     to within one per cent.  Prediction: near one at every mb -- if THAT number is
     stable while the baseline's is not, the contrast is the finding.
  C  the bootstrap p-values at each mb, for both theta.  Prediction: the baseline's
     asymptotic p stays far from its bootstrap p at every mb.

Run:  python gates/b7_blocklength.py [--quick]
Writes results/b7_blocklength.{json,out}.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
from empirical2_blockboot import stat_only, asym_crit, blocks  # noqa: E402
from empirical2_cp import load                                 # noqa: E402

OUT = HERE.parent / "results"
SEED = 20260905
MB_GRID = (2, 4, 8, 16, 32, 64, 128)
THETAS = (0.0, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    B = 199 if args.quick else 999

    dX, _Yn, Yv, _d = load()
    n = len(Yv)
    shipped_mb = max(2, int(round(n ** (1 / 3))))

    res = {"seed": SEED, "quick": args.quick,
           "config": {"B": B, "n": n, "mb_grid": list(MB_GRID),
                      "shipped_mb": shipped_mb, "thetas": list(THETAS)}}

    L = []
    P = L.append
    P("=" * 92)
    P("Repair B7: does the 26.95 ratio depend on the mean block length?")
    P(f"seed {SEED}   n={n}   B={B}   shipped mb={shipped_mb}"
      + ("   [--quick]" if args.quick else ""))
    P("=" * 92)
    P("")
    P("   The stationary bootstrap imposes H0 by resampling X-blocks and Y-blocks with")
    P("   INDEPENDENT starts: each series keeps its own dependence, the cross-relation")
    P("   is destroyed.  mb is the only tuning parameter, and the shipped run takes the")
    P("   n^(1/3) rule of thumb.  A headline that moved with mb would be an artefact.")
    P("")

    rows = {}
    for theta in THETAS:
        stat = stat_only(dX, Yv, theta)
        acrit = asym_crit(dX, Yv, theta, rng)
        P(f"   theta = {theta}:  statistic {stat:.4g}   asymptotic 95% {acrit:.4g}")
        P(f"      {'mean block':>12}{'boot 95%':>14}{'asym/boot':>12}{'boot p':>10}")
        per_mb = {}
        for mb in MB_GRID:
            draws = np.empty(B)
            for b in range(B):
                ix = blocks(n, mb, rng)
                iy = blocks(n, mb, rng)
                draws[b] = stat_only(dX[ix], Yv[iy], theta)
            bcrit = float(np.quantile(draws, 0.95))
            ratio = acrit / bcrit if bcrit > 0 else float("inf")
            bp = float(np.mean(draws >= stat))
            per_mb[str(mb)] = {"boot_crit": bcrit, "ratio": ratio, "boot_p": bp}
            mark = "  <- shipped" if mb == shipped_mb else ""
            P(f"      {mb:>12}{bcrit:>14.4g}{ratio:>11.2f}x{bp:>10.4f}{mark}")
        ratios = [v["ratio"] for v in per_mb.values()]
        rows[str(theta)] = {"statistic": stat, "asym_crit": acrit, "by_mb": per_mb,
                            "ratio_min": min(ratios), "ratio_max": max(ratios),
                            "ratio_at_shipped": per_mb[str(shipped_mb)]["ratio"]}
        P(f"      ratio ranges {min(ratios):.2f}x to {max(ratios):.2f}x "
          f"across the grid")
        P("")

    r0 = rows["0.0"]
    r1 = rows["1.0"]
    baseline_large = bool(r0["ratio_min"] > 5.0)
    reweighted_near_one = bool(0.5 < r1["ratio_min"] and r1["ratio_max"] < 2.0)
    res["A_baseline"] = {"ratio_min": r0["ratio_min"], "ratio_max": r0["ratio_max"],
                         "large_at_every_mb": baseline_large, "pass": baseline_large}
    res["B_reweighted"] = {"ratio_min": r1["ratio_min"], "ratio_max": r1["ratio_max"],
                           "near_one_at_every_mb": reweighted_near_one,
                           "pass": reweighted_near_one}
    res["rows"] = rows

    P("=" * 92)
    P(f"   baseline (theta=0) ratio stays above 5x at every mb: "
      f"{'yes' if baseline_large else 'NO'}"
      f"   (range {r0['ratio_min']:.2f}-{r0['ratio_max']:.2f})")
    P(f"   reweighted (theta=1) ratio stays within [0.5, 2] at every mb: "
      f"{'yes' if reweighted_near_one else 'NO'}"
      f"   (range {r1['ratio_min']:.2f}-{r1['ratio_max']:.2f})")
    P("   The SIZE of the ratio moves with mb, as it must -- a longer block means a")
    P("   wider bootstrap null.  What the grid decides is whether the CONTRAST between")
    P("   the two thetas is a property of the data or of one arbitrary mb.")
    res["all_pass"] = all(res[k]["pass"] for k in res
                          if isinstance(res[k], dict) and "pass" in res[k])
    P("ALL PASS" if res["all_pass"] else "FAILURE -- a prediction did not hold")
    P("=" * 92)

    text = "\n".join(L)
    print(text)
    OUT.mkdir(exist_ok=True)
    (OUT / "b7_blocklength.out").write_text(text + "\n", encoding="utf-8")
    (OUT / "b7_blocklength.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
