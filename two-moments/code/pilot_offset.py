#!/usr/bin/env python3
r"""Repair A1: the split threshold at a FIXED pilot depth, without the false claim.

WHAT WAS WRONG.  `rem:splitledger` asserted

    "The pilot needs consistency, not a rate: ||beta_0 - beta|| = o_P(1) at fixed m_0
     follows from lem:conc applied on D1 together with ker A_theta = {0}."

That is false, and a PaperMentor review (2026-09-05, critical #6) named it.  At FIXED
m_0 the empirical conjugate-gradient iterate converges in probability to the POPULATION
m_0-step iterate beta^(m_0), not to beta.  ker A_theta = {0} gives beta^(m) -> beta as
m -> infinity, which is a different limit.  lem:trvsplit's proof closes with "both
remainders vanish because ||d|| = o_P(1)", so the lemma as stated rested on it.

This is a review remark reintroduced by its own repair: a statement broader
than what proves it.

WHAT IS ACTUALLY TRUE, and what this script checks.  At fixed m_0 the split threshold
converges not to tr V_theta but to

    tr V_theta + Delta(m_0),
    Delta(m_0) = 2 E[eps <d_0,X> w] + E[<d_0,X>^2 w],    d_0 = beta - beta^(m_0),
    w = ||X||^{2-2theta},

and Delta is FINITE and BOUNDED by the two moment conditions already assumed:

    |Delta(m_0)| <= 2 ||d_0|| (tr V_theta * M)^{1/2} + ||d_0||^2 M,   M = E||X||^{4-2theta},

by Cauchy-Schwarz -- the same two applications the existing proof already makes.  Under
(MI), E[eps | X] = 0, the cross term has mean zero and the bound sharpens to

    0 <= Delta(m_0) <= ||d_0||^2 M.

That is exactly what step (f) of the rate proposition needs: a single number within a
CONSTANT FACTOR of tr V_theta, fixed in advance and independent of the stopping path.
It is also what the manuscript's own prose beside the lemma already claimed.  Only the
lemma overreached.

THE CHECKS, each stated before it is run:

  A  the pilot is NOT consistent at fixed m_0.  ||beta_hat_{m_0} - beta|| must converge
     to a POSITIVE constant as n grows, not to zero.  A test that merely showed the
     error shrinking would be the false claim restated, so the prediction is the
     opposite: it floors.
  B  the offset is real but small, and obeys the bound above.
  C  under (MI) the cross term's MEAN is zero, so the test is |estimate| < 3 s.e.
     of its own Monte Carlo error -- not a comparison with the bias term, which is
     itself small enough here that a fraction of it lies below the noise floor.
     With the cross term zero, Delta >= 0, so the threshold is
     conservative -- it over-estimates the noise level and therefore stops EARLIER,
     which is the safe direction for a regularisation parameter.
  D  what step (f) needs: tr V_split / tr V_theta converges to a constant in [1, 1+c],
     measured across theta and tail index rather than argued.
  E  the alternative route is real: letting m_0 grow shrinks the offset.  Reported so
     the choice between the two routes is made on numbers, not on preference.

Run:  python code/pilot_offset.py [--quick]
Writes results/pilot_offset.{json,out}.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from split_stopping_rule import DESIGN, operator, cg, weights, tr_v_split  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 20260905
M0 = 3
RHO = 0.35


def population_pilot(theta, index, n_pop, rng, m0=M0):
    """d_0 = beta - beta^(m_0), the population pilot error, by a very large sample."""
    X, Y = DESIGN.draw(n_pop, index, rng)
    A, r = operator(X, Y, theta)
    return DESIGN.beta - cg(A, r, m0)


def offset_terms(theta, index, d0, n_pop, rng):
    """The three pieces of E[(eps + <d0,X>)^2 w] and the Cauchy-Schwarz bound."""
    X, Y = DESIGN.draw(n_pop, index, rng)
    w = weights(X, theta)
    eps = Y - X @ DESIGN.beta
    proj = X @ d0
    trv = float(np.mean(eps ** 2 * w))
    bias = float(np.mean(proj ** 2 * w))
    cross_terms = 2 * eps * proj * w
    cross = float(np.mean(cross_terms))
    # The cross term's mean is EXACTLY zero under (MI), so the only honest test is
    # against its own Monte Carlo error.  A first version compared it with the bias
    # term instead and reported a failure: the bias is itself ~5e-4 here, so a
    # quarter of it sits BELOW the noise floor and the test could not resolve what
    # it was asking.  Same defect class as the three caught in repair (a).
    cross_se = float(np.std(cross_terms, ddof=1) / math.sqrt(len(cross_terms)))
    M = float(np.mean(np.linalg.norm(X, axis=1) ** (4 - 2 * theta)))
    nd = float(np.linalg.norm(d0))
    bound = 2 * nd * math.sqrt(max(trv, 0.0) * M) + nd ** 2 * M
    # The manuscript prints Delta/tr V as a PERCENTAGE, so the percentage is what the
    # artifact must hold: a CoE audit resolves printed values, and 0.12 is not 0.0012.
    return {"trV": trv, "bias": bias, "cross": cross, "cross_se": cross_se,
            "delta_over_trV_pct": 100.0 * (bias + cross) / trv if trv else 0.0,
            "cross_z": cross / cross_se if cross_se > 0 else 0.0,
            "M": M, "norm_d0": nd,
            "delta": bias + cross, "cs_bound": bound,
            "bound_holds": bool(abs(bias + cross) <= bound)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    n_pop = 100000 if args.quick else 400000
    ns_a = (2000, 8000, 32000) if args.quick else (2000, 8000, 32000, 128000)
    reps_a = 20 if args.quick else 60
    cells = ((1.0, 2.5), (0.5, 2.5), (2.0, 2.5), (1.0, 3.5))
    m0_grid = (1, 2, 3, 5, 8, 12)

    res = {"seed": SEED, "quick": args.quick, "m0": M0, "rho": RHO,
           "config": {"n_pop": n_pop, "ns_part_a": list(ns_a), "reps_part_a": reps_a,
                      "cells": [{"theta": t, "index": i} for t, i in cells],
                      "m0_grid": list(m0_grid)}}

    # ---- A: the pilot floors, it does not converge to beta ------------------
    theta, index = 1.0, 2.5
    d0 = population_pilot(theta, index, n_pop, rng)
    floor = float(np.linalg.norm(d0) / np.linalg.norm(DESIGN.beta))
    rows_a = {}
    for n in ns_a:
        errs = []
        for _ in range(reps_a):
            X, Y = DESIGN.draw(n, index, rng)
            A, r = operator(X, Y, theta)
            errs.append(float(np.linalg.norm(cg(A, r, M0) - DESIGN.beta)
                              / np.linalg.norm(DESIGN.beta)))
        rows_a[str(n)] = {"median_rel_pilot_error": float(np.median(errs))}
    seq = [rows_a[str(n)]["median_rel_pilot_error"] for n in ns_a]
    # It must approach the floor from above, and must NOT approach zero.
    approaches_floor = bool(abs(seq[-1] - floor) < 0.5 * abs(seq[0] - floor))
    not_consistent = bool(seq[-1] > 0.5 * floor)
    res["A_pilot_floors"] = {"population_floor_rel": floor, "rows": rows_a,
                             "approaches_the_floor": approaches_floor,
                             "does_not_go_to_zero": not_consistent,
                             "pass": bool(approaches_floor and not_consistent)}

    # ---- B, C: the offset, its bound, and its sign under (MI) ---------------
    rows_b = {}
    for th, ix in cells:
        d = population_pilot(th, ix, n_pop, rng)
        rows_b[f"theta={th}, index={ix}"] = offset_terms(th, ix, d, n_pop, rng)
    bounds_hold = all(v["bound_holds"] for v in rows_b.values())
    # (MI) holds in this design, so the cross term's mean is exactly zero and the
    # test is whether the estimate is within Monte Carlo error of zero.
    cross_negligible = all(abs(v["cross_z"]) < 3.0 for v in rows_b.values())
    res["B_offset"] = {"rows": rows_b, "cauchy_schwarz_bound_holds": bounds_hold,
                       "cross_term_negligible_under_MI": cross_negligible,
                       "pass": bool(bounds_hold and cross_negligible)}

    # ---- D: what step (f) needs -- a constant factor -------------------------
    rows_d = {}
    for th, ix in cells:
        ratios = []
        for _ in range(reps_a):
            X, Y = DESIGN.draw(8000, ix, rng)
            trv, _ = tr_v_split(X, Y, th, rng, rho=RHO, m0=M0)
            key = f"theta={th}, index={ix}"
            ratios.append(trv / rows_b[key]["trV"])
        rows_d[f"theta={th}, index={ix}"] = {
            "median_ratio": float(np.median(ratios)),
            "q10": float(np.quantile(ratios, 0.10)),
            "q90": float(np.quantile(ratios, 0.90))}
    within = all(0.5 <= v["median_ratio"] <= 2.0 for v in rows_d.values())
    res["D_constant_factor"] = {"rows": rows_d,
                                "median_ratio_within_factor_two": within,
                                "pass": bool(within)}

    # ---- E: the alternative route, priced ------------------------------------
    rows_e = {}
    for m in m0_grid:
        d = population_pilot(1.0, 2.5, n_pop, rng, m0=m)
        tm = offset_terms(1.0, 2.5, d, n_pop, rng)
        rows_e[str(m)] = {"norm_d0_rel": float(np.linalg.norm(d)
                                               / np.linalg.norm(DESIGN.beta)),
                          "delta": tm["delta"], "delta_over_trV": tm["delta"] / tm["trV"],
                          "delta_over_trV_pct": 100.0 * tm["delta"] / tm["trV"]}
    res["E_growing_m0"] = {"rows": rows_e, "pass": True}

    res["all_pass"] = all(res[k]["pass"] for k in res
                          if isinstance(res[k], dict) and "pass" in res[k])

    L, P = [], None
    P = L.append
    P("=" * 78)
    P("Repair A1: the split threshold at a fixed pilot depth")
    P(f"seed {SEED}   m0={M0}   rho={RHO}" + ("   [--quick]" if args.quick else ""))
    P("=" * 78)
    a = res["A_pilot_floors"]
    P("")
    P("A. the pilot is NOT consistent at fixed m0 (theta=1, index 2.5)")
    P(f"   population {M0}-step CG error, relative: {a['population_floor_rel']:.5f}")
    for n in ns_a:
        P(f"      n={n:>7}   median relative pilot error "
          f"{a['rows'][str(n)]['median_rel_pilot_error']:.5f}")
    P(f"   approaches that floor rather than zero: "
      f"{'yes' if a['approaches_the_floor'] else 'NO'}; "
      f"stays above half of it: {'yes' if a['does_not_go_to_zero'] else 'NO'}")
    P("   -> the claim 'consistency follows from ker A_theta = {0}' is REFUTED:")
    P("      ker A_theta = {0} gives beta^(m) -> beta as m -> infinity, not as n -> infinity.")
    P("")
    P("B/C. the offset it induces, and the bound the existing proof already supports")
    P(f"      {'cell':<24}{'||d0||':>9}{'bias':>11}{'cross':>11}{'cross/se':>10}"
      f"{'Delta':>11}{'CS bound':>10}{'Delta/trV':>11}")
    for k, v in res["B_offset"]["rows"].items():
        P(f"      {k:<24}{v['norm_d0']:>9.4f}{v['bias']:>11.6f}{v['cross']:>11.6f}"
          f"{v['cross_z']:>10.2f}{v['delta']:>11.6f}{v['cs_bound']:>10.4f}"
          f"{v['delta']/v['trV']:>10.3%}")
    P(f"   Cauchy-Schwarz bound holds in every cell: "
      f"{'yes' if res['B_offset']['cauchy_schwarz_bound_holds'] else 'NO'}")
    P(f"   cross term within 3 s.e. of zero, as (MI) predicts: "
      f"{'yes' if res['B_offset']['cross_term_negligible_under_MI'] else 'NO'}")
    P("")
    P("D. what step (f) actually needs: tr V_split / tr V_theta, a constant factor")
    P(f"      {'cell':<24}{'q10':>9}{'median':>9}{'q90':>9}")
    for k, v in res["D_constant_factor"]["rows"].items():
        P(f"      {k:<24}{v['q10']:>9.4f}{v['median_ratio']:>9.4f}{v['q90']:>9.4f}")
    P("")
    P("E. the alternative route, priced: growing m0 shrinks the offset")
    P(f"      {'m0':>4}{'||d0||/||beta||':>17}{'Delta':>12}{'Delta/trV':>12}")
    for m in m0_grid:
        v = res["E_growing_m0"]["rows"][str(m)]
        P(f"      {m:>4}{v['norm_d0_rel']:>17.5f}{v['delta']:>12.6f}"
          f"{v['delta_over_trV']:>11.3%}")
    P("   So route (A), m0 = m0(n) -> infinity, is real and would restore the lemma as")
    P("   written; route (B), fixed m0 with the offset stated, needs no new sequence and")
    P("   is what the manuscript's own prose beside the lemma already claimed.")
    P("")
    P("=" * 78)
    P("ALL PASS" if res["all_pass"] else "FAILURE -- a prediction did not hold")
    P("=" * 78)

    text = "\n".join(L)
    print(text)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "pilot_offset.out").write_text(text + "\n", encoding="utf-8")
    (RESULTS / "pilot_offset.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
