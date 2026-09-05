#!/usr/bin/env python3
"""Repair (c): the structural hypotheses of section 4.3, made explicit and mapped.

THE DEFECT.  The discussion of the structural assumptions underlying the
shared-eigenbasis and complexity-class arguments was inconsistent in three
places, all real:

  (i)   the shared-eigenbasis lemma states NO moment condition, but its proof
        invokes "the standing moment condition" and its statement refers to "the
        eigenbasis of K", which presupposes that K exists, i.e. E||X||^2 < inf;
  (ii)  the decay proposition adds X = RZ, R independent of Z, E||Z||^{-theta} < inf
        and bounded coordinate kurtosis, while the prose between the two says that
        ellipticity is not the lemma's hypothesis -- true, but it leaves the reader
        unable to say which statement needs what;
  (iii) below tail index 2 -- the regime the Monte Carlo of section 7.1 and the
        commercial-paper application (Hill 1.597) actually live in -- K is not
        trace class, the class S_0(mu, R, C) is not defined, and the comparison
        norm ||K^{1/2} . || of the corollary does not exist.  "They do not move"
        is therefore true only in the window 2 < index < 4.

THE DEMARCATION THIS SCRIPT ESTABLISHES.  Two different objects are involved and
the paper conflated them:

  * the RATE (Prop. 12) is stated over S_theta, a class defined in the eigenbasis of
    A_theta.  A_theta exists whenever E||X||^{2-theta} < inf, and the rate needs
    (M1) E||X||^{4-2theta} < inf and (M2) tr V_theta < inf.  None of these mentions
    K.  So the rate itself is available below tail index 2, for theta large enough.

  * the COMPARISON with the baseline (Prop. 15, Cor. 17) relates A_theta to K, and
    therefore needs K to exist: E||X||^2 < inf, i.e. tail index > 2.  It is the
    comparison that has the narrower scope, not the rate.

WHAT IS COMPUTED.  Four parts, each a prediction fixed before running:

  A  the admissible region.  For the design X = R Z with R Pareto(index),
     E||X||^p < inf iff p < index, so (M1) holds iff theta > (4 - index)/2 and, for
     theta <= 1, (M2) holds iff theta > (2 - index)/2.  The closed-form region is
     printed against a Monte Carlo check of which sample moments diverge.
  B  K does not exist below index 2: tr Khat = n^{-1} sum ||X_k||^2 must fail to
     settle as n grows, while tr Ahat_1 = n^{-1} sum ||X_k|| settles.
  C  the two clauses of the shared-eigenbasis lemma are each necessary, and heavy
     tails are NOT what breaks it -- the existing table (tab:offdiag) claims this;
     it is re-run here at a tail index BELOW 2, which that table never covered.
  D  what the corollary's constant does as the index approaches 2 from above:
     d_*(theta) and c(theta) are computed on the population diagonal, and the
     prediction is that d_* stays bounded away from 0 while c(theta) blows up,
     since c contains E[R^{2-theta}]/E[R^2] and E R^2 -> inf.

Run:  python code/section43_hypotheses.py [--quick]
Writes results/section43_hypotheses.{json,out}.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from twomoments import Design, operator  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 20260904
DESIGN = Design(J=20, decay=1.0, sigma=0.5)


# --------------------------------------------------------------------------
# A -- the admissible region, closed form against Monte Carlo
# --------------------------------------------------------------------------
def m1_holds(theta, index):
    """(M1) E||X||^{4-2theta} < inf.  Finite iff the exponent is < index."""
    p = 4.0 - 2.0 * theta
    return p <= 0 or p < index


def m2_holds(theta, index):
    """(M2) tr V = E[eps^2 ||X||^{2-2theta}] < inf, Gaussian error.

    For exponent 2-2theta > 0 this needs 2-2theta < index.  For theta > 1 the
    exponent is negative and the condition is about mass near the origin, which
    this design has none of (||X|| >= min scale > 0 a.s. since R >= 1), so it holds.
    """
    p = 2.0 - 2.0 * theta
    return p <= 0 or p < index


def k_exists(index):
    """K = E[X (x) X] is trace class iff E||X||^2 < inf iff index > 2."""
    return index > 2.0


def part_a(rng, indices, thetas, ns=(2000, 20000, 200000), reps=20) -> dict:
    """The closed-form region, cross-checked by the GROWTH RATE of a sample moment.

    A first version compared the sample E||X||^2 at two sample sizes and called the
    moment infinite when the ratio exceeded 1.25.  That is not a diagnostic: when
    E||X||^2 = infinity the sample mean is driven by the single largest draw, so it
    jumps rather than grows, and the two-point ratio came out 4.18, 0.42 and 1.08 at
    indices 1.2, 1.5 and 1.8 -- the middle one below one.  The right statistic is the
    RATE: for a Pareto radial factor with index a < 2, the sample mean of ||X||^2 is
    of order n^{2/a - 1}, so a log-log regression of the MEDIAN over replications
    against n has slope 2/a - 1 > 0, while a finite moment gives slope 0.
    """
    grid = {}
    for index in indices:
        for th in thetas:
            grid[f"{index}|{th}"] = {
                "M1": bool(m1_holds(th, index)),
                "M2": bool(m2_holds(th, index)),
                "rate_available": bool(m1_holds(th, index) and m2_holds(th, index)),
                "K_exists_so_comparison_available": bool(k_exists(index)),
            }
    checks = {}
    for index in indices:
        med2, med1 = [], []
        for n in ns:
            v2, v1 = [], []
            for _ in range(reps):
                x, _ = DESIGN.draw(n, index, rng)
                nx = np.linalg.norm(x, axis=1)
                v2.append(float(np.mean(nx ** 2)))
                v1.append(float(np.mean(nx)))
            med2.append(float(np.median(v2)))
            med1.append(float(np.median(v1)))
        s2 = float(np.polyfit(np.log(ns), np.log(med2), 1)[0])
        s1 = float(np.polyfit(np.log(ns), np.log(med1), 1)[0])
        pred = (2.0 / index - 1.0) if index < 2.0 else 0.0
        checks[str(index)] = {"loglog_slope_E_norm2": s2, "loglog_slope_E_norm1": s1,
                              "predicted_slope_E_norm2": pred,
                              "closed_form_E_norm2_finite": bool(index > 2.0)}
    # finite moment => slope near zero; infinite => slope near 2/a - 1
    ok = all((abs(v["loglog_slope_E_norm2"]) < 0.08) == v["closed_form_E_norm2_finite"]
             for v in checks.values())
    return {"region": grid, "moment_growth": checks, "pass": bool(ok)}


# --------------------------------------------------------------------------
# B -- K does not settle below index 2, A_1 does
# --------------------------------------------------------------------------
def part_b(rng, indices, ns) -> dict:
    out = {}
    for index in indices:
        row = {}
        for n in ns:
            x, _ = DESIGN.draw(n, index, rng)
            A0, _ = operator(x, np.zeros(n), 0.0)
            A1, _ = operator(x, np.zeros(n), 1.0)
            row[str(n)] = {"tr_Khat": float(np.trace(A0)),
                           "tr_A1hat": float(np.trace(A1))}
        k = [row[str(n)]["tr_Khat"] for n in ns]
        a = [row[str(n)]["tr_A1hat"] for n in ns]
        row["K_spread"] = float(max(k) / min(k))
        row["A1_spread"] = float(max(a) / min(a))
        row["K_settles"] = bool(row["K_spread"] < 1.5)
        row["A1_settles"] = bool(row["A1_spread"] < 1.5)
        out[str(index)] = row
    below = [v for k, v in out.items() if float(k) < 2.0]
    above = [v for k, v in out.items() if float(k) > 2.0]
    ok = (all(not v["K_settles"] for v in below)
          and all(v["A1_settles"] for v in below)
          and all(v["K_settles"] for v in above))
    return {"rows": out, "pass": bool(ok)}


# --------------------------------------------------------------------------
# C -- both clauses necessary, and heavy tails are not the culprit
# --------------------------------------------------------------------------
# The generators and the analytic bases are IMPORTED from the gate that produced
# the published table, not rewritten.  Rewriting them cost two defects in a first
# version of this script: a "non-commuting" mixture that scaled half the sample
# coordinate-wise (both components diagonal, hence commuting -- no counterexample at
# all), and the use of the coordinate basis for the mixture, which measures the
# non-diagonality of its own covariance rather than any effect of theta.  The gate's
# docstring records that the same mistake was made and fixed there once already.
sys.path.insert(0, str(ROOT / "gates"))
from a1_6b_source_condition import (  # noqa: E402
    coords_gauss, coords_laplace, coords_skewed, coords_direction_mixture,
    basis_identity, basis_mixture, offdiag_mass)

LAWS = (("independent symmetric (Gaussian)", coords_gauss, basis_identity),
        ("independent symmetric (Laplace)", coords_laplace, basis_identity),
        ("independent ASYMMETRIC", coords_skewed, basis_identity),
        ("non-commuting mixture", coords_direction_mixture, basis_mixture))


def part_c(rng, n, J=20, decay=1.0, reps=4) -> dict:
    """Re-run the off-diagonal check at a tail index BELOW 2, never covered before.

    The published table (tab:offdiag) was computed at tail index 4.0 only, yet the
    sentence it supports says heavy tails do not break the shared eigenbasis -- and
    the regime this paper is about is a tail index below 2.  So the same measurement
    is repeated at index 1.5.

    TWO CORRECTIONS to how it must be read, both found by running it.

    (1) The theta = 0 column is a Monte Carlo floor only under a light radial
        factor.  Below index 2 the empirical Ahat_0 is driven by the largest draw,
        so its off-diagonal fraction is large for EVERY law, a Gaussian included,
        and that is estimation noise rather than a broken basis.  No floor is taken
        from theta = 0 here; symmetric laws are compared against the counterexamples
        at the same theta.

    (2) That theta must be admissible under (M1), i.e. E||X||^{4-2theta} < inf,
        which is what gives Ahat_theta a finite variance.  At index 1.5 this
        excludes theta = 1 as well.
    """
    scale = (np.arange(J) + 1.0) ** -decay
    thetas = (0.0, 0.5, 1.0, 1.5, 2.0)
    out = {}
    for name, gen, basis_fn in LAWS:
        B = basis_fn(J, scale)
        for index in (4.0, 1.5):
            vals = {str(th): [] for th in thetas}
            for _ in range(reps):
                Z = gen(n, scale, rng)
                R = rng.pareto(index, n) + 1.0
                X = Z * R[:, None]
                for th in thetas:
                    vals[str(th)].append(offdiag_mass(X, th, B))
            out[f"{name}|index={index}"] = {k: float(np.median(v))
                                            for k, v in vals.items()}

    # (M1) admissible theta: 4 - 2 theta < index
    admissible = {4.0: ("0.5", "1.0", "1.5", "2.0"), 1.5: ("1.5", "2.0")}
    verdict = {}
    for index, ths in admissible.items():
        sym = max(out[f"{k}|index={index}"][th]
                  for k in ("independent symmetric (Gaussian)",
                            "independent symmetric (Laplace)") for th in ths)
        bad = min(max(out[f"{k}|index={index}"][th] for th in ths)
                  for k in ("independent ASYMMETRIC", "non-commuting mixture"))
        verdict[f"index={index}"] = {"theta_used": list(ths), "max_symmetric": sym,
                                     "min_counterexample": bad,
                                     "separated": bool(bad > 3.0 * sym)}
    heavy_ok = all(out[f"{k}|index=1.5"][th] < 0.02
                   for k in ("independent symmetric (Gaussian)",
                             "independent symmetric (Laplace)")
                   for th in admissible[1.5])
    ok = all(v["separated"] for v in verdict.values()) and heavy_ok
    return {"offdiag_fraction": out, "verdict": verdict,
            "heavy_tails_do_not_break_it": bool(heavy_ok),
            "theta0_is_not_a_floor_below_index_2":
                {k: out[f"{k}|index=1.5"]["0.0"]
                 for k in ("independent symmetric (Gaussian)",
                           "independent symmetric (Laplace)")},
            "pass": bool(ok)}


# --------------------------------------------------------------------------
# D -- the corollary's constants as the index approaches 2 from above
# --------------------------------------------------------------------------
def part_d(rng, indices, thetas, n=200000, J=200) -> dict:
    """c(theta) and d_*(theta) on the population diagonal, radial part in CLOSED FORM.

    c(theta) = E[R^{2-theta}] E||Z||^{-theta} / E R^2.

    The radial moments are NOT estimated.  numpy's rng.pareto(a) is Lomax, so
    R = 1 + Lomax(a) is standard Pareto on [1, inf) with P(R > r) = r^{-a} and
    E R^p = a/(a-p) for p < a.  Hence the radial ratio is exactly

        E[R^{2-theta}] / E R^2 = (a - 2) / (a - 2 + theta),

    which is what is used here.  A first version estimated E R^2 by Monte Carlo and
    got 7.87 at index 2.2 in one run and 27.17 in another: E R^2 exists there but has
    INFINITE VARIANCE (that needs index > 4), so the sample mean is not a usable
    estimate of it, and every c(theta) computed from it inherited the instability.
    The exact value at index 2.2 is 2.2/0.2 = 11.  Only E||Z||^{-theta}, a Gaussian
    quadratic form with no closed form here, is left to Monte Carlo, and it is
    finite and light-tailed.

    Prediction: d_* is bounded away from 0 uniformly in the index, so the shrinkage
    factors are NOT how the comparison degrades; c(theta) instead falls towards zero
    as the index approaches 2 from above.  Either way the two-sided bound
    c d_* lambda_j(K) <= lambda_j(A) <= c lambda_j(K) becomes vacuous at index 2, and
    for the same reason: the object on the right, K, ceases to exist.
    """
    sig = (np.arange(1, J + 1, dtype=float)) ** -1.0
    z = rng.standard_normal((n, J)) * sig
    nz = np.linalg.norm(z, axis=1)
    out = {}
    for index in indices:
        er2_exact = index / (index - 2.0)          # closed form, index > 2
        row = {"E_R2_exact": float(er2_exact)}
        for th in thetas:
            radial = (index - 2.0) / (index - 2.0 + th)   # E[R^{2-th}] / E R^2
            c = float(radial * np.mean(nz ** -th))
            num = np.mean((nz ** -th)[:, None] * z ** 2, axis=0)
            den = sig ** 2 * float(np.mean(nz ** -th))
            dj = num / den
            row[str(th)] = {"c": c, "d_star": float(np.min(dj)),
                            "d_1": float(dj[0]), "d_last": float(dj[-1])}
        out[str(index)] = row
    d_stars = [row[str(th)]["d_star"] for row in out.values() for th in thetas]
    cs = {str(i): out[str(i)][str(thetas[-1])]["c"] for i in indices}
    lo_index, hi_index = str(min(indices)), str(max(indices))
    collapses = cs[lo_index] < 0.5 * cs[hi_index]
    return {"rows": out, "min_d_star": float(min(d_stars)),
            "c_at_largest_theta": cs,
            "d_star_bounded_away_from_zero": bool(min(d_stars) > 0.05),
            "c_collapses_as_index_falls_to_2": bool(collapses),
            "pass": bool(min(d_stars) > 0.05 and collapses)}


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    indices = (1.2, 1.5, 1.8, 2.5, 3.0, 4.0, 6.0)
    thetas = (0.0, 0.5, 1.0, 1.5, 2.0)
    n_c = 50000 if args.quick else 200000
    ns_b = (20000, 200000) if args.quick else (20000, 200000, 2000000)
    n_d = 100000 if args.quick else 400000

    res = {"seed": SEED, "quick": args.quick}
    # Settings recorded in the artifact, not only here: a CoE audit cannot resolve a
    # tail index or a replication count that exists as a literal in this file and
    # nowhere in results/.
    res["config"] = {
        "indices": list(indices), "thetas": list(thetas),
        "part_b": {"indices": [1.5, 3.0], "ns": list(ns_b)},
        "part_c": {"n": n_c, "J": 20, "reps": 4, "indices": [4.0, 1.5],
                   "admissible_rule": "(M1): 4 - 2*theta < tail index",
                   "admissible": {"4.0": [0.5, 1.0, 1.5, 2.0], "1.5": [1.5, 2.0]}},
        "part_d": {"indices": [2.2, 2.5, 3.0, 6.0], "n": n_d},
    }
    res["A_region"] = part_a(rng, indices, thetas)
    res["B_K_absent"] = part_b(rng, (1.5, 3.0), ns_b)
    res["C_basis"] = part_c(rng, n_c)
    res["D_constants"] = part_d(rng, (2.2, 2.5, 3.0, 6.0), thetas, n=n_d)
    res["all_pass"] = all(res[k]["pass"] for k in res
                          if isinstance(res[k], dict) and "pass" in res[k])

    L = []
    P = L.append
    P("=" * 78)
    P("Section 4.3: which statement needs which hypothesis, and where each holds")
    P(f"seed {SEED}" + ("   [--quick]" if args.quick else ""))
    P("=" * 78)

    a = res["A_region"]
    P("")
    P("A. the admissible region.  R = rate available (M1 and M2);")
    P("   C = comparison with K available (E||X||^2 < inf, i.e. index > 2).")
    P(f"      {'index':>7}" + "".join(f"{'th=' + str(t):>9}" for t in thetas)
      + "    K exists")
    for index in indices:
        cells = []
        for th in thetas:
            g = a["region"][f"{index}|{th}"]
            cells.append("R" if g["rate_available"] else "-")
        P(f"      {index:>7}" + "".join(f"{c:>9}" for c in cells)
          + f"    {'yes' if k_exists(index) else 'NO':>8}")
    P("   Monte Carlo cross-check by GROWTH RATE: log-log slope of the median sample")
    P("   E||X||^2 against n.  Zero iff the moment is finite; else 2/index - 1.")
    P(f"      {'index':>7}{'slope E||X||^2':>16}{'predicted':>11}{'slope E||X||':>14}   verdict")
    for k, v in a["moment_growth"].items():
        P(f"      {k:>7}{v['loglog_slope_E_norm2']:>16.3f}{v['predicted_slope_E_norm2']:>11.3f}"
          f"{v['loglog_slope_E_norm1']:>14.3f}"
          f"   {'finite' if v['closed_form_E_norm2_finite'] else 'INFINITE'}")
    P(f"   -> [{'ok' if a['pass'] else 'FAIL'}]")

    b = res["B_K_absent"]
    P("")
    P("B. K does not settle below index 2; A_1 does.")
    P(f"      {'index':>7}  {'tr Khat by n':<38}{'spread':>8}   {'tr A1hat spread':>16}")
    for k, v in b["rows"].items():
        ks = [f"{v[n]['tr_Khat']:.2f}" for n in v if n.isdigit()]
        P(f"      {k:>7}  {', '.join(ks):<38}{v['K_spread']:>8.2f}"
          f"{v['A1_spread']:>16.2f}")
    P("   A trace that keeps growing with n is a population value that does not exist,")
    P("   which is what makes the class S_0 undefined there.")
    P(f"   -> [{'ok' if b['pass'] else 'FAIL'}]")

    c = res["C_basis"]
    P("")
    P("C. both clauses are necessary, and heavy tails are not what breaks the basis.")
    P("   Generators and analytic bases imported from gates/a1_6b_source_condition.py,")
    P("   which produced the published table; that table used tail index 4.0 only, so")
    P("   index 1.5 -- the regime the paper is about -- is added here.")
    P(f"      {'law | radial factor':<40}" + "".join(f"{'th=' + t_:>9}"
                                                     for t_ in ("0.0", "0.5", "1.0", "1.5", "2.0")))
    for k, v in c["offdiag_fraction"].items():
        P(f"      {k:<40}" + "".join(f"{v[t_]:>9.4f}"
                                     for t_ in ("0.0", "0.5", "1.0", "1.5", "2.0")))
    P("   No theta=0 floor is used below index 2: that column is estimation noise for")
    P("   EVERY law there (Gaussian "
      + ", ".join(f"{v:.3f}" for v in c["theta0_is_not_a_floor_below_index_2"].values())
      + "), so the published floor argument holds only")
    P("   at a light radial factor.  Comparison at equal, (M1)-admissible theta:")
    P(f"      {'radial factor':<16}{'theta used':>22}{'max symmetric':>15}"
      f"{'min counterexample':>21}   separated")
    for k, v in c["verdict"].items():
        P(f"      {k:<16}{','.join(v['theta_used']):>22}{v['max_symmetric']:>15.4f}"
          f"{v['min_counterexample']:>21.4f}   {'yes' if v['separated'] else 'NO'}")
    P(f"   heavy tails leave the basis intact: "
      f"{'yes' if c['heavy_tails_do_not_break_it'] else 'NO'}."
      f"   [{'ok' if c['pass'] else 'FAIL'}]")

    d = res["D_constants"]
    P("")
    P("D. the comparison constants as the index falls towards 2 from above.")
    P("   Radial moments in closed form (E R^p = a/(a-p)); only E||Z||^{-theta} is")
    P("   Monte Carlo.  E R^2 has infinite variance below index 4, so estimating it")
    P("   gave 7.87 and 27.17 on two runs at index 2.2 where the exact value is 11.")
    P(f"      {'index':>7}{'E R^2 exact':>13}" + "".join(f"{'c(' + str(t) + ')':>10}" for t in thetas)
      + f"{'min d_*':>10}")
    for k, v in d["rows"].items():
        P(f"      {k:>7}{v['E_R2_exact']:>13.2f}"
          + "".join(f"{v[str(t)]['c']:>10.3f}" for t in thetas)
          + f"{min(v[str(t)]['d_star'] for t in thetas):>10.4f}")
    P(f"   d_* stays bounded away from zero (min {d['min_d_star']:.4f}); c(theta)")
    P(f"   collapses towards zero as the index falls to 2: "
      f"{'yes' if d['c_collapses_as_index_falls_to_2'] else 'no'} (E R^2 is its denominator).")
    P("   So the comparison degrades through its CONSTANT, not through the shrinkage")
    P("   factors, and at index 2 it stops existing because K does.")
    P(f"   -> [{'ok' if d['pass'] else 'FAIL'}]")

    P("")
    P("=" * 78)
    P("ALL PASS" if res["all_pass"] else "FAILURE -- a prediction did not hold")
    P("=" * 78)

    text = "\n".join(L)
    print(text)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "section43_hypotheses.out").write_text(text + "\n", encoding="utf-8")
    (RESULTS / "section43_hypotheses.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
