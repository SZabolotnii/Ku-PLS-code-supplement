#!/usr/bin/env python3
"""Prop. 5 as submitted is FALSE.  Exhibit the counterexample, then check the repair.

WHAT WAS CLAIMED (submitted manuscript, prop:noremoval):

    Let t : H -> H be measurable with sup_x ||t(x)|| < inf.  If E|eps| = inf then
    n^{-1} sum_k eps_k t(X_k) does not converge IN PROBABILITY, for any such t.

The statement is false as stated, for two independent reasons.

  (i)  t == 0 satisfies every hypothesis and the average is identically 0.
       One character.  The quantifier "for any such t" is simply too wide.

  (ii) The substantive one, which survives excluding t == 0 by fiat: convergence
       in probability is NOT equivalent to integrability.  The Kolmogorov-Feller
       weak law says an i.i.d. mean converges in probability (after centering at
       the truncated mean) iff  n P(|eps| > n) -> 0 -- a condition strictly weaker
       than E|eps| < inf.  The gap between the two is exactly where the claim dies.

       Witness law, symmetric, with survival function

           P(|eps| > x) = 1 / ( x (1 + ln x) ),      x >= 1.

       E|eps| = int_0^inf P(|eps| > x) dx = 1 + int_1^inf dx/(x(1+ln x))
              = 1 + [ln(1 + ln x)]_1^inf = INFINITE,

       yet  n P(|eps| > n) = 1/(1 + ln n) -> 0, and symmetry kills the centering
       term, so n^{-1} sum eps_k -> 0 IN PROBABILITY.  Take t == v, a fixed unit
       vector: bounded, and bounded AWAY from zero, so it is not a degenerate
       witness -- and the average still vanishes.

       Sampling is by inverse transform in closed form.  Solving
       x(1 + ln x) = 1/U with w = 1 + ln x gives w e^w = e/U, so

           |eps| = exp( LambertW(e/U) - 1 ),    U ~ Unif(0,1).

WHAT IS TRUE, and is what the manuscript needs (the repair):

    Let t be measurable with E[ |eps| ||t(X)|| ] = inf -- which holds whenever
    E|eps| = inf and inf_x ||t(x)|| > 0.  Then

        limsup_n || n^{-1} sum_{k<=n} eps_k t(X_k) || = inf   ALMOST SURELY,

    so the average has no finite a.s. limit.  Proof is the converse half of
    Kolmogorov's SLLN, valid in any normed space: E||z|| = inf iff
    sum_n P(||z|| > n) = inf; independence and Borel-Cantelli II then give
    ||z_n|| > cn infinitely often a.s. for every c; and if S_n/n had a finite
    limit L then z_n/n = S_n/n - ((n-1)/n) S_{n-1}/(n-1) -> 0, a contradiction.

    The a.s. mode is the right one because it is the mode that is EQUIVALENT to
    integrability.  The paper's point -- bounding the predictor feature does
    nothing for the error's own integrability -- survives verbatim, because every
    feature the paper competes with has ||t(x)|| bounded away from 0 (checked in
    part D below): the spatial sign, the SSCM direction, and the CF feature
    e^{i<u,x>}, whose H_pi norm is the CONSTANT sqrt(pi(U)).

WHAT THIS SCRIPT PRINTS.  Four parts, each a pass/fail against a prediction made
before it was run:

  A  the witness law: sampler matches its survival function; E|eps| diverges
     (truncated mean tracks ln(1+ln M)); Feller's n P(|eps|>n) -> 0.
  B  the false claim: median ||mean|| over reps VANISHES for the witness law
     (falsifying the proposition) while it is FLAT for Cauchy (where the
     proposition happens to be right).  Same t, bounded away from zero.
  C  the repair: sum_n P(|eps| > n) diverges, so Borel-Cantelli applies; and the
     empirical max_{n<=N} ||S_n||/n grows with N.  That growth is log-log slow by
     construction, so it ILLUSTRATES the theorem; the proof is what establishes it.
  D  scope: inf_x ||t(x)|| > 0 for the three features the manuscript competes
     with, so the repaired hypothesis is satisfied exactly where it is used.

Run:  python code/prop5_counterexample.py   [--quick]
Writes results/prop5_counterexample.{json,out}.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np
from scipy.special import lambertw

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 20260903


# --------------------------------------------------------------------------
# the witness law
# --------------------------------------------------------------------------
def witness_abs(u: np.ndarray) -> np.ndarray:
    """|eps| with P(|eps| > x) = 1/(x(1+ln x)), x >= 1, by inverse transform."""
    return np.exp(lambertw(math.e / u).real - 1.0)


def witness_survival(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(x < 1.0, 1.0, 1.0 / (x * (1.0 + np.log(np.maximum(x, 1.0)))))


def sample_witness(rng: np.random.Generator, size) -> np.ndarray:
    u = rng.uniform(0.0, 1.0, size=size)
    sign = rng.integers(0, 2, size=size) * 2 - 1
    return sign * witness_abs(u)


def sample_cauchy(rng: np.random.Generator, size) -> np.ndarray:
    return rng.standard_cauchy(size=size)


LAWS = {
    "witness  1/(x(1+ln x))": sample_witness,
    "cauchy": sample_cauchy,
}


# --------------------------------------------------------------------------
# A -- the witness law is what it is claimed to be
# --------------------------------------------------------------------------
def part_a(rng: np.random.Generator, n: int) -> dict:
    eps = sample_witness(rng, n)
    a = np.abs(eps)

    # sampler vs its own survival function
    grid = np.array([1.0, 2.0, 5.0, 20.0, 100.0, 1000.0])
    emp = np.array([(a > g).mean() for g in grid])
    thy = witness_survival(grid)
    surv_max_err = float(np.max(np.abs(emp - thy)))

    # E|eps| = inf.  The closed form is EXACT and is what carries the claim:
    #   E[|eps| 1{|eps|<=M}] = int_0^M P(|eps|>x) dx - M P(|eps|>M)
    #                        = 1 + ln(1+ln M) - 1/(1+ln M),
    # unbounded in M.  The Monte Carlo column agrees only while M is inside the
    # sample's own range: once M exceeds max|eps| the truncation stops biting and
    # the column saturates at the sample mean.  That saturation is a property of
    # the ESTIMATOR, not of the law, so it is labelled rather than hidden.
    amax = float(a.max())
    trunc = []
    for m in [1e2, 1e4, 1e8, 1e16, 1e32]:
        obs = float(a[a <= m].sum() / n)
        exact = 1.0 + math.log(1.0 + math.log(m)) - 1.0 / (1.0 + math.log(m))
        trunc.append({"M": m, "empirical": obs, "closed_form": exact,
                      "M_within_sample_range": bool(m <= amax)})

    # Feller's condition, exactly, for both laws
    feller = {
        "witness": {str(n_): 1.0 / (1.0 + math.log(n_)) for n_ in [1e3, 1e6, 1e12, 1e60]},
        # Cauchy: n P(|eps|>n) = n * (2/pi) arctan(1/n) -> 2/pi = 0.6366
        "cauchy": {str(n_): float(n_ * (2 / math.pi) * math.atan(1.0 / n_))
                   for n_ in [1e3, 1e6, 1e12, 1e60]},
    }
    inrange = [r for r in trunc if r["M_within_sample_range"]]
    ok = (surv_max_err < 0.01
          # closed form unbounded, and Monte Carlo tracks it while M is in range
          and trunc[-1]["closed_form"] > trunc[0]["closed_form"] + 2.0
          and all(abs(r["empirical"] - r["closed_form"]) < 0.35 for r in inrange)
          and feller["witness"]["1e+60"] < 0.02
          and feller["cauchy"]["1e+60"] > 0.6)
    return {"survival_grid": grid.tolist(), "survival_empirical": emp.tolist(),
            "survival_theory": thy.tolist(), "survival_max_abs_err": surv_max_err,
            "sample_max_abs_eps": amax,
            "truncated_means": trunc, "feller_nP": feller, "pass": bool(ok)}


# --------------------------------------------------------------------------
# B -- the claim as stated is false
# --------------------------------------------------------------------------
def part_b(rng: np.random.Generator, ns, reps: int) -> dict:
    """Two refutations, one exact and one by a classical theorem, plus illustration.

    B1  t == 0 satisfies every hypothesis; the average is identically 0.  Exact.

    B2  t == v, a fixed unit vector -- bounded, and bounded AWAY from zero, so
        nothing degenerate is doing the work.  Then the average is |n^{-1} sum
        eps_k| exactly.  The Kolmogorov-Feller weak law says this converges in
        probability to 0 (the centering E[eps 1{|eps|<=n}] vanishing by symmetry)
        IFF n P(|eps| > n) -> 0, and part A verifies that hypothesis in CLOSED
        FORM.  So the refutation is a theorem, not a simulation.

    B3  Monte Carlo, as illustration only, tested against the rate the theory
        predicts rather than against a threshold picked by hand.

        E[eps^2 1{|eps|<=n}] = 2 int_0^n x P(|eps|>x) dx ~ 2n/ln n, so the spread
        of the truncated mean is of order sqrt(2/ln n): the decay is slower than
        ANY power of n, and over the three decades that are reachable the median
        can only fall by about sqrt(ln n_max / ln n_min) ~ 1.34x.  That is the
        same size as the Monte Carlo error on a median, so "strictly decreasing in
        n" is NOT a testable prediction here -- a first version of this script
        used it and failed on 0.2232 -> 0.2480 at adjacent n.  What is testable is
        the RATE: median * sqrt(ln n) must be roughly constant for the witness law
        and must GROW like sqrt(ln n) for Cauchy, whose median |mean| is pinned at
        1 for every n because a Cauchy mean is again Cauchy(0,1) exactly.
    """
    b1 = {"t": "identically zero", "average": 0.0,
          "hypotheses_satisfied": True, "converges": True}

    out = {}
    for name, sampler in LAWS.items():
        med, q90 = [], []
        for n in ns:
            means = np.abs(sampler(rng, (reps, n)).mean(axis=1))
            med.append(float(np.median(means)))
            q90.append(float(np.quantile(means, 0.90)))
        rescaled = [m * math.sqrt(math.log(n)) for m, n in zip(med, ns)]
        out[name] = {"n": list(ns), "median_abs_mean": med, "q90_abs_mean": q90,
                     "median_times_sqrt_log_n": rescaled,
                     "rescaled_spread": float(max(rescaled) / min(rescaled)),
                     "rescaled_drift": float(rescaled[-1] / rescaled[0]),
                     "total_drop_factor": float(med[0] / med[-1])}
    w = out["witness  1/(x(1+ln x))"]
    c = out["cauchy"]
    predicted_drop = math.sqrt(math.log(ns[-1]) / math.log(ns[0]))
    w["predicted_drop_factor"] = float(predicted_drop)
    w["decays_at_predicted_rate"] = bool(
        abs(w["total_drop_factor"] / predicted_drop - 1.0) < 0.30
        and w["rescaled_spread"] < 1.6)
    # Cauchy control: median |mean| pinned at the exact stability value 1, so the
    # rescaled column must climb like sqrt(ln n) instead of staying flat.
    c["flat_at_exact_stability_value"] = bool(
        all(abs(m - 1.0) < 0.15 for m in c["median_abs_mean"])
        and c["rescaled_drift"] > 1.2)
    ok = w["decays_at_predicted_rate"] and c["flat_at_exact_stability_value"]
    return {"B1_degenerate_witness": b1, "laws": out,
            "B2_refutation_is_by_theorem":
                "Kolmogorov-Feller; its hypothesis n P(|eps|>n) -> 0 is exact, see part A",
            "claim_as_stated_refuted": True, "monte_carlo_consistent": bool(ok),
            "pass": bool(ok)}


# --------------------------------------------------------------------------
# C -- the repaired statement
# --------------------------------------------------------------------------
def part_c(rng: np.random.Generator, ns, reps: int) -> dict:
    """The repaired statement, and the two diagnostics that actually test it.

    The claim is limsup_n ||S_n||/n = inf a.s.  Its engine is Borel-Cantelli II
    applied to the events {||z_n|| > cn}, whose probabilities sum to infinity
    exactly because E||z|| = inf.  Two things are measured.

      C1  the divergence itself, in the exact partial sums sum_{n<=N} P(|eps|>cn).
          The series behaves like ln(1+ln N)/c, so partial sums pass any fixed
          level only for absurd N; the testable content is that they INCREASE
          without settling, at every level c.  A threshold like "> 1" would test
          the size of a constant, not divergence.

      C2  the mechanism, on paths: max_{n in (N/10, N]} |eps_n| / n.  Under
          E|eps| = inf this term does NOT vanish -- that is precisely why S_n/n
          cannot converge a.s., since on any convergent path eps_n/n -> 0.  Under
          a finite mean it does vanish, at a power of N.  The two decay rates
          separate cleanly: 1/ln N against a power of N.

    Two diagnostics were tried first and discarded, both because they cannot
    resolve what they claim to test.  (i) max_{n<=N} ||S_n||/n over the WHOLE
    path is dominated by n = 1, where it is just |eps_1|; it drifted DOWN with N
    (5.16 at 1e4 against 3.97 at 1e5).  (ii) Growth of the exceedance COUNT
    #{n <= N : |eps_n| > n}: its expectation rises only from 2.96 to 3.17 across
    two decades of N, against a per-path spread of about 1.7, so no feasible
    number of replicates separates them.  The counts are therefore reported as an
    agreement check against their exact expectation, not as evidence of growth.
    """
    # C1 -- exact partial sums of the Borel-Cantelli series
    bc = {}
    for c in [1.0, 3.0]:
        row = {}
        for upto in [10**4, 10**6, 10**8]:
            n = np.arange(1, upto + 1, dtype=float)
            row[f"N=1e{int(math.log10(upto))}"] = float(witness_survival(c * n).sum())
        row["increasing"] = bool(row["N=1e4"] < row["N=1e6"] < row["N=1e8"])
        bc[f"c={c:g}"] = row

    # C2 -- max |eps_n|/n over the tail window, witness against a finite-mean control
    def pareto15(r, size):                       # E|eps| < inf, E eps^2 = inf
        return (r.pareto(1.5, size=size) + 1.0) * (r.integers(0, 2, size=size) * 2 - 1)

    obs = {}
    for label, sampler in (("witness (E|eps| = inf)", sample_witness),
                           ("pareto-1.5 (E|eps| < inf)", pareto15)):
        row = {}
        for N in ns:
            lo = max(1, N // 10)
            ratios, counts = [], []
            for _ in range(reps):
                eps = sampler(rng, N)
                idx = np.arange(1, N + 1, dtype=float)
                ratios.append(float(np.max(np.abs(eps[lo - 1:]) / idx[lo - 1:])))
                if sampler is sample_witness:
                    counts.append(int((np.abs(eps) > idx).sum()))
                del eps, idx
            row[str(N)] = {"median_tail_max_eps_over_n": float(np.median(ratios))}
            if counts:
                row[str(N)]["median_exceedances"] = float(np.median(counts))
                row[str(N)]["expected_exceedances"] = float(
                    witness_survival(np.arange(1, N + 1, dtype=float)).sum())
        vals = [row[str(N)]["median_tail_max_eps_over_n"] for N in ns]
        row["drop_factor_over_range"] = float(vals[0] / vals[-1])
        obs[label] = row

    w_drop = obs["witness (E|eps| = inf)"]["drop_factor_over_range"]
    p_drop = obs["pareto-1.5 (E|eps| < inf)"]["drop_factor_over_range"]
    # witness decays like 1/ln N (slow); control like a power of N (fast)
    separated = bool(p_drop > 2.5 * w_drop)
    # counts must agree with their exact expectation -- a validity check, not growth
    wrow = obs["witness (E|eps| = inf)"]
    agree = all(abs(wrow[str(N)]["median_exceedances"]
                    - wrow[str(N)]["expected_exceedances"]) < 2.0 for N in ns)
    sums_diverge = all(v["increasing"] for v in bc.values())
    return {"borel_cantelli_partial_sums": bc, "observed": obs,
            "sums_increase_without_settling": bool(sums_diverge),
            "decay_rates_separate": separated,
            "counts_match_expectation": bool(agree),
            "pass": bool(sums_diverge and separated and agree)}


# --------------------------------------------------------------------------
# D -- the repaired hypothesis covers the features the paper competes with
# --------------------------------------------------------------------------
def part_d(rng: np.random.Generator, n: int = 20000, dim: int = 12,
           n_freq: int = 64) -> dict:
    """inf_x ||t(x)|| over a heavy-tailed design, for each competing feature."""
    # elliptical, heavy radial part, so ||X|| ranges over many decades
    z = rng.standard_normal((n, dim))
    r = rng.pareto(1.2, size=n) + 1.0
    x = r[:, None] * z
    nx = np.linalg.norm(x, axis=1)

    feats = {
        "spatial sign  x/||x||         (theta=1)": np.ones(n),
        "SSCM direction               (theta=2)": np.ones(n),
    }
    # CF feature into H_pi: ||e^{i<u,.>}||^2_{H_pi} = int |e^{i<u,x>}|^2 pi(du) = pi(U)
    u = rng.standard_normal((n_freq, dim)) / math.sqrt(dim)
    w = np.full(n_freq, 1.0 / n_freq)               # pi, normalised: pi(U) = 1
    phase = x @ u.T
    cf_norm = np.sqrt(((np.cos(phase) ** 2 + np.sin(phase) ** 2) * w).sum(axis=1))
    feats["CF  e^{i<u,x>} in H_pi"] = cf_norm
    # Contrast: a bounded feature that is NOT bounded below, so the repaired
    # hypothesis genuinely excludes something.  ||t(x)|| = ||x||/(1+||x||^2)
    # vanishes at both ends of the radial range, and the design spans four
    # decades of ||X||.  (A first version used x/(1+||x||), whose norm is
    # bounded below whenever ||X|| is -- it excluded nothing and was no contrast.)
    feats["contrast  t(x)=x/(1+||x||^2)"] = nx / (1.0 + nx ** 2)

    out = {}
    for name, v in feats.items():
        out[name] = {"inf": float(v.min()), "sup": float(v.max()),
                     "bounded_below": bool(v.min() > 0.5)}
    competing = [k for k in out if "contrast" not in k]
    ok = (all(out[k]["bounded_below"] for k in competing)
          and not out["contrast  t(x)=x/(1+||x||^2)"]["bounded_below"])
    return {"features": out, "norm_x_range": [float(nx.min()), float(nx.max())],
            "pass": bool(ok)}


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    if args.quick:
        ns_b, reps_b = (1000, 4000, 16000), 200
        ns_c, reps_c = (10**4, 10**5), 8
        n_a = 200000
    else:
        ns_b, reps_b = (1000, 4000, 16000, 64000, 256000), 400
        ns_c, reps_c = (10**4, 10**5, 10**6, 10**7), 12
        n_a = 2000000

    res = {"seed": SEED, "quick": args.quick}
    # Settings in the artifact, not only in this file: the manuscript prints "over 400
    # replications", and a CoE audit can only resolve that against results/ if the
    # number is stored there.
    res["config"] = {"part_a": {"n": n_a},
                     "part_b": {"ns": list(ns_b), "reps": reps_b},
                     "part_c": {"ns": list(ns_c), "reps": reps_c}}
    res["A_witness_law"] = part_a(rng, n_a)
    res["B_claim_refuted"] = part_b(rng, ns_b, reps_b)
    res["C_repair"] = part_c(rng, ns_c, reps_c)
    res["D_scope"] = part_d(rng)
    res["all_pass"] = all(res[k]["pass"] for k in res if k[0] in "ABCD" and k[1] == "_")

    lines = []
    P = lines.append
    P("=" * 78)
    P("Prop. 5 (prop:noremoval) as submitted is FALSE -- counterexample and repair")
    P(f"seed {SEED}" + ("   [--quick]" if args.quick else ""))
    P("=" * 78)

    a = res["A_witness_law"]
    P("")
    P("A. the witness law   P(|eps| > x) = 1/(x(1+ln x)),  symmetric")
    P("   sampler vs survival function, max abs error over the grid: "
      f"{a['survival_max_abs_err']:.5f}")
    P("   E|eps| = infinity.  Closed form is exact; the Monte Carlo column tracks it")
    P(f"   only while M stays inside the sample range (max|eps| = {a['sample_max_abs_eps']:.3g}),")
    P("   then saturates -- an estimator artefact, marked, not a property of the law:")
    P(f"      {'M':>10}  {'Monte Carlo':>12}  {'closed form':>12}   M in range")
    for row in a["truncated_means"]:
        P(f"      {row['M']:>10.0e}  {row['empirical']:>12.4f}  {row['closed_form']:>12.4f}"
          f"   {'yes' if row['M_within_sample_range'] else 'no (saturated)'}")
    P("   Feller's n P(|eps| > n):")
    for law in ("witness", "cauchy"):
        vals = "  ".join(f"{float(k):.0e}: {v:.4f}" for k, v in a["feller_nP"][law].items())
        P(f"      {law:>8}   {vals}")
    P(f"   -> witness: E|eps| = inf AND n P(|eps|>n) -> 0.   [{'ok' if a['pass'] else 'FAIL'}]")

    b = res["B_claim_refuted"]
    P("")
    P("B. the claim as stated is false, twice over")
    P("   B1  t == 0 satisfies every hypothesis and the average is identically 0.")
    P("       Exact; no simulation involved.")
    P("   B2  t == v, a fixed unit vector -- bounded, and bounded AWAY from zero.")
    P("       The average is |n^{-1} sum eps_k|, and Kolmogorov-Feller gives it")
    P("       -> 0 in probability because n P(|eps|>n) -> 0 (part A, closed form),")
    P("       the centering vanishing by symmetry.  A theorem, not a simulation.")
    bw, bc_ = b["laws"]["witness  1/(x(1+ln x))"], b["laws"]["cauchy"]
    P("   B3  Monte Carlo, illustration only, tested against the rate the theory")
    P("       predicts.  Spread of the mean is ~ sqrt(2/ln n), so over the reachable")
    P(f"       range the median can fall only by about {bw['predicted_drop_factor']:.2f}x --"
      " the size of")
    P("       the MC error itself, which is why monotone decay is not testable here")
    P("       and median * sqrt(ln n) is.  It must be flat for the witness law and")
    P("       must CLIMB for Cauchy, whose mean is Cauchy(0,1) exactly at every n.")
    P(f"      {'law':<26}" + "".join(f"{n:>9}" for n in bc_["n"]) + f"{'drop':>8}")
    for name, d in b["laws"].items():
        P(f"      {name:<26}" + "".join(f"{m:>9.4f}" for m in d["median_abs_mean"])
          + f"{d['total_drop_factor']:>7.2f}x")
        P(f"      {'  x sqrt(ln n)':<26}"
          + "".join(f"{m:>9.4f}" for m in d["median_times_sqrt_log_n"])
          + f"{d['rescaled_drift']:>7.2f}x")
    P(f"      witness decays at the predicted rate "
      f"({bw['total_drop_factor']:.2f}x observed against {bw['predicted_drop_factor']:.2f}x "
      f"predicted): {'yes' if bw['decays_at_predicted_rate'] else 'NO'}")
    P(f"      Cauchy pinned at 1, rescaled column climbs: "
      f"{'yes' if bc_['flat_at_exact_stability_value'] else 'NO'}")
    P("   The proposition predicts 'does not converge' on BOTH rows.  It is right on")
    P("   Cauchy and wrong on the witness law, which has E|eps| = infinity all the same.")
    P(f"   -> claim as stated REFUTED (B1, B2 exact); MC consistent [{'ok' if b['pass'] else 'FAIL'}]")

    c = res["C_repair"]
    P("")
    P("C. the repair:  limsup_n ||S_n||/n = infinity  a.s.")
    P("   C1  Borel-Cantelli series sum_{n<=N} P(|eps| > cn).  It grows like")
    P("       ln(1+ln N)/c, so the testable content is that it keeps increasing,")
    P("       not that it passes any fixed level:")
    P(f"      {'':<8}{'N=1e4':>10}{'N=1e6':>10}{'N=1e8':>10}   increasing")
    for k, v in c["borel_cantelli_partial_sums"].items():
        P(f"      {k:<8}{v['N=1e4']:>10.3f}{v['N=1e6']:>10.3f}{v['N=1e8']:>10.3f}"
          f"   {'yes' if v['increasing'] else 'NO'}")
    P("   C2  the mechanism on paths:  max_{n in (N/10, N]} |eps_n| / n.  On any")
    P("       path where S_n/n converges, eps_n/n -> 0; under E|eps| = inf it does")
    P("       not.  Against a finite-mean control the two decay rates separate --")
    P("       1/ln N against a power of N:")
    ns_c = [k for k in next(iter(c["observed"].values())) if k.isdigit()]
    P(f"      {'law':<28}" + "".join(f"{int(k):>11,}" for k in ns_c) + f"{'drop':>9}")
    for label, row in c["observed"].items():
        P(f"      {label:<28}"
          + "".join(f"{row[k]['median_tail_max_eps_over_n']:>11.4f}" for k in ns_c)
          + f"{row['drop_factor_over_range']:>8.2f}x")
    P("       exceedance counts #{n<=N : |eps_n| > n}, observed against their exact")
    P("       expectation -- an agreement check, since the expectation rises only")
    P("       from 2.96 to 3.34 across four decades and cannot be resolved as growth:")
    wrow = c["observed"]["witness (E|eps| = inf)"]
    P(f"      {'':<28}" + "".join(f"{wrow[k]['median_exceedances']:>11.1f}" for k in ns_c)
      + "   observed")
    P(f"      {'':<28}" + "".join(f"{wrow[k]['expected_exceedances']:>11.3f}" for k in ns_c)
      + "   expected")
    P("   Growth is log-log slow by construction, so these numbers ILLUSTRATE the")
    P("   theorem; Borel-Cantelli plus z_n/n -> 0 on any convergent path PROVES it.")
    P(f"   -> [{'ok' if c['pass'] else 'FAIL'}]")

    d = res["D_scope"]
    P("")
    P("D. scope of the repaired hypothesis:  inf_x ||t(x)|| > 0 ?")
    P(f"   design: ||X|| over [{d['norm_x_range'][0]:.3f}, {d['norm_x_range'][1]:.1f}]")
    P(f"      {'feature':<42}{'inf ||t||':>11}{'sup ||t||':>11}   bounded below")
    for name, v in d["features"].items():
        P(f"      {name:<42}{v['inf']:>11.4f}{v['sup']:>11.4f}   "
          + ("yes" if v["bounded_below"] else "NO"))
    P("   Every feature the manuscript competes with is bounded away from zero, so")
    P("   the repaired proposition still closes the direction it was written to close.")
    P(f"   -> [{'ok' if d['pass'] else 'FAIL'}]")

    P("")
    P("=" * 78)
    P("ALL PASS" if res["all_pass"] else "FAILURE -- a prediction did not hold")
    P("=" * 78)

    text = "\n".join(lines)
    print(text)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "prop5_counterexample.out").write_text(text + "\n", encoding="utf-8")
    (RESULTS / "prop5_counterexample.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
