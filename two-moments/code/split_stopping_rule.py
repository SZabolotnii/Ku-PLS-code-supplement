#!/usr/bin/env python3
"""Repair (b): a stopping rule whose threshold does not depend on the stopping path.

THE DEFECT.  An earlier version of the rule was

    mhat = min{ m : ||rhat - Ahat beta_m|| <= tau sqrt(tr Vhat(m)/n) },
    tr Vhat(m) = n^{-1} sum_k (Y_k - <beta_m, X_k>)^2 ||X_k||^{2-2theta},

and the proof of Prop. 8 (submitted numbering) treats it as a substitution into
BCT's chain, whose threshold is a DETERMINISTIC constant.  It is not one.  The
threshold is recomputed from the residuals of the very iterate whose stopping it
decides, so it is a random FUNCTION of m, and step (f) of the chain -- the bound
on |Q'_mhat(0)| -- would need it uniformly over m.  A Chebyshev bound uniform in m
costs E[eps^4 ||X||^{4-4theta}]: the fourth moment the paper exists to remove.
There is also a bias: the residual is eps_k + <beta - beta_m, X_k>, not eps_k, and
it shrinks as m grows, so the threshold falls exactly when the rule needs it not to.

THE REPAIR.  Estimate tr V_theta ONCE, off the stopping path, from a pilot:

    D1 (a fraction rho of the sample):  beta_0 = CG(D1, m0), m0 FIXED in advance;
    D2 (the rest):                      tr Vhat_split
                                          = |D2|^{-1} sum_{k in D2} (Y_k - <beta_0, X_k>)^2 w_k;
    then run CG on the full sample and stop at the FIXED threshold
    tau sqrt(tr Vhat_split / n).

MOMENT ACCOUNTING, which is the point.  Write d = beta - beta_0 and w = ||X||^{2-2theta}.
Conditionally on D1, tr Vhat_split is an average of i.i.d. terms

    (eps + <d,X>)^2 w  =  eps^2 w  +  2 eps <d,X> w  +  <d,X>^2 w.

  * eps^2 w has mean tr V_theta, finite by (M2).  Khinchin's weak law needs the
    FIRST moment only, so the average converges in probability.  No fourth moment.
  * <d,X>^2 w <= ||d||^2 ||X||^{4-2theta}, whose mean is finite by (M1); it is
    O_P(||d||^2) -> 0.
  * the cross term is bounded by 2||d|| E[|eps| ||X||^{3-2theta}], and by
    Cauchy-Schwarz E[|eps| ||X||^{3-2theta}] <= sqrt(tr V_theta) sqrt(E||X||^{4-2theta}),
    the product of (M2) and (M1).

So tr Vhat_split -> tr V_theta in probability under EXACTLY (M1)+(M2), the two
conditions Prop. 8 already assumes.  No new hypothesis enters, and the threshold is
one number: on the event {|tr Vhat_split / tr V_theta - 1| <= 1/2}, whose probability
tends to one, it sits within a constant factor of the deterministic scale BCT's step
(f) uses, and the substitution is legitimate.  Note what is NOT claimed: the
threshold is not independent of the sample (CG runs on data that include D2).  What
the split buys is that it does not depend on m, which is what the proof needs.

WHAT THIS SCRIPT TESTS.  Five parts, each against a prediction made before running:

  A  the defect is real: tr Vhat(m) falls with m along the stopping path, and the
     fall is larger than the sampling error, so the threshold chases the iterate.
  B  consistency of tr Vhat_split under (M1)+(M2), INCLUDING a design with no
     fourth moment; and the bias/variance split above, term by term.
  C  the price: median slope error, split rule against the self-referential rule
     and against an oracle that knows tr V_theta, across tail indices.
  D  does the rescaling gain survive?  The old OPERATOR-scale rule is the thing
     the 2026-08-21 rescale beat by 1.78x at theta=2; that comparison is re-run
     under the split rule, since a gain that only exists for a rule with a broken
     proof is not a gain.
  E  the choice of rho and m0: how much does the answer move?

Run:  python code/split_stopping_rule.py [--quick]
Writes results/split_stopping_rule.{json,out}.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from twomoments import (Design, operator, cg, discrepancy_stop,  # noqa: E402
                        discrepancy_stop_operator_scale)

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 20260904

# THE canonical design of this track (twomoments.Design defaults), so that every
# number here is comparable with a1_5_consistency.json and its OLDRULE twin.  A
# private design would make the part-D comparison meaningless.
DESIGN = Design(J=20, decay=1.0, sigma=0.5)
DIM = DESIGN.J


# --------------------------------------------------------------------------
# design
# --------------------------------------------------------------------------
def make_beta() -> np.ndarray:
    return DESIGN.beta


def draw(rng, n, tail_index, beta=None):
    """The track's canonical draw.  X = R Z, Z ~ N(0, diag(scale^2)), R Pareto.

    E||X||^p < inf iff p < index, exactly.  At theta = 1 the two conditions are
    (M1) E||X||^2 < inf, i.e. index > 2, and (M2) E eps^2 < inf, always true for
    the Gaussian error.  A fourth moment exists only for index > 4, so an index in
    (2, 4) is the window where the paper's claim bites and no uniform-in-m
    Chebyshev bound exists.  `beta` is ignored: the design owns it.
    """
    return DESIGN.draw(n, tail_index, rng)


def weights(X, theta):
    p = 2.0 - 2.0 * theta
    if p == 0:
        return np.ones(len(X))
    nrm = np.linalg.norm(X, axis=1)
    w = np.zeros(len(X))
    nz = nrm > 0
    w[nz] = nrm[nz] ** p
    return w


# --------------------------------------------------------------------------
# the split rule
# --------------------------------------------------------------------------
def tr_v_split(X, Y, theta, rng, rho=0.5, m0=3):
    """Pilot on D1, trace on D2.  Returns (tr Vhat, diagnostics)."""
    n = len(Y)
    idx = rng.permutation(n)
    n1 = min(max(int(round(rho * n)), DIM + 1), n - DIM - 1)
    d1, d2 = idx[:n1], idx[n1:]
    A1, r1 = operator(X[d1], Y[d1], theta)
    beta0 = cg(A1, r1, m0)
    e2 = Y[d2] - X[d2] @ beta0
    w2 = weights(X[d2], theta)
    return float(np.mean(e2 ** 2 * w2)), {"n1": int(n1), "n2": int(len(d2)),
                                          "beta0": beta0}


# BCT's own practical values (arXiv:2402.11134, Supplement S.5.1): tau = 1.01 because
# their theory needs tau > 1, delta = 0.1 for a 90% confidence level.  The threshold
# is tau * sqrt(2 tr V / (delta n)) -- BCT's Assumption 4 with sigma^2 E||X||^2
# replaced by the quantity it bounds.  The 2/delta was DROPPED in the first version of
# this rule (audit 2026-09-05, MATH-AUDIT-2026-09-05.md, finding 1): without it the
# transport of BCT's Lemma S.6 needs tau > c sqrt(2/delta), which grows like sqrt(n)
# at delta = 1/n, and tau = 1 puts the threshold at the RMS of the noise itself.
TAU_BCT, DELTA_BCT = 1.01, 0.1


def multiplier(tau, delta):
    """The factor in front of sqrt(tr V / n).  delta=None reproduces the legacy rule."""
    return tau if delta is None else tau * math.sqrt(2.0 / delta)


def stop_split(X, Y, theta, rng, tau=1.0, m_max=25, rho=0.5, m0=3, delta=None):
    """CG on the full sample, threshold fixed in advance by the split estimate.

    threshold = tau * sqrt(2 tr Vhat_split / (delta n));  delta=None means the legacy
    tau * sqrt(tr Vhat_split / n), kept so the older parts of this file reproduce.
    """
    n = len(Y)
    trv, info = tr_v_split(X, Y, theta, rng, rho=rho, m0=m0)
    thr = multiplier(tau, delta) * math.sqrt(max(trv, 0.0) / n)
    A, r = operator(X, Y, theta)
    a = np.zeros_like(r)
    res = r.copy()
    d = res.copy()
    for m in range(1, m_max + 1):
        Ad = A @ d
        den = d @ Ad
        if den <= 0 or not np.isfinite(den):
            return a, m - 1, trv
        step = (res @ res) / den
        a = a + step * d
        new = res - step * Ad
        if not np.isfinite(new).all():
            return a, m - 1, trv
        rr = res @ res
        if rr <= 0:
            return a, m, trv
        res, d = new, new + (new @ new) / rr * d
        if np.linalg.norm(res) <= thr:
            return a, m, trv
    return a, m_max, trv


def stop_oracle(X, Y, theta, trv_true, tau=1.0, m_max=25):
    """Same, with the population tr V_theta -- the unattainable benchmark."""
    n = len(Y)
    thr = tau * math.sqrt(trv_true / n)
    A, r = operator(X, Y, theta)
    a = np.zeros_like(r)
    res = r.copy()
    d = res.copy()
    for m in range(1, m_max + 1):
        Ad = A @ d
        den = d @ Ad
        if den <= 0 or not np.isfinite(den):
            return a, m - 1
        step = (res @ res) / den
        a = a + step * d
        new = res - step * Ad
        if not np.isfinite(new).all():
            return a, m - 1
        rr = res @ res
        if rr <= 0:
            return a, m
        res, d = new, new + (new @ new) / rr * d
        if np.linalg.norm(res) <= thr:
            return a, m
    return a, m_max


def trv_population(rng, theta, tail_index, n=400000):
    """tr V_theta = E[eps^2 ||X||^{2-2theta}] = sigma^2 E||X||^{2-2theta}.

    Estimated on a large independent draw.  At theta = 0 and index below 2 this is
    infinite and the sample value is meaningless; the designs used here keep it
    finite, which is the scope condition (M2) itself.
    """
    x, _ = DESIGN.draw(n, tail_index, rng)
    return float(DESIGN.sigma ** 2 * np.mean(weights(x, theta)))


# --------------------------------------------------------------------------
# A -- the defect is real
# --------------------------------------------------------------------------
def part_a(rng, theta, tail_index, n, reps) -> dict:
    """tr Vhat(m) along the CG path: does the threshold chase the iterate?"""
    m_grid = [1, 2, 3, 5, 8, 12]
    paths = []
    for _ in range(reps):
        X, Y = draw(rng, n, tail_index)
        A, r = operator(X, Y, theta)
        w = weights(X, theta)
        row = []
        for m in m_grid:
            b = cg(A, r, m)
            e = Y - X @ b
            row.append(float(np.mean(e ** 2 * w)))
        paths.append(row)
    P = np.array(paths)
    med = P.mean(axis=0)
    se = P.std(axis=0, ddof=1) / math.sqrt(reps)
    fall = float(med[0] / med[-1])
    # is the fall bigger than the sampling error on the same quantity?
    resolved = bool(abs(med[0] - med[-1]) > 4.0 * math.hypot(se[0], se[-1]))
    return {"m_grid": m_grid, "mean_trVhat": med.tolist(), "se": se.tolist(),
            "fall_m1_to_m12": fall, "monotone_decreasing":
                bool(all(med[i + 1] <= med[i] for i in range(len(med) - 1))),
            "fall_resolved_against_noise": resolved,
            "pass": bool(fall > 1.0 and resolved)}


# --------------------------------------------------------------------------
# B -- consistency of the split estimate, term by term
# --------------------------------------------------------------------------
def part_b(rng, theta, tail_index, ns, reps, rho, m0) -> dict:
    trv_true = trv_population(rng, theta, tail_index)
    beta = make_beta()
    rows = {}
    for n in ns:
        rel, bias_t, cross_t, var_t = [], [], [], []
        for _ in range(reps):
            X, Y = draw(rng, n, tail_index, beta=beta)
            trv, info = tr_v_split(X, Y, theta, rng, rho=rho, m0=m0)
            rel.append(trv / trv_true - 1.0)
            # decompose on a fresh independent half, using the same pilot
            d = beta - info["beta0"]
            Xf, Yf = draw(rng, 20000, tail_index, beta=beta)
            wf = weights(Xf, theta)
            ef = Yf - Xf @ beta                      # the true errors
            proj = Xf @ d
            var_t.append(float(np.mean(ef ** 2 * wf)))
            cross_t.append(float(np.mean(2 * ef * proj * wf)))
            bias_t.append(float(np.mean(proj ** 2 * wf)))
        rows[str(n)] = {
            "median_rel_error": float(np.median(rel)),
            "iqr_rel_error": float(np.subtract(*np.percentile(rel, [75, 25]))),
            "median_eps2w_term": float(np.median(var_t)),
            "median_cross_term": float(np.median(cross_t)),
            "median_bias_term": float(np.median(bias_t)),
        }
    keys = list(rows)
    shrinks = abs(rows[keys[-1]]["median_rel_error"]) <= abs(rows[keys[0]]["median_rel_error"]) + 0.05
    tight = abs(rows[keys[-1]]["median_rel_error"]) < 0.25
    bias_small = abs(rows[keys[-1]]["median_bias_term"]) < 0.5 * trv_true
    return {"trV_population": trv_true, "rows": rows,
            "error_does_not_grow": bool(shrinks), "within_25pc_at_largest_n": bool(tight),
            "bias_term_small": bool(bias_small),
            "pass": bool(shrinks and tight and bias_small)}


# --------------------------------------------------------------------------
# C -- the price
# --------------------------------------------------------------------------
def part_c(rng, theta, cells, reps, rho, m0) -> dict:
    """The price of the split, and -- first -- whether the threshold binds at all.

    A comparison of stopping rules is empty if every rule stops at the same step in
    every replication.  A first version of this part reported a cost of exactly
    1.000x everywhere, which is not a measurement: at n = 1000 and tau = 1 all three
    rules returned mhat = 2 in every replication.  The disagreement rate therefore
    comes first, and the error ratio is read only where it is non-zero.

    tau is swept because it is what makes the threshold bind: at tau = 1 the rule
    stops after two or three steps and the threshold is nowhere near critical.

    The index is swept across the (M1) boundary.  At theta = 1, (M1) is
    E||X||^2 < inf, i.e. tail index > 2.  The bias term of the split estimate is
    controlled by (M1) and by nothing else, so the prediction made before running
    is that the two rules agree where (M1) holds and part company where it fails.
    """
    out = {}
    for (ti, n, tau) in cells:
        trv_true = trv_population(rng, theta, ti)
        err = {"self": [], "split": [], "oracle": []}
        steps = {"self": [], "split": [], "oracle": []}
        for _ in range(reps):
            X, Y = draw(rng, n, ti)
            b1, m1 = discrepancy_stop(X, Y, theta, tau=tau)
            b2, m2, _ = stop_split(X, Y, theta, rng, tau=tau, rho=rho, m0=m0)
            b3, m3 = stop_oracle(X, Y, theta, trv_true, tau=tau)
            for k, (b, m) in (("self", (b1, m1)), ("split", (b2, m2)),
                              ("oracle", (b3, m3))):
                err[k].append(float(np.linalg.norm(b - DESIGN.beta)
                                    / np.linalg.norm(DESIGN.beta)))
                steps[k].append(m)
        s_self, s_split = np.array(steps["self"]), np.array(steps["split"])
        row = {k: {"median_rel_error": float(np.median(v)),
                   "median_steps": float(np.median(steps[k])),
                   "steps_min": int(min(steps[k])), "steps_max": int(max(steps[k]))}
               for k, v in err.items()}
        row["M1_holds"] = bool(ti > 2.0)          # theta = 1: (M1) is E||X||^2 < inf
        row["disagree_rate_split_vs_self"] = float((s_self != s_split).mean())
        row["split_over_self"] = (row["split"]["median_rel_error"]
                                  / row["self"]["median_rel_error"])
        row["split_over_oracle"] = (row["split"]["median_rel_error"]
                                    / row["oracle"]["median_rel_error"])
        out[f"index={ti}, n={n}, tau={tau:g}"] = row

    inside = [v for v in out.values() if v["M1_holds"]]
    outside = [v for v in out.values() if not v["M1_holds"]]
    dis_in = max(v["disagree_rate_split_vs_self"] for v in inside) if inside else 0.0
    dis_out = max(v["disagree_rate_split_vs_self"] for v in outside) if outside else 0.0
    costs = [v["split_over_self"] for v in inside] or [1.0]
    return {"rows": out,
            "max_disagreement_where_M1_holds": float(dis_in),
            "max_disagreement_where_M1_fails": float(dis_out),
            "agreement_tracks_M1": bool(dis_out > dis_in),
            "max_cost_where_M1_holds": float(max(costs)),
            "median_cost_where_M1_holds": float(np.median(costs)),
            "pass": bool(max(costs) < 1.25 and dis_out > dis_in)}


# --------------------------------------------------------------------------
# D -- does the 2026-08-21 rescaling gain survive the split?
# --------------------------------------------------------------------------
def part_d(rng, indices, ns, reps, rho, m0) -> dict:
    """The OPERATOR-scale rule (superseded) against the split rule, over an n grid.

    THE CLAIM BEING RE-EXAMINED.  On 2026-08-21 the rescale from the operator scale
    to the noise scale was recorded as worth 1.78x/1.70x/1.64x at theta = 2.  Those
    ratios were read off a1_5_consistency.json against its OLDRULE twin -- at the
    LARGEST n only, one replication set per cell, and against the SELF-REFERENTIAL
    rule whose proof does not survive.  Reading the same two artifacts across the
    whole n grid gives, at theta = 2 and index 1.5, the sequence 1.64, 0.98, 1.35,
    1.81: it is not monotone and it crosses one.  So the honest question is not
    "does the gain survive the split" but "was there a stable gain at all".

    This part therefore reports the ratio at EVERY n, and the spread across n.  Read
    with part F, which recomputes the published table at its own n = 1000 and finds
    the theta = 2 column intact and its stated mechanism confirmed, the conclusion is
    narrower than "the gain is an artefact": the theta = 2 gain is real and has a
    named cause, but the RATIO ITSELF is not stable in n at any theta, so no single
    cell of it should be quoted as a headline.  CG steps are integers, so the error
    jumps as mhat moves, and a ratio of two medians inherits those jumps.
    """
    out = {}
    for theta in (1.0, 2.0):
        for ti in indices:
            row = {}
            for n in ns:
                e_old, e_new = [], []
                for _ in range(reps):
                    X, Y = draw(rng, n, ti)
                    bo, _ = discrepancy_stop_operator_scale(
                        X, Y, theta, kappa=multiplier(TAU_BCT, DELTA_BCT))
                    bn, _, _ = stop_split(X, Y, theta, rng, rho=rho, m0=m0,
                                          tau=TAU_BCT, delta=DELTA_BCT)
                    e_old.append(float(np.linalg.norm(bo - DESIGN.beta)
                                       / np.linalg.norm(DESIGN.beta)))
                    e_new.append(float(np.linalg.norm(bn - DESIGN.beta)
                                       / np.linalg.norm(DESIGN.beta)))
                mo, mn = float(np.median(e_old)), float(np.median(e_new))
                row[str(n)] = {"operator_scale": mo, "split": mn, "ratio": mo / mn}
            ratios = [row[str(n)]["ratio"] for n in ns]
            row["ratio_min"] = float(min(ratios))
            row["ratio_max"] = float(max(ratios))
            row["crosses_one"] = bool(min(ratios) < 1.0 < max(ratios))
            out[f"theta={theta:g}, index={ti}"] = row

    any_cross = any(v["crosses_one"] for v in out.values())
    worst = min(v["ratio_min"] for v in out.values())
    # The prediction, made before running: the ratio is NOT stable in n, so the
    # 1.78x is an artefact of reading one cell.  "pass" means the split rule is
    # never materially WORSE than the operator-scale rule, not that it wins.
    return {"rows": out, "any_cell_crosses_one": bool(any_cross),
            "worst_ratio_over_grid": float(worst),
            "gain_is_stable_in_n": bool(not any_cross),
            "split_never_materially_worse": bool(worst > 0.85),
            "pass": bool(worst > 0.85)}


def part_f(rng, reps, rho, m0) -> dict:
    """Recompute Table 7.3 (tab:stop) under the split rule, and test its explanation.

    The published table is the ratio of median slope errors, operator-scale
    threshold against the noise-scale threshold, at n = 1000, over
    index in {1.5, 2.5, 4.0} and theta in {0, 0.5, 1, 2}.  Its caption explains the
    theta = 2 column -- 1.78x, 1.70x, 1.64x -- by saying the operator-scale rule
    "halts the spatial sign after a single step".  That is a mechanism, and a
    mechanism is checkable: at theta = 2 the operator scale is
    tr A_2 = E||X||^0 = P(X != 0) ~ 1, so the threshold is ~ sqrt(1/n) and stops
    immediately, whereas tr V_2 = E[eps^2 ||X||^{-2}] is small.  So the median
    number of steps of the operator-scale rule at theta = 2 must be 1.

    Reported: the recomputed ratios under the split rule, and the median step counts
    that the caption's explanation predicts.

    2026-09-05, after the delta repair.  Both rules now run at BCT's multiplier
    tau sqrt(2/delta) = 4.52 rather than at 1.  At that threshold both rules stop
    after one step at theta = 2, so the theta = 2 column is 1.00x by construction
    and the mechanism above, while still true of the operator-scale rule, no longer
    separates the rules there.  The columns that now separate them are theta in
    {1/2, 1} at tail index 1.5, where tr A_theta is finite and the noise scale is
    the right one; the checks below were rewritten to say that, and the old
    criterion is kept in the artifact so the change is visible.
    """
    ratios, steps = {}, {}
    for ti in (1.5, 2.5, 4.0):
        for th in (0.0, 0.5, 1.0, 2.0):
            e_op, e_sp, m_op, m_sp = [], [], [], []
            for _ in range(reps):
                X, Y = draw(rng, 1000, ti)
                # both rules at the SAME multiplier tau sqrt(2/delta), BCT's values
                bo, mo = discrepancy_stop_operator_scale(
                    X, Y, th, kappa=multiplier(TAU_BCT, DELTA_BCT))
                bs, ms, _ = stop_split(X, Y, th, rng, rho=rho, m0=m0,
                                       tau=TAU_BCT, delta=DELTA_BCT)
                e_op.append(float(np.linalg.norm(bo - DESIGN.beta) / np.linalg.norm(DESIGN.beta)))
                e_sp.append(float(np.linalg.norm(bs - DESIGN.beta) / np.linalg.norm(DESIGN.beta)))
                m_op.append(mo); m_sp.append(ms)
            ratios[f"{ti}|{th}"] = float(np.median(e_op) / np.median(e_sp))
            steps[f"{ti}|{th}"] = {"operator": float(np.median(m_op)),
                                   "split": float(np.median(m_sp))}
    theta2 = {k: v for k, v in ratios.items() if k.endswith("|2.0")}
    op_steps_theta2 = [steps[k]["operator"] for k in theta2]
    mechanism = all(s <= 1.0 for s in op_steps_theta2)
    others = {k: v for k, v in ratios.items() if not k.endswith("|2.0")
              and not (k.startswith("1.5|") and k.endswith("|0.0"))}
    agree = all(0.80 < v < 1.45 for v in others.values())
    # At the legacy multiplier 1 the theta=2 column read 1.59-1.84x, because the
    # operator-scale rule stopped at m=1 there while the noise-scale rule went to
    # m=2.  At BCT's multiplier BOTH rules stop at m=1 at theta=2, so that column
    # is 1.00x by construction and the old criterion (> 1.15) cannot hold.  What
    # survives, and is now the check: the split rule is never materially worse than
    # the operator scale (min ratio > 0.9), and where the two scales differ most --
    # tail index 1.5, where tr A_theta is finite and tr V_theta is not at small
    # theta -- the noise scale wins.
    split_steps_theta2 = [steps[k]["split"] for k in theta2]
    heavy = {k: v for k, v in ratios.items()
             if k.startswith("1.5|") and (k.endswith("|0.5") or k.endswith("|1.0"))}
    return {"ratios": ratios, "median_steps": steps,
            "theta2_ratios": theta2,
            "operator_median_steps_at_theta2": op_steps_theta2,
            "split_median_steps_at_theta2": split_steps_theta2,
            "caption_mechanism_holds": bool(mechanism),
            "theta2_column_still_favours_noise_scale":
                bool(min(theta2.values()) > 1.15),
            "both_rules_one_step_at_theta2":
                bool(mechanism and all(s <= 1.0 for s in split_steps_theta2)),
            "never_materially_worse": bool(min(ratios.values()) > 0.9),
            "noise_scale_wins_at_index_1p5_mid_theta": bool(min(heavy.values()) > 1.15),
            "other_cells_agree": bool(agree),
            "pass": bool(mechanism and min(ratios.values()) > 0.9
                         and min(heavy.values()) > 1.15)}


# --------------------------------------------------------------------------
# E -- sensitivity to rho and m0
# --------------------------------------------------------------------------
def part_e(rng, theta, tail_index, n, reps) -> dict:
    beta = make_beta()
    out = {}
    for rho in (0.2, 0.35, 0.5):
        for m0 in (2, 3, 5):
            errs = []
            for _ in range(reps):
                X, Y = draw(rng, n, tail_index, beta=beta)
                b, _, _ = stop_split(X, Y, theta, rng, rho=rho, m0=m0,
                                     tau=TAU_BCT, delta=DELTA_BCT)
                errs.append(float(np.linalg.norm(b - beta) / np.linalg.norm(beta)))
            out[f"rho={rho:g}, m0={m0}"] = float(np.median(errs))
    vals = list(out.values())
    spread = max(vals) / min(vals)
    best = min(out, key=out.get)
    return {"median_rel_error": out, "spread": float(spread), "best": best,
            "pass": bool(spread < 1.35)}


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    if args.quick:
        n, reps = 600, 60
        ns_b, reps_b = (500, 2000), 30
        cells_c, reps_c = ((2.5, 2000, 1.0), (2.5, 2000, 0.3), (1.5, 2000, 0.3)), 60
        idx_d, ns_d, reps_d = (1.5,), (500, 2000), 60
        reps_e, reps_f = 40, 60
    else:
        n, reps = 1000, 300
        ns_b, reps_b = (500, 2000, 8000), 120
        cells_c, reps_c = ((2.5, 2000, 1.0), (2.5, 2000, 0.3), (2.5, 8000, 0.3),
                           (2.5, 2000, 0.15), (3.5, 2000, 0.3),
                           (1.5, 2000, 0.3), (1.5, 2000, 0.15)), 300
        idx_d, ns_d, reps_d = (1.5, 3.0), (500, 2000, 8000), 200
        reps_e, reps_f = 200, 400

    rho, m0 = 0.35, 3
    res = {"seed": SEED, "quick": args.quick, "rho": rho, "m0": m0, "dim": DIM,
           "theta_main": 1.0}
    # The run's own settings, recorded in the artifact rather than only in this file.
    # A CoE audit could not resolve "over 300 replications" or "at tail index 2.5"
    # against results/split_stopping_rule.json, because the numbers the manuscript
    # prints as design settings existed only as literals here.  A setting that is not
    # in the artifact is a setting the audit has to take on trust.
    res["config"] = {
        # parts C uses tau as the WHOLE multiplier of sqrt(tr V/n) (legacy form,
        # swept downward so the threshold binds); parts E and F use BCT's
        # tau sqrt(2/delta) at their practical values.
        "tau_bct": TAU_BCT, "delta_bct": DELTA_BCT,
        "multiplier_bct": multiplier(TAU_BCT, DELTA_BCT),
        "part_a": {"theta": 1.0, "tail_index": 2.5, "n": n, "reps": reps},
        "part_b": {"theta": 1.0, "tail_index": 2.5, "ns": list(ns_b), "reps": reps_b},
        "part_c": {"theta": 1.0, "reps": reps_c,
                   "cells": [{"tail_index": c[0], "n": c[1], "tau": c[2]} for c in cells_c]},
        "part_d": {"indices": list(idx_d), "ns": list(ns_d), "reps": reps_d},
        "part_e": {"theta": 1.0, "tail_index": 2.5, "n": n, "reps": reps_e},
        "part_f": {"n": 1000, "reps": reps_f, "indices": [1.5, 2.5, 4.0],
                   "thetas": [0.0, 0.5, 1.0, 2.0]},
    }
    res["A_defect"] = part_a(rng, 1.0, 2.5, n, reps)
    res["B_consistency"] = part_b(rng, 1.0, 2.5, ns_b, reps_b, rho, m0)
    res["C_price"] = part_c(rng, 1.0, cells_c, reps_c, rho, m0)
    res["D_rescale_gain"] = part_d(rng, idx_d, ns_d, reps_d, rho, m0)
    res["E_sensitivity"] = part_e(rng, 1.0, 2.5, n, reps_e)
    res["F_table73"] = part_f(rng, reps_f, rho, m0)
    res["all_pass"] = all(res[k]["pass"] for k in res if isinstance(res[k], dict)
                          and "pass" in res[k])

    L = []
    P = L.append
    P("=" * 78)
    P("Repair (b): a stopping threshold that does not depend on the stopping path")
    P(f"seed {SEED}   theta=1   rho={rho}   m0={m0}" + ("   [--quick]" if args.quick else ""))
    P("=" * 78)

    a = res["A_defect"]
    P("")
    P("A. the defect: tr Vhat(m) along the CG path (theta=1, tail index 2.5)")
    P(f"      {'m':>6}" + "".join(f"{m:>10}" for m in a["m_grid"]))
    P(f"      {'mean':>6}" + "".join(f"{v:>10.4f}" for v in a["mean_trVhat"]))
    P(f"      {'s.e.':>6}" + "".join(f"{v:>10.4f}" for v in a["se"]))
    P(f"   falls by {a['fall_m1_to_m12']:.3f}x from m=1 to m=12; monotone: "
      f"{'yes' if a['monotone_decreasing'] else 'no'}; resolved against sampling error: "
      f"{'yes' if a['fall_resolved_against_noise'] else 'NO'}")
    P("   The threshold therefore falls as the iterate fits, which is the")
    P(f"   dependence the proof cannot absorb.   [{'ok' if a['pass'] else 'FAIL'}]")

    b = res["B_consistency"]
    P("")
    P(f"B. consistency of the split estimate.  tr V_theta = {b['trV_population']:.4f}")
    P("   (theta=1, tail index 2.5: E||X||^2 finite, E||X||^4 INFINITE)")
    P(f"      {'n':>8}  {'rel. error':>11}  {'IQR':>8}  {'eps^2 w':>9}  {'cross':>9}  {'bias':>9}")
    for k, v in b["rows"].items():
        P(f"      {int(k):>8}  {v['median_rel_error']:>+11.4f}  {v['iqr_rel_error']:>8.4f}"
          f"  {v['median_eps2w_term']:>9.4f}  {v['median_cross_term']:>+9.4f}"
          f"  {v['median_bias_term']:>9.4f}")
    P("   The three columns are the terms of (eps + <d,X>)^2 w: the first has mean")
    P("   tr V_theta and needs only (M2); the third is bounded by ||d||^2 E||X||^{4-2t},")
    P("   which is (M1); the cross term by 2||d|| sqrt((M1)(M2)).  No fourth moment.")
    P(f"   -> [{'ok' if b['pass'] else 'FAIL'}]")

    c = res["C_price"]
    P("")
    P("C. does the threshold bind, and what does the split cost?  (theta=1)")
    P("   A rule comparison is empty where every rule stops at the same step, so the")
    P("   disagreement rate comes first.  tau is swept because it is what makes the")
    P("   threshold bind; the index is swept across the (M1) boundary, which at")
    P("   theta=1 is E||X||^2 < inf, i.e. tail index > 2.")
    P(f"      {'design':<26}{'(M1)':>6}{'disagree':>10}{'self-ref':>10}{'split':>10}"
      f"{'oracle':>9}{'split/self':>12}{'steps':>10}")
    for k, v in c["rows"].items():
        P(f"      {k:<26}{'yes' if v['M1_holds'] else 'NO':>6}"
          f"{v['disagree_rate_split_vs_self']:>9.1%}"
          f"{v['self']['median_rel_error']:>10.4f}{v['split']['median_rel_error']:>10.4f}"
          f"{v['oracle']['median_rel_error']:>9.4f}{v['split_over_self']:>12.3f}"
          f"{v['split']['steps_min']:>7}-{v['split']['steps_max']:<2}")
    P(f"   worst disagreement where (M1) holds: {c['max_disagreement_where_M1_holds']:.1%};"
      f" where it fails: {c['max_disagreement_where_M1_fails']:.1%}")
    P("   The split estimate's bias term is controlled by (M1) and by nothing else,")
    P("   so this is the predicted pattern, not a coincidence.  Where (M1) holds the")
    P(f"   split costs at most {c['max_cost_where_M1_holds']:.3f}x the self-referential")
    P(f"   rule (median {c['median_cost_where_M1_holds']:.3f}x).   [{'ok' if c['pass'] else 'FAIL'}]")

    d = res["D_rescale_gain"]
    P("")
    P("D. is the rescaling ratio stable in n?  operator-scale vs split")
    P("   The 1.78x on record is one cell of a comparison at a single n.  Read across")
    P("   the n grid, the stored artifacts already give 1.64, 0.98, 1.35, 1.81 in that")
    P("   row.  Same comparison against the split rule, at every n (see also part F,")
    P("   which recomputes the published table at its own n=1000):")
    ns_row = [k for k in next(iter(d["rows"].values())) if k.isdigit()]
    P(f"      {'design':<22}" + "".join(f"{'n=' + k:>12}" for k in ns_row)
      + f"{'min':>8}{'max':>8}")
    for k, v in d["rows"].items():
        P(f"      {k:<22}" + "".join(f"{v[n]['ratio']:>11.2f}x" for n in ns_row)
          + f"{v['ratio_min']:>8.2f}{v['ratio_max']:>8.2f}"
          + ("   crosses 1" if v["crosses_one"] else ""))
    P(f"   stable in n: {'yes' if d['gain_is_stable_in_n'] else 'NO -- the ratio crosses one'}."
      " No single cell of this ratio should be quoted as a headline number.")
    P(f"   worst ratio over the whole grid {d['worst_ratio_over_grid']:.2f}: the split rule")
    P(f"   is never materially worse.   [{'ok' if d['pass'] else 'FAIL'}]")

    e = res["E_sensitivity"]
    P("")
    P("E. sensitivity to the split fraction and the pilot depth")
    for k, v in e["median_rel_error"].items():
        P(f"      {k:<18} {v:>8.4f}")
    P(f"   spread {e['spread']:.3f}x across the grid; best {e['best']}."
      f"   [{'ok' if e['pass'] else 'FAIL'}]")

    f = res["F_table73"]
    P("")
    P("F. Table 7.3 recomputed under the split rule (n=1000), and its caption tested")
    P("   The published table explains its theta=2 column by saying the operator-scale")
    P("   rule halts the spatial sign after a single step.  That is checkable.")
    P(f"      {'index':>7}" + "".join(f"{'theta=' + t_:>12}" for t_ in ("0", "0.5", "1", "2")))
    for ti in (1.5, 2.5, 4.0):
        P(f"      {ti:>7}" + "".join(f"{f['ratios'][f'{ti}|{th}']:>11.2f}x"
                                     for th in (0.0, 0.5, 1.0, 2.0)))
    P("   median steps of the operator-scale rule at theta=2: "
      + ", ".join(f"{s:g}" for s in f["operator_median_steps_at_theta2"])
      + f"  -> caption mechanism {'CONFIRMED' if f['caption_mechanism_holds'] else 'REFUTED'}")
    P("   median steps of the split rule at theta=2: "
      + ", ".join(f"{s:g}" for s in f["split_median_steps_at_theta2"])
      + f"  -> both rules one step at theta=2: "
        f"{'yes' if f['both_rules_one_step_at_theta2'] else 'no'}")
    P(f"   split never materially worse (min ratio > 0.9): "
      f"{'yes' if f['never_materially_worse'] else 'NO'};"
      f" noise scale wins at index 1.5, theta in {{1/2, 1}}: "
      f"{'yes' if f['noise_scale_wins_at_index_1p5_mid_theta'] else 'NO'}")
    P(f"   -> [{'ok' if f['pass'] else 'FAIL'}]")

    P("")
    P("=" * 78)
    P("ALL PASS" if res["all_pass"] else "FAILURE -- a prediction did not hold")
    P("=" * 78)

    text = "\n".join(L)
    print(text)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "split_stopping_rule.out").write_text(text + "\n", encoding="utf-8")
    (RESULTS / "split_stopping_rule.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
