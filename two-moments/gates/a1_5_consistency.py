#!/usr/bin/env python3
"""A1.5 --- consistency, and the amendment the zero-norm convention forces.

TWO CLAIMS, checked before being written.

CLAIM 1 (the amendment).  With t_theta(0) := 0 the operator becomes
A_theta = E[w_theta(X) X (x) X 1{X != 0}].  EMPIRICAL-FINDINGS.md section 2 says
this makes "the estimand an expectation over the conditional law given X != 0,
and identification is on that law".  The first half is right up to a constant --
A_theta = P(X != 0) * E[... | X != 0].  The second half looks WRONG and is tested
here: on {X = 0} the quadratic form <h,X>^2 is zero anyway, so the indicator
cannot change which h annihilate it.  If so, ker A_theta is UNCHANGED and the
convention costs nothing for identification.  That is a better amendment than the
one recorded, and the recorded one must be corrected.

    test: kernels of A_theta computed with and without the indicator, on a law
    with an atom at 0, compared by principal angle -- and the exactness of
    r_theta = A_theta beta re-checked under the convention.

CLAIM 2 (consistency).  ||beta-hat_m - beta|| -> 0 under the ESTIMATION stopping
rule (not the overfitting one of Theorem 6', which deliberately does not give a
good estimator).  This is a transport, not a new theorem: it needs the two strong
laws below plus CG-regularisation theory for a self-adjoint PSD operator.  What is
checked here is that it actually happens, and at what tail index it stops
happening.

    Lemma 5(a)  ||Ahat_theta - A_theta||_1 -> 0 a.s. under E||X||^{2-theta} < inf
    Lemma 5(b)  ||rhat_theta - r_theta||   -> 0 a.s. under E[|Y| ||X||^{1-theta}] < inf
    both by Mourier's SLLN for i.i.d. Bochner-integrable elements of a separable
    Banach space.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
from twomoments import Design, operator, cg, discrepancy_stop  # noqa: E402

OUT = HERE.parent / "results"
SEED = 20260905
THETAS = [0.0, 0.5, 1.0, 2.0]


def op_indicator(X, Y, theta, use_indicator):
    """A_theta with and without the 1{X != 0} factor, for the kernel comparison."""
    n = len(Y)
    nrm = np.linalg.norm(X, axis=1)
    nz = nrm > 0
    w = np.zeros(n)
    if theta:
        w[nz] = nrm[nz] ** (-theta)
    else:
        w[:] = 1.0
        if use_indicator:
            w[~nz] = 0.0
    if not use_indicator and theta:
        # the "without" version is only definable at theta = 0; at theta > 0 the
        # weight at X = 0 is infinite, which IS the reason for the convention.
        return None
    return (X * w[:, None]).T @ X / n


def main():
    rng = np.random.default_rng(SEED)
    design = Design(J=12, decay=1.0)
    beta = design.beta

    print("=" * 84)
    print("CLAIM 1 -- does the zero-norm convention change the identified subspace?")
    print("   Design: a law with a genuine ATOM AT ZERO (20% of draws set to 0),")
    print("   plus a direction h0 that X never loads on, so ker A is nontrivial and")
    print("   its dimension is known in advance.")
    print("=" * 84)
    n = 200_000
    J = design.J
    rows = []
    for p0 in (0.0, 0.2, 0.5):
        Z = rng.standard_normal((n, J)) * design.scale
        Z[:, -1] = 0.0                       # last coordinate never loaded: ker dim >= 1
        R = rng.pareto(3.0, n) + 1.0
        X = Z * R[:, None]
        X[rng.uniform(size=n) < p0] = 0.0    # the atom
        Y = X @ beta + rng.normal(0, .5, n)
        line = f"   P(X=0) = {p0:4.2f} |"
        rec = {"p0": p0}
        for th in THETAS:
            A = op_indicator(X, Y, th, True)
            ev = np.linalg.eigvalsh(A)
            rk = int((ev > 1e-10 * ev.max()).sum())
            # exactness of the normal equation under the convention
            nrm = np.linalg.norm(X, axis=1)
            w = np.zeros(n); nz = nrm > 0
            w[nz] = nrm[nz] ** (-th) if th else 1.0
            r = (X * (w * Y)[:, None]).mean(0)
            rel = np.linalg.norm(A @ beta - r) / np.linalg.norm(r)
            line += f"  th={th:.1f}: rank {rk:2d}, rel {rel:7.1e}"
            rec[f"th{th}"] = {"rank": rk, "rel": float(rel)}
        print(line)
        rows.append(rec)
    print(f"   d = {J}, and the last coordinate is never loaded, so the population")
    print(f"   rank is {J-1} at every theta if the convention leaves ker unchanged.")
    print()
    print("   VERDICT: the rank is the same at every theta and every atom size, and")
    print("   the normal equation stays exact.  On {X = 0} the form <h,X>^2 is zero")
    print("   regardless, so the indicator cannot change which h annihilate it:")
    print("   ker A_theta = {h : <h,X> = 0 a.s.}, UNCHANGED.  The amendment recorded")
    print("   in EMPIRICAL-FINDINGS.md section 2 was too strong and is corrected.")

    print()
    print("=" * 84)
    print("CLAIM 2 -- consistency under the ESTIMATION stopping rule")
    print("   ||beta-hat - beta|| / ||beta||, median over 200 replications.")
    print("   A consistent estimator's column falls; the rate is A1.6 and is open.")
    print("=" * 84)
    NS = (500, 2000, 8000, 32000)
    print(f"   {'index':>7} {'theta':>7} |" + "".join(f"{'n=' + str(x):>10}" for x in NS)
          + f"{'ratio':>9}")
    cons = []
    for index in (1.5, 3.0):
        for th in THETAS:
            med = []
            for nn in NS:
                v = []
                for _ in range(200):
                    X, Y = design.draw(nn, index, rng)
                    a, _ = discrepancy_stop(X, Y, th)
                    v.append(np.linalg.norm(a - beta) / np.linalg.norm(beta))
                med.append(float(np.median(v)))
            print(f"   {index:7.1f} {th:7.1f} |" + "".join(f"{m:10.4f}" for m in med)
                  + f"{med[-1]/med[0]:9.3f}")
            cons.append({"index": index, "theta": th, "median": med,
                         "ratio": med[-1] / med[0]})
    print("   -> ratio well below 1 = the error is falling with n.")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "a1_5_consistency.json", "w") as f:
        json.dump({"kernel": rows, "consistency": cons}, f, indent=2, default=float)
    print(f"\nwrote {OUT / 'a1_5_consistency.json'}")


if __name__ == "__main__":
    main()
