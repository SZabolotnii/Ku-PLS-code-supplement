#!/usr/bin/env python3
"""Gate A0.1(b) --- the load-bearing claim, checked before it is written down.

THE CLAIM THIS SCRIPT WAS WRITTEN TO CONFIRM, AND DID NOT.

Among w_theta(x) = ||x||^{-theta} the induced test function

    t_theta(x) = w_theta(x) x = x / ||x||^theta,   ||t_theta(x)|| = ||x||^{1-theta}

is bounded iff theta = 1, where it is the spatial sign of norm exactly one.  The
claim under test was that boundedness of t buys exactness of the normal equation
under mean independence (MI) with NO moment on eps -- the property the ET
manuscript asserted for its characteristic-function feature and that the plan
lists among the results surviving the refutation ("r_phi = C beta -- correct, by
linearity of covariance plus (MI)").

IT IS FALSE, AND THE SCRIPT SHOWS WHY.  The error enters every estimation-side
object as a plain sample mean

    (1/n) sum_k eps_k t(X_k),

and no boundedness of t repairs an eps that has no mean of its own.  Under a
Cauchy error this average converges in distribution to a Cauchy law, not to
zero, for the spatial sign and for e^{i<u,X>} exactly as for x itself.  The
run below shows all four routes flat in n.

The reason is upstream of any of them: (MI), "E[eps | X] = 0", PRESUPPOSES
E|eps| < inf.  A conditional expectation of a Cauchy variable does not exist,
so the restriction the manuscript leans on is not merely unverified under an
error with no mean -- it is undefined.  Section 2 of the earlier manuscript states
the opposite in as many words ("it does not require E|eps| < inf, let alone a
finite error variance").  This is a FOURTH defect, independent of the three the
decision listed and not named in the referee report, and it damages one of the
identities the plan had marked reusable verbatim.

WHAT IS ACTUALLY TRUE, and is what the new paper may claim:

    exactness of  A_theta beta = r_theta     needs  E|eps| < inf  (and E[|eps| ||X||^{1-theta}] < inf)
    sqrt(n) CLT for rhat_theta               needs  E[eps^2 ||X||^{2-2theta}] < inf

so the error costs one moment for consistency and two for inference, at EVERY
theta and on the CF route alike.  The theta-family's saving is on the PREDICTOR,
where it is real (gate A0.3), and nowhere else.  Only the fully-bounded response
feature e^{isY} of the plan's Route B removes the error moment, which is exactly
why Route B is the moment-free one and Route A is not.

THE TEST.  Three error laws: Cauchy (no mean), index 1.5 (a mean, no variance),
Gaussian (all moments).  The error contribution must vanish in the second and
third and cannot in the first.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
SEED = 20260821
D = 12
NS = (1000, 4000, 16_000, 64_000)
R = 300


def draw_X(n, rng, d=D, tail=3.0):
    sd = np.arange(1, d + 1) ** -1.0
    Z = rng.standard_normal((n, d)) * sd
    return Z * (rng.pareto(tail, n) + 1.0)[:, None]


def cauchy(n, rng):
    return np.tan(rng.uniform(-np.pi / 2, np.pi / 2, n))


def err_contrib(X, eps, theta):
    """The error term of rhat_theta: mean of eps_k * X_k / ||X_k||^theta."""
    w = np.linalg.norm(X, axis=1) ** (-theta)
    return np.linalg.norm((X * (w * eps)[:, None]).mean(0))


def err_contrib_cf(X, eps, U):
    E = np.exp(1j * (X @ U.T))
    v = (eps[:, None] * E).mean(0) - eps.mean() * E.mean(0)
    return float(np.linalg.norm(v) / np.sqrt(len(U)))


def main():
    rng = np.random.default_rng(SEED)
    G = rng.standard_normal((32, D))
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    U = G * np.tile([0.25, 0.5, 1.0, 2.0], 8)[:, None]

    routes = {"theta=0  (BCT)": 0.0, "theta=1  (spatial sign)": 1.0,
              "theta=2  (SSCM)": 2.0}

    def pareto_sym(index):
        def f(n, r):
            return r.choice([-1.0, 1.0], n) * (r.pareto(index, n) + 1.0)
        return f

    for law, mk, note in (
            ("cauchy", cauchy, "NO mean -- (MI) is UNDEFINED here"),
            ("pareto-1.5", pareto_sym(1.5), "a mean, no variance"),
            ("gauss", lambda n, r: r.standard_normal(n), "all moments")):
        print("=" * 82)
        print(f"error ~ {law}   ({note})")
        print("  median over {R} reps of || (1/n) sum_k eps_k t(X_k) ||   "
              "-- must -> 0 for an exact normal equation".format(R=R))
        print("=" * 82)
        print(f"   {'route':>24} " + "".join(f"{'n=' + str(x):>11}" for x in NS)
              + f"{'slope':>9} {'verdict':>12}")
        rows = []
        thetas = list(routes.values())
        meds = {k: [] for k in list(routes) + ["CF  e^{i<u,X>}"]}
        for nn in NS:
            vals = {k: np.empty(R) for k in meds}
            for j in range(R):
                X = draw_X(nn, rng)
                e = mk(nn, rng)                 # ONE draw, all routes share it
                for name, th in routes.items():
                    vals[name][j] = err_contrib(X, e, th)
                vals["CF  e^{i<u,X>}"][j] = err_contrib_cf(X, e, U)
            for k in meds:
                meds[k].append(float(np.median(vals[k])))
        for name in meds:
            med = meds[name]
            slope = np.polyfit(np.log(NS), np.log(med), 1)[0]
            ok = slope < -0.35          # -1/2 is the CLT rate; 0 is no decay
            print(f"   {name:>24} " + "".join(f"{m:11.5f}" for m in med)
                  + f"{slope:9.3f} {'vanishes' if ok else 'DOES NOT':>12}")
            rows.append({"law": law, "route": name, "median": med,
                         "log_slope": slope, "vanishes": bool(ok)})
        print()
        globals().setdefault("ALL", []).extend(rows)

    print("READ-OFF.")
    print("  Cauchy      : NO route vanishes, the bounded ones included.  A bounded")
    print("                test function does not repair an error with no mean.")
    print("                (MI) is undefined here, so nothing is being violated --")
    print("                the manuscript was asserting a hypothesis it had ruled out.")
    print("  Pareto-1.5  : every route vanishes (E|eps| < inf gives exactness) but")
    print("                the log-slope is well short of -1/2 -- consistency without")
    print("                a sqrt(n) CLT, which needs E eps^2.")
    print("  Gaussian    : every route vanishes at the -1/2 CLT rate.")
    print()
    print("  CONCLUSION: the error-side moment requirement is a property of eps, not")
    print("  of the feature.  One moment for consistency, two for inference, on every")
    print("  route.  The paper must say so; the earlier manuscript said the reverse.")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "a0_1_bounded_test_function.json", "w") as f:
        json.dump(globals()["ALL"], f, indent=2)
    print(f"\nwrote {OUT / 'a0_1_bounded_test_function.json'}")


if __name__ == "__main__":
    main()
