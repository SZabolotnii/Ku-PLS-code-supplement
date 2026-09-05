#!/usr/bin/env python3
"""Gate A0.3(b) --- the RESPONSE side of the moment accounting.

A0.3 settled the operator side.  The cross element is the other half, and after
A0.4 it is the binding half: the FRED response |Y_t| has a dependence-robust 95%
tail-index interval of roughly [1.96, 3.48], which does NOT exclude 2.  E Y^2 is
therefore NOT established by the very data the paper uses, and E Y^2 is exactly
what the plan's Route A needs for rhat_phi.

Accounting to be checked, not asserted:

    r_theta = E[ ||X||^{-theta} Y X ]
    summand norm  = ||X||^{1-theta} |Y|
    sqrt(n) CLT   <=>  E[ ||X||^{2-2theta} Y^2 ] < inf

    theta = 0 (BCT)          : E[ ||X||^2 Y^2 ]        ~ a fourth moment
    theta = 1                : E[ Y^2 ]                 <- the plan's Route A
    theta = 2 (spatial sign) : E[ Y^2 / ||X||^2 ]       <- weaker than E Y^2

    CF route rhat_phi(u) = mean of Y_k e^{i<u,X_k>} : summand norm |Y_k|,
    so the CF route needs E Y^2, the SAME as theta = 1 and strictly more than
    theta = 2.  The bounded X-feature buys nothing on the response side.

Design: eps is drawn with its own tail index so that E Y^2 can be switched off
while the X side is held fixed.  A route with a genuine CLT keeps
sqrt(n)||rhat - r|| bounded in n; one without shows it grow.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
SEED = 20260821
D = 12
THETAS = [0.0, 1.0, 2.0]
NS = (2000, 8000, 32000)
R = 600


def draw_X(n, rng, d=D, decay=1.0, tail=3.0):
    sd = np.arange(1, d + 1) ** -decay
    Z = rng.standard_normal((n, d)) * sd
    Rr = rng.pareto(tail, n) + 1.0
    return Z * Rr[:, None]


def eps_draw(n, rng, tail):
    """Symmetric Pareto-tailed error with index `tail`: E|eps|^p < inf iff p<tail."""
    s = rng.choice([-1.0, 1.0], n)
    return s * (rng.pareto(tail, n) + 1.0)


def r_theta(X, Y, theta):
    w = np.linalg.norm(X, axis=1) ** (-theta)
    return (X * (w * Y)[:, None]).mean(0)


def r_cf(X, Y, U):
    E = np.exp(1j * (X @ U.T))
    return (Y[:, None] * E).mean(0) - Y.mean() * E.mean(0)


def freq_grid(rng, L=32, d=D):
    G = rng.standard_normal((L, d))
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    return G * np.tile([0.25, 0.5, 1.0, 2.0], L // 4)[:L][:, None]


def main():
    rng = np.random.default_rng(SEED)
    beta = np.arange(1, D + 1) ** -1.5
    U = freq_grid(rng)
    out = []

    for eps_tail, label in ((4.0, "E eps^2 < inf   (index 4)"),
                            (2.5, "E eps^2 < inf   (index 2.5, near the FRED value)"),
                            (1.6, "E eps^2 = inf   (index 1.6)")):
        print("=" * 78)
        print(f"error tail index {eps_tail}   --   {label}")
        print("=" * 78)
        print("   building the population reference (n = 3,000,000) ...", flush=True)
        Xr = draw_X(3_000_000, rng, tail=3.0)
        Yr = Xr @ beta + eps_draw(3_000_000, rng, eps_tail)
        ref = {th: r_theta(Xr, Yr, th) for th in THETAS}
        ref_cf = r_cf(Xr, Yr, U)
        del Xr, Yr

        print(f"   median of sqrt(n)||rhat - r||   --   bounded in n <=> sqrt(n) CLT")
        print(f"   {'route':>18} " + "".join(f"{'n=' + str(x):>11}" for x in NS)
              + f"{'n32/n2':>9}")
        for th in THETAS:
            med = []
            for nn in NS:
                v = np.empty(R)
                for j in range(R):
                    X = draw_X(nn, rng, tail=3.0)
                    Y = X @ beta + eps_draw(nn, rng, eps_tail)
                    v[j] = np.sqrt(nn) * np.linalg.norm(r_theta(X, Y, th) - ref[th])
                med.append(float(np.median(v)))
            name = {0.0: "theta=0  (BCT)", 1.0: "theta=1", 2.0: "theta=2  (sign)"}[th]
            print(f"   {name:>18} " + "".join(f"{m:11.4f}" for m in med)
                  + f"{med[-1] / med[0]:9.2f}")
            out.append({"eps_tail": eps_tail, "route": name, "median": med,
                        "growth": med[-1] / med[0]})
        med = []
        for nn in NS:
            v = np.empty(R)
            for j in range(R):
                X = draw_X(nn, rng, tail=3.0)
                Y = X @ beta + eps_draw(nn, rng, eps_tail)
                v[j] = np.sqrt(nn) * np.linalg.norm(r_cf(X, Y, U) - ref_cf)
            med.append(float(np.median(v)))
        print(f"   {'CF  r_phi':>18} " + "".join(f"{m:11.4f}" for m in med)
              + f"{med[-1] / med[0]:9.2f}")
        out.append({"eps_tail": eps_tail, "route": "CF r_phi", "median": med,
                    "growth": med[-1] / med[0]})
        print()

    print("READ-OFF: the CF cross element tracks theta=1, not theta=2.  Bounding")
    print("the X-feature does nothing for the response side -- rhat_phi still")
    print("carries the unbounded factor Y.  Only theta=2 divides it out.")
    OUT.mkdir(exist_ok=True)
    with open(OUT / "a0_3b_response_side.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT / 'a0_3b_response_side.json'}")


if __name__ == "__main__":
    main()
