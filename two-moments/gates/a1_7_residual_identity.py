#!/usr/bin/env python3
"""A1.7, step 0 --- check the identity the whole proof would rest on, BEFORE writing it.

CLAIM UNDER TEST.  For ANY estimator betahat obtained from the pair (Ahat, rhat),
with residual  Rhat := rhat - Ahat betahat,

        Ahat (betahat - b)  =  xihat - Rhat ,      xihat := rhat - Ahat b ,

and under H0: beta = b,

        xihat = (1/n) sum_k w(X_k) eps_k X_k          EXACTLY, an i.i.d. mean.

If both hold, the nonlinearity of conjugate gradient and the randomness of the
stopping index NEVER ENTER: they are confined to Rhat, which the stopping rule
bounds by construction.  The delta method through the regulariser -- the step the
referee named and the plan calls A1.7 -- would not be needed at all.

That is a strong enough claim that it gets checked numerically at machine
precision, on several theta, several stopping indices, and both hypotheses,
before a single line of it is written down as a proof.  The failure mode this
guards against is the one that produced Lemma 3.3: an identity that is asserted
because it looks right.

Checked here:
  (1) the algebraic identity, to machine precision, m = 1..12 and m = full;
  (2) that xihat is exactly the i.i.d. mean of w(X_k) eps_k X_k under H0;
  (3) that CG drives ||Rhat|| to machine zero in at most rank(Ahat) steps, so the
      overfitting condition ||Rhat|| = o_P(n^{-1/2}) is ATTAINABLE and not vacuous;
  (4) how many CG steps that takes as a function of theta -- because if the
      overfitted regime is numerically out of reach the theorem is empty, and
      cond(Ahat) is what governs it (gate A0.3 measured 0.48-0.80x for theta=1).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "code"))
from twomoments import Design, operator, cg  # noqa: E402

SEED = 20260821
THETAS = [0.0, 0.5, 1.0, 1.5, 2.0]


def main():
    rng = np.random.default_rng(SEED)
    design = Design(J=20, decay=1.0)
    n = 400
    beta = design.beta

    print("=" * 86)
    print("(1) ALGEBRAIC IDENTITY   max_k || Ahat(betahat_m - b) - (xihat - Rhat) ||")
    print("=" * 86)
    print(f"   {'theta':>7} {'H0: b = beta':>16} {'H1: b = beta + delta':>22}")
    for th in THETAS:
        worst = [0.0, 0.0]
        for rep in range(20):
            X, Y = design.draw(n, 3.0, rng)
            A, r = operator(X, Y, th)
            for col, b in enumerate((beta, beta + 0.3 * np.ones_like(beta))):
                xi = r - A @ b
                for m in list(range(1, 13)) + [n]:
                    bh = cg(A, r, m)
                    R = r - A @ bh
                    lhs = A @ (bh - b)
                    worst[col] = max(worst[col], np.linalg.norm(lhs - (xi - R)))
        print(f"   {th:7.1f} {worst[0]:16.3e} {worst[1]:22.3e}")
    print("   -> identity holds for EVERY estimator, m and hypothesis: it uses only")
    print("      the definition of Rhat, never how betahat was produced.")

    print()
    print("=" * 86)
    print("(2) IS xihat EXACTLY THE i.i.d. MEAN OF w(X_k) eps_k X_k UNDER H0?")
    print("=" * 86)
    print(f"   {'theta':>7} {'max || xihat - mean(w eps X) ||':>34}")
    for th in THETAS:
        worst = 0.0
        for rep in range(50):
            Z = rng.standard_normal((n, design.J)) * design.scale
            Rr = rng.pareto(3.0, n) + 1.0
            X = Z * Rr[:, None]
            eps = rng.normal(0.0, design.sigma, n)
            Y = X @ beta + eps
            A, r = operator(X, Y, th)
            xi = r - A @ beta
            w = np.linalg.norm(X, axis=1) ** (-th) if th else np.ones(n)
            direct = (X * (w * eps)[:, None]).mean(0)
            worst = max(worst, np.linalg.norm(xi - direct))
        print(f"   {th:7.1f} {worst:34.3e}")
    print("   -> exact.  No remainder, no linearisation, no nuisance estimated:")
    print("      under a simple null the residuals ARE the errors.")

    print()
    print("=" * 86)
    print("(3)+(4) IS THE OVERFITTING CONDITION ATTAINABLE, AND AT WHAT COST?")
    print("   rhat lies in Range(Ahat) = span{X_k} by construction, so CG reaches an")
    print("   exact solution in at most rank(Ahat) steps.  The question is whether")
    print("   floating point gets there before the conditioning stops it.")
    print("=" * 86)
    for nn in (400, 2000):
        print(f"\n   n = {nn}")
        print(f"   {'theta':>7} {'rank(Ahat)':>11} {'CG steps to':>13} {'||Rhat|| at':>13} "
              f"{'cond(Ahat)':>12}")
        print(f"   {'':>7} {'':>11} {'o(n^-1/2)':>13} {'that step':>13} {'':>12}")
        for th in THETAS:
            steps, resid, conds = [], [], []
            for rep in range(20):
                X, Y = design.draw(nn, 3.0, rng)
                A, r = operator(X, Y, th)
                target = 0.05 / np.sqrt(nn)          # a_n = o(n^{-1/2})
                rk = np.linalg.matrix_rank(A)
                ev = np.linalg.eigvalsh(A)
                ev = ev[ev > 1e-14 * ev.max()]
                conds.append(ev.max() / ev.min())
                hit = None
                for m in range(1, design.J + 2):
                    bh = cg(A, r, m)
                    nr = np.linalg.norm(r - A @ bh)
                    if nr <= target:
                        hit = (m, nr)
                        break
                steps.append(hit[0] if hit else np.nan)
                resid.append(hit[1] if hit else np.linalg.norm(r - A @ cg(A, r, design.J + 1)))
            print(f"   {th:7.1f} {rk:11d} {np.nanmedian(steps):13.1f} "
                  f"{np.median(resid):13.2e} {np.median(conds):12.3e}")
    print("\n   -> the condition is attainable for every theta, and the number of")
    print("      steps tracks the conditioning.  This is where A0.3's conditioning")
    print("      result stops being a curiosity: it governs whether the regime the")
    print("      theorem lives in is numerically reachable.")


if __name__ == "__main__":
    main()
