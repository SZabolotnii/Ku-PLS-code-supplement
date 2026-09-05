#!/usr/bin/env python3
"""Prop 6, closed: what BCT's proof actually uses, and the one thing it does not.

READING BCT's APPENDIX (arXiv:2402.11134, functional_pls.tex).  Their adaptivity
chain is Lemma A.1 (Chebyshev in H) -> Lemma A.2 (two probability bounds) ->
Lemma A.3 (operator perturbation) + Lemma A.4 (residual polynomials) -> Lemma A.5
(residual) -> Lemma A.6 (|Q'|) -> Theorem 3.  The DATA enter in exactly one place,
Lemma A.2, and there in exactly two bounds:

   ||rhat - Khat beta|| = ||n^-1 sum eps_i X_i||  <= sigma sqrt(2 E||X||^2/(gamma n))
                                                    ^ a CRUDE BOUND on
                                                      tr V_0 = E[eps^2 ||X||^2]
   ||Khat - K||_HS                                <=       sqrt(2 E||X||^4/(gamma n))

Everything downstream is operator-level: it needs self-adjointness, positivity,
compactness, and those two numbers.  Nothing else in the chain looks at X.  So the
transport to A_theta is substitution, with the two numbers re-evaluated:

   (M1)  E||X||^{4-2theta} < inf                        replaces  E||X||^4 < inf
   (M2)  tr V_theta = E[eps^2 ||X||^{2-2theta}] < inf   replaces  sigma^2 E||X||^2

which is the moment ledger of theory/A0-operator-theory.md, arrived at from the
other end.  BCT's own threshold is sigma sqrt(2 E||X||^2/(delta n)) -- the DATA
noise level -- so the rule now in twomoments.discrepancy_stop is BCT's rule with
the crude bound replaced by the quantity it bounds.  The superseded rule dropped
sigma altogether and was never BCT's rule.

WHAT DOES NOT TRANSPORT BY SUBSTITUTION is Assumption 3: the complexity class
S(mu,R,C) is stated for K, via beta = K^mu w and the eigenbasis (v_j) of K.  At
theta > 0 the class is stated for A_theta.  Two questions decide the cost.

  Q1  Do A_theta and K share an eigenbasis?  If not, S_theta and S_0 are
      incomparable and Theorem 3 would be a rate over a class that moves with
      theta -- which would make the comparison to BCT meaningless.
  Q2  If they do share it, does the eigenvalue map change the DECAY EXPONENT?
      If yes, mu moves with theta and so does the rate.  If it only rescales the
      top of the spectrum, S_theta = S_0 with the same mu: the rate exponent of
      Theorem 3 is preserved and theta buys the moments and the constant.

Q1 has a proof, not a measurement.  Let (v_j) diagonalise K and write X in that
basis.  If the coordinates (X_j) are INDEPENDENT and each SYMMETRIC about 0, then
for j != k,  E[||X||^{-theta} X_j X_k] = E[ h(X_j,X_k) X_j X_k ] where
h(x,y) = E[(x^2+y^2+S)^{-theta/2}] is even in each argument separately, so the
expectation factorises into E[h(.,X_k) X_j] * ... and vanishes by oddness in X_j.
Hence A_theta is diagonal in (v_j).  The hypothesis is INDEPENDENCE + SYMMETRY,
which is weaker than ellipticity and is what the elliptical case supplies.  The
run below checks the algebra, and -- because the first version of this script used
a "non-elliptical" design that also had independent symmetric coordinates and so
could not possibly break it -- includes a design that violates SYMMETRY, which is
where it must and does fail.

Q2 must not be answered by fitting a slope over j = 1..20: the reweighting shrinks
the top of the spectrum, and a log-log line through 20 points reads that shrinkage
as a change of slope.  It is answered here on the population diagonal
d_j(theta) = E[||Z||^{-theta} Z_j^2] / (sigma_j^2 E[||Z||^{-theta}]), the exact
shrinkage factor, over J = 200 coordinates.  If d_j -> 1, the tail exponent is
untouched.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
from twomoments import operator  # noqa: E402

OUT = HERE.parent / "results"
SEED = 20260911
THETAS = [0.0, 0.5, 1.0, 1.5, 2.0]


def principal_angle(U, V):
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(s.min(), -1.0, 1.0))))


# ------------------------------------------------------------------- Q1 designs

def coords_gauss(n, scale, rng):
    """independent, symmetric (the elliptical case)"""
    return rng.standard_normal((n, len(scale))) * scale


def coords_laplace(n, scale, rng):
    """independent, symmetric, NOT Gaussian -- the hypothesis is symmetry, not
    ellipticity, so this must also give a zero angle."""
    return rng.laplace(0.0, 1.0 / np.sqrt(2.0), (n, len(scale))) * scale


def coords_skewed(n, scale, rng):
    """independent, mean zero, ASYMMETRIC -- centred exponential.  This is the
    design that must break diagonality, and it is the honest scope limit."""
    return (rng.exponential(1.0, (n, len(scale))) - 1.0) * scale


def _rot(J, ang=np.pi / 4.0):
    c, s_ = np.cos(ang), np.sin(ang)
    Rot = np.eye(J)
    Rot[0, 0] = c; Rot[0, 1] = -s_; Rot[1, 0] = s_; Rot[1, 1] = c
    return Rot


def coords_direction_mixture(n, scale, rng):
    """NOT a scale mixture of one law: two Gaussian components whose covariances
    do NOT commute, so no single basis diagonalises the conditional laws.  Its
    population covariance is NOT diagonal in the coordinate basis, which is why
    it needs its own reference basis below -- the first version of this row read
    that non-diagonality as an effect of theta."""
    J = len(scale)
    Z = rng.standard_normal((n, J)) * scale
    Rot = _rot(J)
    pick = rng.uniform(size=n) < 0.5
    Z[pick] = Z[pick] @ Rot.T * 3.0
    return Z


def basis_identity(J, scale):
    return np.eye(J)


def basis_mixture(J, scale):
    """Analytic eigenbasis of K for coords_direction_mixture:
    E[Z (x) Z] = 0.5 * Sigma + 0.5 * 9 * Rot Sigma Rot^T,  Sigma = diag(scale^2)."""
    Sig = np.diag(scale ** 2)
    Rot = _rot(J)
    M = 0.5 * Sig + 0.5 * 9.0 * (Rot @ Sig @ Rot.T)
    w, V = np.linalg.eigh(M)
    return V[:, np.argsort(w)[::-1]]


def offdiag_mass(X, theta, basis):
    """||A_theta - diag(A_theta)||_HS / ||A_theta||_HS in a KNOWN population basis.

    The basis is supplied analytically, never estimated, so nothing here is
    circular.  At theta = 0 the quantity is exactly 0 in population, so the
    theta = 0 column IS the Monte-Carlo noise floor -- and a real off-diagonal is
    O(1) in n while the floor is O(n^{-1/2}), which the two sample sizes separate.
    """
    A, _ = operator(X, np.zeros(len(X)), theta)
    A = basis.T @ A @ basis
    tot = np.linalg.norm(A)
    off = np.sqrt(max(tot ** 2 - np.sum(np.diag(A) ** 2), 0.0))
    return float(off / tot)


def q1(n, J, decay, index, rng):
    scale = (np.arange(J) + 1.0) ** -decay
    laws = (("independent SYMMETRIC (Gaussian)", coords_gauss, basis_identity),
            ("independent SYMMETRIC (Laplace)", coords_laplace, basis_identity),
            ("independent ASYMMETRIC (centred exp)", coords_skewed, basis_identity),
            ("non-commuting 2-component mixture", coords_direction_mixture,
             basis_mixture))
    print("=" * 100)
    print("Q1 -- is A_theta diagonal in the eigenbasis of K?")
    print("     off-diagonal HS mass as a fraction of ||A_theta||_HS, in the")
    print("     POPULATION basis.  theta = 0 is the Monte-Carlo noise floor.")
    print("     A real off-diagonal does NOT shrink when n grows; the floor does.")
    print("=" * 100)
    rows = {}
    for name, gen, mkbasis in laws:
        rows[name] = {}
        B = mkbasis(J, scale)
        for nn in (n // 4, n):
            Z = gen(nn, scale, rng)
            R = rng.pareto(index, nn) + 1.0
            X = Z * R[:, None]
            vals = [offdiag_mass(X, th, B) for th in THETAS]
            rows[name][nn] = vals
        print(f"   {name}")
        for nn in (n // 4, n):
            print(f"      n = {nn:>9,} |"
                  + "".join(f"{v:11.5f}" for v in rows[name][nn])
                  + "   <- theta = " + ", ".join(f"{t:.1f}" for t in THETAS))
        f0 = rows[name][n // 4][2] / max(rows[name][n][2], 1e-12)
        print(f"      ratio of the theta=1 column between the two n: {f0:5.2f}"
              f"   (2.00 = pure noise, 1.00 = a real off-diagonal)")
    print()
    print("   Independence + symmetry of the coordinates in the eigenbasis of K is")
    print("   the hypothesis, and it is weaker than ellipticity: the Laplace row")
    print("   satisfies it and the Gaussian row is not special.  A first version of")
    print("   this script called the centred-exponential row a counterexample on the")
    print("   strength of a principal ANGLE, which the wide eigenvalue gaps of a")
    print("   j^-2 spectrum make insensitive; the mass measured here is the")
    print("   quantity that decides it.")
    return rows


# -------------------------------------------------------------------- Q2 exact

def q2(n, J, decay, chunk, rng):
    """Population shrinkage factor d_j(theta), computed on the diagonal only.

    lambda_j(A_theta) = E[R^{2-theta}] * E[||Z||^{-theta} Z_j^2]  (elliptical),
    so the whole theta-dependence of the SHAPE of the spectrum is in
        d_j(theta) = E[||Z||^{-theta} Z_j^2] / (sigma_j^2 E[||Z||^{-theta}]).
    d_j == 1 for all j would mean theta rescales the spectrum and nothing else.
    """
    scale = (np.arange(J) + 1.0) ** -decay
    acc = {th: np.zeros(J) for th in THETAS}
    accw = {th: 0.0 for th in THETAS}
    done = 0
    while done < n:
        m = min(chunk, n - done)
        Z = rng.standard_normal((m, J)) * scale
        nrm = np.linalg.norm(Z, axis=1)
        Z2 = Z ** 2
        for th in THETAS:
            w = nrm ** (-th) if th else np.ones(m)
            acc[th] += w @ Z2
            accw[th] += w.sum()
        done += m
    print("=" * 94)
    print(f"Q2 -- does theta change the DECAY EXPONENT or only the top of the")
    print(f"      spectrum?   shrinkage d_j = E[||Z||^-th Z_j^2]/(sig_j^2 E[||Z||^-th])")
    print(f"      population diagonal, J = {J}, {n:,} draws")
    print("=" * 94)
    probe = [1, 2, 5, 10, 25, 50, 100, 150, 200]
    probe = [p for p in probe if p <= J]
    print(f"   {'theta':>6} |" + "".join(f"{'j=' + str(p):>9}" for p in probe)
          + f"{'tail slope':>13}{'  (j in [J/2, J])':>18}")
    rows = []
    for th in THETAS:
        d = acc[th] / (accw[th] * scale ** 2)
        jj = np.arange(1, J + 1)
        lo = J // 2
        sl = float(np.polyfit(np.log(jj[lo:]), np.log(d[lo:]), 1)[0])
        print(f"   {th:6.2f} |" + "".join(f"{d[p-1]:9.4f}" for p in probe)
              + f"{sl:13.5f}")
        rows.append({"theta": th, "d": [float(d[p - 1]) for p in probe],
                     "tail_slope": sl})
    print()
    print("   d_j -> 1 with a tail slope indistinguishable from 0 means the eigenvalue")
    print("   map is lambda_j -> c(theta) * lambda_j * (1 + o(1)): the DECAY EXPONENT")
    print("   is untouched, S_theta = S_0 with the same mu, and Theorem 3 transports")
    print("   with the SAME RATE.  What theta buys is the moment conditions and the")
    print("   condition number, not a faster rate -- and that is the honest claim.")
    return {"probe": probe, "rows": rows}


def main():
    nq1 = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000
    nq2 = int(sys.argv[2]) if len(sys.argv) > 2 else 4_000_000
    rng = np.random.default_rng(SEED)
    r1 = q1(nq1, 20, 1.0, 4.0, rng)
    print()
    r2 = q2(nq2, 200, 1.0, 200_000, rng)
    OUT.mkdir(exist_ok=True)
    with open(OUT / "a1_6b_source_condition.json", "w") as f:
        json.dump({"q1": r1, "q2": r2, "n_q1": nq1, "n_q2": nq2}, f, indent=2)
    print(f"wrote {OUT / 'a1_6b_source_condition.json'}")


if __name__ == "__main__":
    main()
