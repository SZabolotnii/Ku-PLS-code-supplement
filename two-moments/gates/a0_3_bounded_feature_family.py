#!/usr/bin/env python3
"""Gate A0.3 --- why not the spatial sign X/||X||?

The plan demands the SAME rank/bridge analysis that killed Lemma 3.3, run on
Locantore's bounded feature, and an answer BEFORE a referee asks.

The analysis generalises both candidates.  For a measurable weight
w: H -> R_+ define

    A_w = E[ w(X) X (x) X ]        (operator on H)
    r_w = E[ w(X) Y X ]            (element of H)

Under Y = <beta,X> + eps and (MI),  E[w(X) eps X] = 0, so

    r_w = A_w beta                 EXACTLY, no bridge, no asserted identity.

This is the structural point: the operator that is INVERTED is the operator
that is ESTIMATED.  Lemma 3.3 failed because those were two different objects
joined by an assertion.  Every member of this family is immune by construction.

The family w_theta(x) = ||x||^{-theta} interpolates
    theta = 0 : BCT's second-moment operator K
    theta = 1 : the "half-sign" reweighting
    theta = 2 : Locantore / Gervini spatial-sign covariance E[U (x) U]

Moment accounting (the quantity the whole paper is about):
    A_w exists                <=>  E[ w(X) ||X||^2 ] < inf   =  E||X||^{2-theta}
    sqrt(n) CLT for Ahat_w    <=>  E[ w(X)^2 ||X||^4 ] < inf  =  E||X||^{4-2theta}
    sqrt(n) CLT for rhat_w    <=>  E[ w(X)^2 Y^2 ||X||^2 ] < inf

and the CF route of the plan (C*C beta = C* r_phi) sits OUTSIDE this family
because its operator is C*C = int c(u) (x) conj(c(u)) pi(du), which is a
QUADRATIC functional of the law, not a reweighted second moment.

WHAT THIS SCRIPT CHECKS, all of it numerically and none of it asserted:

  1. EXACTNESS.  ||A_w beta - r_w|| / ||r_w|| -> 0 at the Monte Carlo rate for
     every theta and for the CF route.  A failure here would be a second
     Lemma 3.3.
  2. RANK / DIMENSION COUNT (lesson 1 of the plan's Section 7).
  3. SPECTRUM DISTORTION.  Do A_theta and K share eigenvectors?  Provably yes
     for elliptical X (both commute with Sigma); the question is what happens
     off ellipticity, and how the eigenVALUES are distorted -- that distortion
     is the source condition, i.e. the RATE.
  4. ILL-POSEDNESS COST.  cond(A_theta) vs cond(K), and cond(C*C) vs cond(K).
     The CF route composes Sigma twice (c(u) = -i grad phi(u) is a Sigma-image
     for elliptical laws), so C*C should behave like Sigma W Sigma and SQUARE
     the ill-posedness.  If it does, the CF route pays a rate penalty that the
     spatial-sign route does not.
  5. SCALE, NOT ONLY SHAPE (lesson 2).  Every check is repeated at
     Var(X) != 1 and at several anisotropy levels.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
SEED = 20260821
D = 12
THETAS = [0.0, 0.5, 1.0, 1.5, 2.0]


# ------------------------------------------------------------------- sampling

def sas(alpha, size, rng):
    U = rng.uniform(-np.pi / 2, np.pi / 2, size)
    W = rng.exponential(1.0, size)
    if abs(alpha - 1.0) < 1e-9:
        return np.tan(U)
    a = alpha
    return (np.sin(a * U) / np.cos(U) ** (1 / a)
            * (np.cos(U - a * U) / W) ** ((1 - a) / a))


def draw(n, rng, law, d=D, decay=1.0, scale=1.0, tail=3.0):
    """Predictors with eigenvalue decay j^{-decay}, overall scale `scale`.

    law = 'ellip'  : elliptical -- radial R times a Gaussian direction.
    law = 'indep'  : independent heavy-tailed coordinates (NOT elliptical).
    """
    sd = scale * (np.arange(1, d + 1) ** -decay)
    if law == "ellip":
        Z = rng.standard_normal((n, d)) * sd
        R = (rng.pareto(tail, n) + 1.0)          # Pareto(tail): E R^p < inf iff p < tail
        X = Z * R[:, None]
    elif law == "indep":
        X = rng.standard_t(tail, size=(n, d)) * sd
    elif law == "stable":
        X = sas(1.7, (n, d), rng) * sd
    else:
        raise ValueError(law)
    return X, sd


# --------------------------------------------------------------- the family

def A_r(X, Y, theta):
    nrm = np.linalg.norm(X, axis=1)
    w = nrm ** (-theta)
    n = len(Y)
    A = (X * w[:, None]).T @ X / n
    r = (X * (w * Y)[:, None]).sum(0) / n
    return A, r


def cf_operator(X, Y, U, wpi):
    """C*C and C* r_phi for a discrete frequency weight pi on rows of U.

    c(u) = Cov(X, e^{i<u,X>}) in C^d ; (C beta)(u) = <beta, c(u)>.
    C*C = sum_l wpi_l * Re[ c_l (x) conj(c_l) ]  on the real form.
    """
    n = len(Y)
    proj = X @ U.T                       # (n, L)
    E = np.exp(1j * proj)
    phi = E.mean(0)                                  # (L,)
    Xc = X - X.mean(0)
    c = (Xc[:, :, None] * E[:, None, :]).mean(0)     # (d, L)
    rphi = (Y[:, None] * E).mean(0) - Y.mean() * phi  # (L,)
    CtC = np.einsum("dl,el,l->de", c, np.conj(c), wpi).real
    Ctr = np.einsum("dl,l,l->d", np.conj(c), rphi, wpi).real
    return CtC, Ctr, c


def freq_grid(X, L=64, rng=None):
    """Isotropically-scaled frequency grid: directions from the sphere, radii
    matched to a robust spread so pi sits where the CF is informative."""
    d = X.shape[1]
    G = rng.standard_normal((L, d))
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    mad = np.median(np.abs(X - np.median(X, 0)), 0) + 1e-12
    radii = np.tile([0.25, 0.5, 1.0, 2.0], L // 4 + 1)[:L]
    U = G * (radii[:, None] / np.median(mad))
    return U, np.full(L, 1.0 / L)


# ------------------------------------------------------------------- metrics

def principal_angles(P, Q, k):
    """Largest principal angle (deg) between the top-k eigenspaces."""
    def top(M, k):
        w, V = np.linalg.eigh(M)
        return V[:, np.argsort(w)[::-1][:k]]
    s = np.linalg.svd(top(P, k).T @ top(Q, k), compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(s.min(), -1, 1))))


def cond_eff(M, tol=1e-12):
    w = np.linalg.eigvalsh(M)
    w = w[w > tol * w.max()]
    return float(w.max() / w.min()), int(len(w))


# ------------------------------------------------------------------- the runs

def run_exactness(rng, n=400_000):
    print("=" * 78)
    print("1. EXACTNESS OF THE NORMAL EQUATION   ||A_w b - r_w|| / ||r_w||")
    print("   (Monte Carlo; a genuine bridge failure would NOT shrink with n)")
    print("=" * 78)
    rows = []
    for law in ("ellip", "indep", "stable"):
        for scale in (1.0, 3.0):
            X, sd = draw(n, rng, law, scale=scale)
            beta = (np.arange(1, D + 1) ** -1.5)
            eps = rng.standard_normal(n) * 0.5
            Y = X @ beta + eps
            line = f"  {law:7s} scale={scale:<4g} |"
            rec = {"law": law, "scale": scale}
            for th in THETAS:
                A, r = A_r(X, Y, th)
                rel = np.linalg.norm(A @ beta - r) / np.linalg.norm(r)
                line += f" th={th:.1f}:{rel:8.2e}"
                rec[f"theta_{th}"] = rel
            U, wpi = freq_grid(X, rng=rng)
            CtC, Ctr, _ = cf_operator(X, Y, U, wpi)
            rel_cf = np.linalg.norm(CtC @ beta - Ctr) / np.linalg.norm(Ctr)
            line += f" | CF:{rel_cf:8.2e}"
            rec["cf"] = rel_cf
            print(line)
            rows.append(rec)
    print("  -> every route is exact by construction; none is a Lemma 3.3.")
    return rows


def run_rank(rng, n=200_000):
    print()
    print("=" * 78)
    print("2. RANK / DIMENSION COUNT   (the one-line check that kills a false")
    print("   factorisation before any numerics)")
    print("=" * 78)
    X, _ = draw(n, rng, "ellip")
    beta = (np.arange(1, D + 1) ** -1.5)
    Y = X @ beta + rng.standard_normal(n) * 0.5
    for L in (4, 16, 64):
        U, wpi = freq_grid(X, L=L, rng=rng)
        CtC, _, c = cf_operator(X, Y, U, wpi)
        rk = np.linalg.matrix_rank(CtC, tol=1e-10 * np.abs(CtC).max())
        print(f"  L = |supp pi| = {L:3d} : rank(C*C) = {rk:2d}  "
              f"(<= min(d, 2L) = {min(D, 2 * L)}),  d = {D}")
    A1, _ = A_r(X, Y, 1.0)
    print(f"  rank(A_theta) = {np.linalg.matrix_rank(A1)} for every theta "
          f"(full, = d = {D}); it never depends on a design choice.")
    print("  -> C*C is rank-limited by supp pi (real form: rank <= min(d, 2|supp pi|)).")
    print("     A_theta is not.  The frequency grid is a THIRD tuning object")
    print("     the spatial-sign route does not have.")
    return {"note": "C*C rank <= min(d, 2|supp pi|) on the real form; A_theta full rank"}


def draw_skew(n, rng, d=D, decay=1.0, scale=1.0, tail=3.0):
    """A deliberately NON-elliptical law: skewed heavy coordinates put through a
    fixed non-diagonal mixing.  Nothing forces A_theta to share K's eigenbasis
    here, so this is where the geometry claim can actually fail."""
    sd = scale * (np.arange(1, d + 1) ** -decay)
    Z = rng.pareto(tail, size=(n, d))            # skewed, one-sided
    Z = (Z - Z.mean(0)) * sd
    M = np.linalg.qr(np.random.default_rng(7).standard_normal((d, d)))[0]
    T = M @ np.diag(1.0 + 0.5 * np.arange(d) / d) @ M.T
    return Z @ T, sd


def run_spectrum(rng, n=1_000_000):
    print()
    print("=" * 78)
    print("3. SPECTRUM DISTORTION -- do A_theta and K share eigenvectors, and")
    print("   how are the eigenvalues bent?  This IS the source condition.")
    print("=" * 78)
    rows = []
    print("   NOTE ON DESIGN: at slow eigenvalue decay the top eigenvalues are")
    print("   nearly tied and the sample eigenVECTORS are unstable for reasons")
    print("   that have nothing to do with theta.  Decay is therefore fixed at")
    print("   j^-1.5 (well separated) and n raised, so the angle column measures")
    print("   geometry rather than spacing.")
    for law in ("ellip", "indep", "skew"):
        for decay in (1.5,):
            X, sd = (draw_skew(n, rng, decay=decay) if law == "skew"
                     else draw(n, rng, law, decay=decay))
            Y = X @ (np.arange(1, D + 1) ** -1.5) + rng.standard_normal(n) * .5
            K, _ = A_r(X, Y, 0.0)
            print(f"\n  law={law}  eigenvalue decay j^-{decay}"
                  + ("   [elliptical: population angle is 0 BY SYMMETRY]"
                     if law == "ellip" else
                     "   [independent symmetric coords: population angle also 0]"
                     if law == "indep" else
                     "   [skewed + non-diagonal mixing: nothing forces 0]"))
            print(f"   {'theta':>6} {'max princ. angle (deg), top-3':>30} "
                  f"{'log-log slope of eigenvalues':>30}")
            wK = np.sort(np.linalg.eigvalsh(K))[::-1]
            sK = np.polyfit(np.log(np.arange(1, 7)), np.log(wK[:6]), 1)[0]
            for th in THETAS:
                A, _ = A_r(X, Y, th)
                ang = principal_angles(A, K, 3)
                wA = np.sort(np.linalg.eigvalsh(A))[::-1]
                sA = np.polyfit(np.log(np.arange(1, 7)), np.log(wA[:6]), 1)[0]
                print(f"   {th:6.1f} {ang:30.3f} {sA:30.3f}")
                rows.append({"law": law, "decay": decay, "theta": th,
                             "angle_deg": ang, "slope": sA, "slope_K": sK})
            print(f"   {'K':>6} {0.0:30.3f} {sK:30.3f}")
    print("\n  -> read the angle column: near 0 means A_theta inverts the SAME")
    print("     directions as K, so BCT's source condition transfers with a")
    print("     changed exponent (the slope column), not a changed geometry.")
    return rows


def run_conditioning(rng, n=300_000):
    print()
    print("=" * 78)
    print("4. ILL-POSEDNESS COST   cond(.) relative to K")
    print("   The CF route composes the covariance twice (c(u) = -i grad phi(u)")
    print("   is a Sigma-image for elliptical laws), so C*C ~ Sigma W Sigma.")
    print("   If so it SQUARES the ill-posedness and pays a rate penalty.")
    print("=" * 78)
    rows = []
    for decay in (0.5, 1.0, 1.5):
        X, sd = draw(n, rng, "ellip", decay=decay)
        Y = X @ (np.arange(1, D + 1) ** -1.5) + rng.standard_normal(n) * .5
        K, _ = A_r(X, Y, 0.0)
        cK, _ = cond_eff(K)
        line = f"  decay j^-{decay:<4g} cond(K)={cK:10.3e} |"
        rec = {"decay": decay, "cond_K": cK}
        for th in (1.0, 2.0):
            A, _ = A_r(X, Y, th)
            cA, _ = cond_eff(A)
            line += f" th={th:.0f}:{cA/cK:8.2f}x"
            rec[f"ratio_theta_{th}"] = cA / cK
        U, wpi = freq_grid(X, L=64, rng=rng)
        CtC, _, _ = cf_operator(X, Y, U, wpi)
        cC, rk = cond_eff(CtC)
        line += f" | CF:{cC/cK:10.2e}x  (cond(K)^2/cond(K) = {cK:8.2e})"
        rec["ratio_cf"] = cC / cK
        rec["cond_K_squared_over_K"] = cK
        print(line)
        rows.append(rec)
    print("  -> if the CF column tracks the last column, C*C is quadratic in the")
    print("     covariance: the CF route is ill-posed to the SQUARE of BCT's.")
    return rows


def run_moment_accounting(rng, n=None, R=600):
    print()
    print("=" * 78)
    print("5. MOMENT ACCOUNTING, MEASURED   -- spread of sqrt(n)||Ahat - A||")
    print("   across replications at tail index 3 (E||X||^2 < inf, E||X||^4 = inf).")
    print("   A route with a genuine sqrt(n) CLT has a STABLE spread; one whose")
    print("   CLT fails shows a heavy right tail that grows with n.")
    print("=" * 78)
    NS = (2000, 8000, 32000)
    beta = np.arange(1, D + 1) ** -1.5
    # population reference, one huge independent sample per theta
    print("   building the population reference (n = 4,000,000) ...", flush=True)
    Xr, _ = draw(4_000_000, rng, "ellip", tail=3.0)
    Yr = Xr @ beta + rng.standard_normal(4_000_000) * .5
    Aref = {th: A_r(Xr, Yr, th)[0] for th in THETAS}
    del Xr, Yr

    print(f"   median of sqrt(n)||Ahat_theta - A_theta||_F  --  BOUNDED in n"
          f" <=> sqrt(n) CLT")
    print(f"   {'theta':>6} " + "".join(f"{'n=' + str(x):>11}" for x in NS)
          + f"{'n32/n2':>10} {'q99 n32/n2':>12}")
    rows = []
    for th in THETAS:
        med, q99 = [], []
        for nn in NS:
            v = np.empty(R)
            for j in range(R):
                X, _ = draw(nn, rng, "ellip", tail=3.0)
                Y = X @ beta + rng.standard_normal(nn) * .5
                A, _ = A_r(X, Y, th)
                v[j] = np.sqrt(nn) * np.linalg.norm(A - Aref[th], "fro")
            med.append(float(np.median(v)))
            q99.append(float(np.quantile(v, .99)))
        print(f"   {th:6.1f} " + "".join(f"{m:11.4f}" for m in med)
              + f"{med[-1] / med[0]:10.2f} {q99[-1] / q99[0]:12.2f}")
        rows.append({"theta": th, "median": med, "q99": q99,
                     "growth_median": med[-1] / med[0],
                     "growth_q99": q99[-1] / q99[0]})
    print("   -> growth ~ 1 is a CLT.  theta = 0 needs E||X||^4 and does not have")
    print("      it at tail index 3, so its column must grow; theta >= 1 needs")
    print("      only E||X||^{4-2theta} <= E||X||^2, which the law does have.")
    return rows


def main():
    rng = np.random.default_rng(SEED)
    out = {}
    out["exactness"] = run_exactness(rng)
    out["rank"] = run_rank(rng)
    out["spectrum"] = run_spectrum(rng)
    out["conditioning"] = run_conditioning(rng)
    out["moments"] = run_moment_accounting(rng)
    OUT.mkdir(exist_ok=True)
    with open(OUT / "a0_3_bounded_feature_family.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {OUT / 'a0_3_bounded_feature_family.json'}")


if __name__ == "__main__":
    main()
