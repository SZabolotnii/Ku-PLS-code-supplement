#!/usr/bin/env python3
"""A1.7 verification --- Theorem 6' tested in ITS OWN regime, and one sharp prediction.

Three things, in order of how much they could embarrass the theorem.

  SECTION 1 -- SIZE UNDER (O) vs SIZE AT AN EARLY-STOPPED m.
  The A2 pilot ran its size table at m = 3.  Part 3 of a1_7_residual_identity.py
  shows the overfitting condition needs 6-8 CG steps, so that table was computed
  OUTSIDE the theorem's regime and says nothing about it either way.  This runs
  both, side by side, so the difference is visible rather than assumed.

  SECTION 2 -- THE PREDICTION THE THEOREM MAKES AND THE OLD FRAMING FORBIDS.
  Theorem 6' never uses Ahat -> A.  Its only moment condition is
  tr V_theta = E[eps^2 ||X||^{2-2theta}] < inf, which at theta = 1 is E eps^2 --
  no moment of X at all.  So the theta = 1 test should be VALID at predictor tail
  index 1.5, where E||X||^2 = inf and Ahat_1 itself has no sqrt(n) CLT, and where
  BCT's weights (eigenvalues of E[eps^2 X (x) X], trace sigma^2 E||X||^2) DO NOT
  EXIST.  That is the alpha-stable regime the original manuscript targeted.  If it
  holds, the effect is far larger below index 2 than the thin (2,4) window the A2
  pilot found.  If it fails, the theorem is wrong somewhere and this finds it.

  SECTION 3 -- WHAT theta COSTS IN POWER.
  The limit under beta = b + h/sqrt(n) is ||G_theta + A_theta h||^2.  Lowering the
  moment requirement moves the non-centrality from Kh to A_theta h.  Local
  asymptotic power is computed from population A_theta and V_theta, for h on
  leading and on trailing eigendirections, because the two must differ in sign.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "code"))
from twomoments import Design, operator, cg  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
SEED = 20260823
THETAS = [0.0, 0.5, 1.0, 1.5, 2.0]
LEVEL = 0.05


# ------------------------------------------------------------- the statistic

def stat_under_O(X, Y, b, theta, a_n, m_max=40):
    """T_n at the first m whose residual falls below a_n = o(n^{-1/2}), plus the
    plug-in spectrum of Vhat_theta.  Returns (T_n, weights, m used)."""
    n = len(Y)
    A, r = operator(X, Y, theta)
    for m in range(1, m_max + 1):
        bh = cg(A, r, m)
        if np.linalg.norm(r - A @ bh) <= a_n:
            break
    stat = n * float(np.sum((A @ (bh - b)) ** 2))
    w = np.linalg.norm(X, axis=1) ** (-theta) if theta else np.ones(n)
    Z = X * (w * (Y - X @ b))[:, None]          # = w(X_k) eps_k X_k exactly under H0
    V = Z.T @ Z / n
    ev = np.clip(np.linalg.eigvalsh(V), 0.0, None)
    return stat, ev[ev > 1e-14], m


def stat_fixed_m(X, Y, b, theta, m):
    n = len(Y)
    A, r = operator(X, Y, theta)
    bh = cg(A, r, m)
    stat = n * float(np.sum((A @ (bh - b)) ** 2))
    w = np.linalg.norm(X, axis=1) ** (-theta) if theta else np.ones(n)
    Z = X * (w * (Y - X @ b))[:, None]
    ev = np.clip(np.linalg.eigvalsh(Z.T @ Z / n), 0.0, None)
    return stat, ev[ev > 1e-14]


def crit(ev, rng, level=LEVEL, ndraw=3000):
    if ev.size == 0:
        return np.inf
    return float(np.quantile((rng.chisquare(1.0, (ndraw, ev.size)) * ev).sum(1),
                             1.0 - level))


# ------------------------------------------------------------------ sections

def section1(rng, R, design, indices, n=1000):
    print("=" * 92)
    print(f"SECTION 1 -- empirical size, nominal {LEVEL:.0%}, n = {n}, R = {R}")
    print("   LEFT  block: the theorem's regime, stopping at ||Rhat|| <= n^{-1/2}/log n")
    print("   RIGHT block: the practical regime, fixed m = 3 (OUTSIDE Theorem 6')")
    print("=" * 92)
    se = np.sqrt(LEVEL * (1 - LEVEL) / R)
    a_n = 1.0 / (np.sqrt(n) * np.log(n))
    print(f"   s.e. = {se:.4f}, tolerance [{LEVEL-2*se:.3f}, {LEVEL+2*se:.3f}], "
          f"a_n = {a_n:.2e}")
    hdr = (f"   {'theta':>7} |" + "".join(f"{'idx ' + str(i):>9}" for i in indices)
           + "  |" + "".join(f"{'idx ' + str(i):>9}" for i in indices))
    print(hdr)
    print("   " + "-" * (len(hdr) - 3))
    rows = {}
    for th in THETAS:
        under_O, at_m3 = [], []
        for index in indices:
            rO = rm = 0
            for _ in range(R):
                X, Y = design.draw(n, index, rng)
                s, ev, _ = stat_under_O(X, Y, design.beta, th, a_n)
                rO += s > crit(ev, rng)
                s2, ev2 = stat_fixed_m(X, Y, design.beta, th, 3)
                rm += s2 > crit(ev2, rng)
            under_O.append(rO / R)
            at_m3.append(rm / R)
        rows[th] = {"under_O": under_O, "m3": at_m3}
        mk = lambda v: "".join(f"{x:8.3f}" + ("*" if abs(x - LEVEL) > 2 * se else " ")
                               for x in v)
        tag = "(BCT)" if th == 0 else "(sgn)" if th == 2 else "     "
        print(f"   {th:4.1f}{tag}|{mk(under_O)} |{mk(at_m3)}")
    print("   * = outside +-2 s.e. of nominal")
    return rows


def section2(rng, R, design, n=1000):
    print()
    print("=" * 92)
    print("SECTION 2 -- the prediction: theta = 1 valid where BCT's WEIGHTS DO NOT EXIST")
    print("   At predictor tail index < 2, E||X||^2 = inf, so tr V_0 = sigma^2 E||X||^2")
    print("   is infinite and BCT's limit law has no weights.  tr V_1 = sigma^2 is finite.")
    print("   Theorem 6' never uses Ahat -> A, so theta = 1 should hold its level anyway.")
    print("=" * 92)
    a_n = 1.0 / (np.sqrt(n) * np.log(n))
    se = np.sqrt(LEVEL * (1 - LEVEL) / R)
    idx = [1.2, 1.5, 1.8, 2.5]
    print(f"   {'theta':>7} |" + "".join(f"{'idx ' + str(i):>10}" for i in idx)
          + f"   {'tr Vhat behaviour':>24}")
    print("   " + "-" * 78)
    rows = {}
    for th in (0.0, 1.0, 2.0):
        size, tr_growth = [], []
        for index in idx:
            rej = 0
            trs = []
            for _ in range(R):
                X, Y = design.draw(n, index, rng)
                s, ev, _ = stat_under_O(X, Y, design.beta, th, a_n)
                rej += s > crit(ev, rng)
                trs.append(ev.sum())
            size.append(rej / R)
            tr_growth.append(float(np.median(trs)))
        rows[th] = {"size": size, "tr": tr_growth}
        tag = "(BCT)" if th == 0 else "(sgn)" if th == 2 else "     "
        mk = "".join(f"{x:9.3f}" + ("*" if abs(x - LEVEL) > 2 * se else " ")
                     for x in size)
        print(f"   {th:4.1f}{tag}|{mk}   tr Vhat @1.2 = {tr_growth[0]:8.2f}")
    print("   * = outside +-2 s.e. of nominal")
    print()
    print("   tr Vhat at n = 1000 across index (median): a column that GROWS without")
    print("   settling is the signature of an infinite population trace.")
    for th in (0.0, 1.0, 2.0):
        print(f"     theta = {th:.0f}: " + "  ".join(f"{v:9.3f}" for v in rows[th]["tr"]))
    return rows


def section3(rng, design, n_pop=2_000_000):
    print()
    print("=" * 92)
    print("SECTION 3 -- local asymptotic power: what lowering the moment costs")
    print("   Limit under beta = b + h/sqrt(n) is ||G_theta + A_theta h||^2.")
    print("   Population A_theta and V_theta from n = 2e6; power by 200k draws of the limit.")
    print("=" * 92)
    index = 3.0
    X, Y = design.draw(n_pop, index, rng)
    eps = Y - X @ design.beta
    J = design.J
    pops = {}
    for th in THETAS:
        w = np.linalg.norm(X, axis=1) ** (-th) if th else np.ones(n_pop)
        A = (X * w[:, None]).T @ X / n_pop
        Z = X * (w * eps)[:, None]
        V = Z.T @ Z / n_pop
        pops[th] = (A, V)
    del X, Y, eps

    K = pops[0.0][0]
    evK, VK = np.linalg.eigh(K)
    order = np.argsort(evK)[::-1]
    lead, trail = VK[:, order[0]], VK[:, order[J - 3]]

    print(f"   {'direction':>12} {'||h||':>7} |" +
          "".join(f"{'th=' + str(t):>10}" for t in THETAS))
    rows = {}
    for name, hdir in (("leading", lead), ("trailing", trail),
                       ("beta itself", design.beta / np.linalg.norm(design.beta))):
        for scale in (2.0, 6.0):
            line = f"   {name:>12} {scale:7.1f} |"
            vals = []
            for th in THETAS:
                A, V = pops[th]
                ev, U = np.linalg.eigh(V)
                ev = np.clip(ev, 0, None)
                h = scale * hdir
                mu = U.T @ (A @ h)                       # non-centrality in V's basis
                D = rng.standard_normal((200_000, J)) * np.sqrt(ev) + mu
                stat = (D ** 2).sum(1)
                null = ((rng.standard_normal((200_000, J)) * np.sqrt(ev)) ** 2).sum(1)
                c = np.quantile(null, 1 - LEVEL)
                p = float((stat > c).mean())
                vals.append(p)
                line += f"{p:10.3f}"
            print(line)
            rows[f"{name}_{scale}"] = vals
    print()
    print("   -> read across each row: this is the price of theta, in power, at a")
    print("      tail index where BOTH tests are valid.  A theta that wins on the")
    print("      leading direction and loses on the trailing one is trading, not")
    print("      dominating, and the paper must say which.")
    return rows


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    rng = np.random.default_rng(SEED)
    design = Design(J=20, decay=1.0)
    out = {}
    out["size"] = {str(k): v for k, v in
                   section1(rng, R, design, [2.5, 3.0, 5.0]).items()}
    out["below_two"] = {str(k): v for k, v in section2(rng, R, design).items()}
    out["power"] = section3(rng, design)
    OUT.mkdir(exist_ok=True)
    with open(OUT / "a1_7_verify.json", "w") as f:
        json.dump({"R": R, "level": LEVEL, "seed": SEED, **out}, f, indent=2, default=float)
    print(f"\nwrote {OUT / 'a1_7_verify.json'}")


if __name__ == "__main__":
    main()
