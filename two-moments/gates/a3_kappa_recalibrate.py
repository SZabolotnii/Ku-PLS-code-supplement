#!/usr/bin/env python3
"""kappa, recalibrated at R = 3000 and at the phi the copula relation implies.

This closes open problem 6 of A3-OUTLINE.md.  Three changes from the shipped
calibration, and only the first is a correction:

  1. R = 3000 rather than 400.  At R = 400 kappa's own Monte Carlo s.e. is about
     10% of kappa (0.0104 and 0.0074), and gates/a3_kappa_margin.py showed the
     Treasury pick sitting 1.19 s.e. from a switch -- i.e. unresolved by its own
     calibration.  The null here is light-tailed and simulated once, so the fix
     costs seven times almost nothing.
  2. phi = 0.320 / 0.608 rather than 0.30 / 0.60.  The copula-AR(1) design gives
     rho_S = (6/pi) arcsin(phi/2) ~ 0.955 phi, not rho_S = phi; the measured
     lag-1 Spearman autocorrelations are +0.3034 (Treasury) and +0.5906 (CP),
     from gates/a3_persistence_artifact.py.  Known to be immaterial to every
     theta-hat (gates/a3_kappa_phi_sensitivity.py); applied because it is right,
     not because it changes anything.
  3. A bootstrap interval is reported for kappa, and the MARGIN of each
     application's theta-hat is reported in units of that interval.  A selection
     rule's pick is not a result unless the calibration resolves it.

Everything downstream that depends on kappa is re-checked here in one place: the
level of the rule under dependence, the picks on the three application panels, and
the margins.  If the level moves, the recalibration is not free and must be
reported as a change of result rather than a tightening.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
from dependent_design import DependentDesign            # noqa: E402
from theta_selector import select, max_to_sum, GRID     # noqa: E402
from empirical2_cp import load as load_cp               # noqa: E402
# THE test, imported rather than reimplemented.  A first version of this script
# defined its own stat_and_crit at a FIXED m = 3 and reported the rule's size as
# 0.33 at tail index 1.2.  That is not Theorem 6''s statistic: the theorem needs
# the overfitting condition ||rhat - Ahat betahat|| <= a_n = 1/(sqrt(n) log n), and
# at m = 3 the residual R-hat is not negligible, so T_n is not the score statistic
# and the weighted-chi^2 calibration does not apply to it.  Never reimplement the
# estimand while changing the tuning constant.
from run_theta_selector import stat_and_crit             # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "results"
TREASURY = ROOT / "data" / "fred_cache.npz"
SEED = 20260914
LEVEL = 0.05
B_BOOT = 2000

# (label, phi_shipped, phi_implied, n)   -- phi_implied from the copula relation
CELLS = (("i.i.d. reference", 0.00, 0.000, 1000),
         ("Treasury x S&P 500", 0.30, 0.320, 2494),
         ("CP panels", 0.60, 0.608, 4396))

SHIPPED_KAPPA = {"i.i.d. reference": 0.1336,
                 "Treasury x S&P 500": 0.0966,
                 "CP panels": 0.0777}


def kappa_at(phi, n, rng, R):
    d = DependentDesign(phi=phi)
    rs = np.empty(R)
    for i in range(R):
        X, Y = d.draw(n, 4.0, rng)
        rs[i] = max_to_sum(X, Y - X @ d.beta, 0.0)
    k = float(np.quantile(rs, .95))
    bs = np.quantile(rs[rng.integers(0, R, (B_BOOT, R))], .95, axis=1)
    lo, hi = np.quantile(bs, [.025, .975])
    return k, float(lo), float(hi), float(np.std(bs))


def main():
    R_CAL = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    R_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    rng = np.random.default_rng(SEED)
    out = {"R_cal": R_CAL, "R_size": R_SIZE}

    # ------------------------------------------------------------------ step 1
    print("=" * 104)
    print(f"STEP 1 -- kappa at R = {R_CAL}, at both the shipped and the implied phi")
    print("=" * 104)
    print(f"   {'cell':<22}{'n':>7}{'phi':>8}{'kappa':>9}{'95% CI':>20}"
          f"{'s.e.':>9}{'shipped':>10}{'|z|':>9}{'':>7}")
    kap = {}
    for label, ps, pi, n in CELLS:
        rec = {"n": n}
        for tag, phi in (("shipped", ps), ("implied", pi)):
            k, lo, hi, se = kappa_at(phi, n, rng, R_CAL)
            rec[tag] = {"phi": phi, "kappa": k, "ci": [lo, hi], "se": se}
        sk = SHIPPED_KAPPA[label]
        # The shipped value is ONE draw of an R = 400 estimator, so the right
        # comparison is |shipped - kappa_3000| against the R = 400 standard error,
        # not against the R = 3000 confidence interval.  A first version compared
        # it to the interval and flagged two of three cells as inconsistent; that
        # test rejects a correct R = 400 draw roughly a third of the time.
        se400 = rec["shipped"]["se"] * np.sqrt(R_CAL / 400.0)
        z = abs(sk - rec["shipped"]["kappa"]) / se400
        rec["shipped_kappa"] = sk
        rec["se_at_R400"] = float(se400)
        rec["z_vs_shipped"] = float(z)
        inci = z <= 2.0
        kap[label] = rec
        for tag in ("shipped", "implied"):
            e = rec[tag]
            head = label if tag == "shipped" else ""
            nn = n if tag == "shipped" else ""
            tail = (f"{sk:10.4f}{z:>9.2f}{('  ok' if inci else '  FLAG'):>7}"
                    if tag == "shipped" else "")
            print(f"   {head:<22}{str(nn):>7}{e['phi']:>8.3f}{e['kappa']:>9.4f}"
                  f"   [{e['ci'][0]:.4f}, {e['ci'][1]:.4f}]{e['se']:>9.4f}{tail}")
    out["kappa"] = kap
    print()
    print("   |z| = |kappa_shipped - kappa_3000| / (the R = 400 standard error).")
    print("   A FLAG would mean the shipped calibration was BIASED rather than")
    print("   merely noisy, and every downstream number would need re-running.")

    # ------------------------------------------------------------------ step 2
    print()
    print("=" * 104)
    print(f"STEP 2 -- does the LEVEL of the rule survive the new kappa?  "
          f"n = 1000, R = {R_SIZE}, phi = 0.608")
    print("=" * 104)
    se = np.sqrt(LEVEL * (1 - LEVEL) / R_SIZE)
    knew = kap["CP panels"]["implied"]["kappa"]
    kold = SHIPPED_KAPPA["CP panels"]
    print(f"   tolerance +-2 s.e. = [{LEVEL-2*se:.4f}, {LEVEL+2*se:.4f}]"
          f"   kappa_new = {knew:.4f}  vs  kappa_shipped = {kold:.4f}")
    print(f"\n   {'index':>7}{'size (new)':>13}{'size (shipped)':>17}"
          f"{'mean th-hat (new)':>20}{'(shipped)':>12}")
    d = DependentDesign(phi=0.608)
    lvl = {}
    for idx in (1.2, 1.5, 2.5, 4.0):
        rn = ro = 0
        tn, to = [], []
        for _ in range(R_SIZE):
            X, Y = d.draw(1000, idx, rng)
            resid = Y - X @ d.beta
            for kk, acc, ths in ((knew, "n", tn), (kold, "o", to)):
                th, _ = select(X, resid, kk)
                ths.append(th)
                s, c = stat_and_crit(X, Y, d.beta, th, rng)
                if acc == "n":
                    rn += s > c
                else:
                    ro += s > c
        fn = "*" if abs(rn / R_SIZE - LEVEL) > 2 * se else " "
        fo = "*" if abs(ro / R_SIZE - LEVEL) > 2 * se else " "
        lvl[str(idx)] = {"size_new": rn / R_SIZE, "size_shipped": ro / R_SIZE,
                         "theta_new": float(np.mean(tn)),
                         "theta_shipped": float(np.mean(to))}
        print(f"   {idx:7.1f}{rn/R_SIZE:12.4f}{fn}{ro/R_SIZE:16.4f}{fo}"
              f"{np.mean(tn):20.3f}{np.mean(to):12.3f}")
    out["level"] = lvl
    print("   * = outside +-2 s.e. of nominal")

    # ------------------------------------------------------------------ step 3
    print()
    print("=" * 104)
    print("STEP 3 -- picks and MARGINS on the applications, under the new kappa")
    print("=" * 104)
    dXc, Yn, Yv, _ = load_cp()
    dd = np.load(TREASURY, allow_pickle=True)
    apps = (("CP x VIX", dXc, Yv, "CP panels"),
            ("CP x NASDAQ", dXc, Yn, "CP panels"),
            ("Treasury x S&P 500", dd["dX"], dd["Y"], "Treasury x S&P 500"))
    print(f"   {'application':<20}" + "".join(f"{'R_n(' + f'{t:.2f}' + ')':>11}"
                                              for t in GRID)
          + f"{'kappa':>9}{'th-hat':>8}{'margin/se':>11}{'was':>6}")
    marg = {}
    for label, X, Y, key in apps:
        resid = Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]
        r = np.array([max_to_sum(X, resid, t) for t in GRID])
        k = kap[key]["implied"]["kappa"]
        sek = kap[key]["implied"]["se"]
        ok = np.where(r <= k)[0]
        j = int(ok[0]) if len(ok) else len(GRID) - 1
        up = float(r[j - 1]) if j > 0 else np.inf
        m = min(abs(k - up), abs(k - float(r[j]))) / sek
        okd = np.where(r <= SHIPPED_KAPPA[key])[0]
        was = GRID[int(okd[0])] if len(okd) else GRID[-1]
        marg[label] = {"R_n": [float(x) for x in r], "kappa": k, "se": sek,
                       "theta_hat": GRID[j], "theta_hat_shipped": was,
                       "margin_in_se": float(m), "moved": bool(GRID[j] != was)}
        print(f"   {label:<20}" + "".join(f"{v:11.5f}" for v in r)
              + f"{k:9.4f}{GRID[j]:8.2f}{m:11.2f}{was:6.2f}")
    out["margins"] = marg
    print()
    print("   'margin/se' is now in units of the R = 3000 standard error, so it is")
    print("   the number that decides whether each pick is resolved.  'was' is the")
    print("   pick under the shipped kappa; a difference is a change of result.")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "a3_kappa_recalibrate.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT / 'a3_kappa_recalibrate.json'}")


if __name__ == "__main__":
    main()
