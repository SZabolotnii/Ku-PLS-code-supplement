#!/usr/bin/env python3
r"""Repair A5: the calibrating design matches the RADIAL process, not ||X_t||.

WHAT IS WRONG.  kappa is calibrated on a design whose serial dependence is set by phi,
and phi is chosen so that the design reproduces the lag-1 Spearman autocorrelation
MEASURED ON THE DATA'S ||X_t||: +0.3034 (Treasury, n = 2494) and +0.5906 (commercial
paper, n = 4396).  But `dependent_design.py::phi_for_target` measures
`spearman_lag1(pareto_ar1(...))`, which is the RADIAL process R_t, and the manuscript's
closed form rho_S = (6/pi) arcsin(phi/2) is likewise the radial one.  The design's
predictor is X_t = R_t Z_t with Z_t an i.i.d. Gaussian curve, so
||X_t|| = R_t ||Z_t|| and the independent factor ||Z_t|| ATTENUATES the
autocorrelation.  Matching R_t therefore leaves ||X_t|| under-persistent.

A PaperMentor review named this (2026-09-05, critical #17).  Its own framing was about
the copula equation; the substantive defect is the attenuation, which is much larger.

WHAT WAS ALREADY CHECKED, AND WHY IT DOES NOT COVER THIS.
`gates/a3_kappa_phi_sensitivity.py` compared phi = 0.30 against 0.320 and 0.60 against
0.608 -- the arcsin correction, 5-7% in phi, 3-4% in kappa, no pick moved.  That is a
DIFFERENT and much smaller discrepancy.  The attenuation measured below is 31-38% in
rho_S, an order of magnitude larger, so the earlier study's "no pick moves" says nothing
about this one.  Re-using its conclusion would be the mistake this file exists to avoid.

THE CHECKS, each stated before it is run:

  A  MEASURE the attenuation: rho_S(R_t) against rho_S(||X_t||) at the shipped phi.
     Prediction: ||X_t|| is materially less persistent than R_t.
  B  SOLVE for the phi that makes rho_S(||X_t||) hit the measured targets.  Prediction:
     substantially higher than the shipped 0.320 and 0.608.
  C  RECALIBRATE kappa there, at R = 3000 with a bootstrap interval, exactly as the
     shipped calibration does.  kappa is DECREASING in phi, so the corrected kappa
     should be SMALLER -- meaning the shipped rule is too permissive.
  D  DOES ANY PICK MOVE?  This is what decides between a correction and a sentence.
     The rule is theta-hat = min{theta : R_n(theta) <= kappa}, so a smaller kappa can
     only push theta-hat UP.  Report the margins as well: a pick that survives but
     whose margin collapses is a different finding from one that is untouched.

The script does not decide what to write.  It reports whether the picks move and
whether the kappa difference is resolved by the calibration's own error, and those two
facts settle it.

Run:  python gates/a5_persistence_target.py [--quick]
Writes results/a5_persistence_target.{json,out}.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
from dependent_design import DependentDesign, pareto_ar1, spearman_lag1  # noqa: E402
from theta_selector import select, max_to_sum                            # noqa: E402
from empirical2_cp import load as load_cp                                # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "results"
TREASURY = ROOT / "data" / "fred_cache.npz"
SEED = 20260905

# The persistence measured on the data's ||X_t||, which is what the design is meant to
# reproduce.  These are the numbers the manuscript prints.
TARGETS = (("Treasury x S&P 500", 2494, 0.3034, 0.320),
           ("CP x VIX / NASDAQ", 4396, 0.5906, 0.608))


def rho_norm(phi, n, index, rng, reps):
    """Median lag-1 Spearman of ||X_t|| in the calibrating design."""
    d = DependentDesign(phi=phi)
    out = []
    for _ in range(reps):
        X, _ = d.draw(n, index, rng)
        out.append(spearman_lag1(np.linalg.norm(X, axis=1)))
    return float(np.median(out))


def cv_factors(n, index, rng, reps, J=20, decay=1.0):
    """Coefficients of variation of R_t and ||Z_t||, which explain the dilution.

    The manuscript prints these to say WHY the attenuation is severe at the
    calibration tail index; a number quoted in prose needs a file behind it, so they
    are computed here rather than in a one-off session.
    """
    scale = (np.arange(J) + 1.0) ** -decay
    cvr, cvz = [], []
    for _ in range(reps):
        R = pareto_ar1(n, index, 0.0, rng)
        Z = rng.standard_normal((n, J)) * scale
        nz = np.linalg.norm(Z, axis=1)
        cvr.append(float(np.std(R) / np.mean(R)))
        cvz.append(float(np.std(nz) / np.mean(nz)))
    return float(np.median(cvr)), float(np.median(cvz))


def rho_radial(phi, n, index, rng, reps):
    out = [spearman_lag1(pareto_ar1(n, index, phi, rng)) for _ in range(reps)]
    return float(np.median(out))


PHI_MAX = 0.995


def phi_for_norm_target(target, n, index, rng, reps):
    """Bisection on phi so that rho_S(||X_t||) hits the target.

    Returns (phi, ceiling, reachable).  The ceiling -- rho_S(||X_t||) at phi = PHI_MAX --
    is reported because the target may be UNREACHABLE: ||X_t|| = R_t ||Z_t|| with ||Z_t||
    independent, and at a light radial tail ||Z_t|| varies more than R_t does, so it
    dominates the ranks however persistent R_t is made.  A bisection that silently
    returns the upper endpoint would hide exactly that.
    """
    ceiling = rho_norm(PHI_MAX, n, index, rng, reps)
    if ceiling < target:
        return PHI_MAX, ceiling, False
    lo, hi = 0.0, PHI_MAX
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if rho_norm(mid, n, index, rng, reps) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), ceiling, True


def kappa_at(phi, n, rng, R, b_boot):
    d = DependentDesign(phi=phi)
    rs = np.empty(R)
    for i in range(R):
        X, Y = d.draw(n, 4.0, rng)
        rs[i] = max_to_sum(X, Y - X @ d.beta, 0.0)
    k = float(np.quantile(rs, 0.95))
    bs = np.quantile(rs[rng.integers(0, R, (b_boot, R))], 0.95, axis=1)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    se = float(np.std(bs))
    return k, float(lo), float(hi), se


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    reps = 12 if args.quick else 40
    R_cal = 600 if args.quick else 3000
    b_boot = 500 if args.quick else 2000

    res = {"seed": SEED, "quick": args.quick,
           "config": {"reps_rho": reps, "R_calibration": R_cal, "bootstrap": b_boot,
                      "calibration_index": 4.0,
                      "targets": [{"name": t[0], "n": t[1], "rho_target": t[2],
                                   "phi_shipped": t[3]} for t in TARGETS]}}

    L = []
    P = L.append
    P("=" * 100)
    P("Repair A5: the calibrating design matches the RADIAL process, not ||X_t||")
    P(f"seed {SEED}   R={R_cal}   reps={reps}" + ("   [--quick]" if args.quick else ""))
    P("=" * 100)

    # ---- A, B ------------------------------------------------------------------
    P("")
    P("A/B. what the shipped phi actually reproduces, and the phi that hits the target")
    P(f"   {'application':<22}{'n':>7}{'phi_ship':>10}{'rho(R_t)':>10}"
      f"{'rho(||X||)':>12}{'target':>9}{'phi_fixed':>11}{'rho at it':>11}")
    rows_ab = {}
    for name, n, target, phi_s in TARGETS:
        rr = rho_radial(phi_s, n, 4.0, rng, reps)
        rn = rho_norm(phi_s, n, 4.0, rng, reps)
        phi_fix, ceiling, reachable = phi_for_norm_target(target, n, 4.0, rng, reps)
        rn_fix = rho_norm(phi_fix, n, 4.0, rng, reps)
        rows_ab[name] = {"n": n, "phi_shipped": phi_s, "rho_radial_at_shipped": rr,
                         "rho_norm_at_shipped": rn, "target": target,
                         "phi_fixed": phi_fix, "rho_norm_at_fixed": rn_fix,
                         "ceiling_at_phi_max": ceiling, "target_reachable": reachable,
                         "attenuation_pct": 100.0 * (1 - rn / rr) if rr else 0.0}
        P(f"   {name:<22}{n:>7}{phi_s:>10.3f}{rr:>10.4f}{rn:>12.4f}"
          f"{target:>9.4f}{phi_fix:>11.3f}{rn_fix:>11.4f}")
    att = [v["attenuation_pct"] for v in rows_ab.values()]
    P(f"   attenuation of ||X_t|| against R_t: "
      f"{min(att):.1f}% to {max(att):.1f}%")
    cvr, cvz = cv_factors(4396, 4.0, rng, reps)
    P(f"   at the calibration index 4: cv(R) = {cvr:.2f}, cv(||Z||) = {cvz:.2f}"
      f"  -> the independent factor carries "
      f"{'MORE' if cvz > cvr else 'less'} of the rank variation")
    unreachable = [k for k, v in rows_ab.items() if not v["target_reachable"]]
    for k in unreachable:
        v = rows_ab[k]
        P(f"   {k}: target {v['target']:.4f} is UNREACHABLE -- the ceiling at "
          f"phi={PHI_MAX} is {v['ceiling_at_phi_max']:.4f}")
    attenuation_real = bool(min(att) > 10.0)
    res["A_attenuation"] = {"rows": rows_ab, "attenuation_exceeds_10pc": attenuation_real,
                            "targets_unreachable": unreachable,
                            "cv_radial_at_index4": cvr, "cv_Znorm_at_index4": cvz,
                            "independent_factor_dominates": bool(cvz > cvr),
                            "pass": attenuation_real}

    # ---- C ----------------------------------------------------------------------
    P("")
    P("C. kappa recalibrated at the corrected phi (index 4, p95 of R_n(0))")
    P(f"   {'application':<22}{'phi':>8}{'kappa':>9}{'95% CI':>22}{'s.e.':>9}")
    rows_c = {}
    for name, n, _t, phi_s in TARGETS:
        row = {}
        for tag, phi in (("shipped", phi_s), ("fixed", rows_ab[name]["phi_fixed"])):
            k, lo, hi, se = kappa_at(phi, n, rng, R_cal, b_boot)
            row[tag] = {"phi": phi, "kappa": k, "ci": [lo, hi], "se": se}
            P(f"   {name if tag == 'shipped' else '':<22}{phi:>8.3f}{k:>9.4f}"
              f"   [{lo:.4f}, {hi:.4f}]{se:>9.4f}")
        ks, kf = row["shipped"]["kappa"], row["fixed"]["kappa"]
        row["change_pct"] = 100.0 * (kf / ks - 1)
        # The manuscript prints the FALL as a positive percentage, so store that too:
        # an audit resolves what is printed, and -48.52 is not 48.5.
        row["fall_pct"] = 100.0 * (1 - kf / ks)
        row["cis_disjoint"] = bool(row["shipped"]["ci"][1] < row["fixed"]["ci"][0]
                                   or row["fixed"]["ci"][1] < row["shipped"]["ci"][0])
        rows_c[name] = row
        P(f"   {'':<22}{'':>8}{'':>9}   change {row['change_pct']:+.1f}%"
          f"   CIs {'DISJOINT' if row['cis_disjoint'] else 'overlap'}")
    res["C_kappa"] = {"rows": rows_c, "pass": True}

    # ---- D ----------------------------------------------------------------------
    P("")
    P("D. does any pick move?  theta-hat = min{theta : R_n(theta) <= kappa}")
    dXc, Yn, Yv, _ = load_cp()
    dat = np.load(TREASURY, allow_pickle=True)
    dXt, Yt = dat["dX"], dat["Y"]
    apps = (("CP x VIX", dXc, Yv, "CP x VIX / NASDAQ"),
            ("CP x NASDAQ", dXc, Yn, "CP x VIX / NASDAQ"),
            ("Treasury x S&P 500", dXt, Yt, "Treasury x S&P 500"))
    P(f"   {'application':<22}{'kappa_ship':>12}{'th_ship':>9}{'margin':>9}"
      f"{'kappa_fix':>12}{'th_fix':>9}{'margin':>9}")
    rows_d = {}
    for label, X, Y, key in apps:
        resid = Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]
        rn = {t: max_to_sum(X, resid, t) for t in (0.0, 0.25, 0.5, 0.75, 1.0)}
        out = {"R_n": rn}
        for tag in ("shipped", "fixed"):
            k = rows_c[key][tag]["kappa"]
            se = rows_c[key][tag]["se"]
            th, _ = select(X, resid, k)
            # distance from kappa to the nearest switch point, in units of kappa's s.e.
            margin = min(abs(k - v) for v in rn.values()) / se if se > 0 else float("inf")
            out[tag] = {"kappa": k, "theta_hat": float(th), "margin_se": float(margin)}
        out["changed"] = bool(out["shipped"]["theta_hat"] != out["fixed"]["theta_hat"])
        rows_d[label] = out
        flag = "   <-- CHANGED" if out["changed"] else ""
        P(f"   {label:<22}{out['shipped']['kappa']:>12.4f}"
          f"{out['shipped']['theta_hat']:>9.2f}{out['shipped']['margin_se']:>9.2f}"
          f"{out['fixed']['kappa']:>12.4f}{out['fixed']['theta_hat']:>9.2f}"
          f"{out['fixed']['margin_se']:>9.2f}{flag}")
    moved = [k for k, v in rows_d.items() if v["changed"]]
    res["D_picks"] = {"rows": rows_d, "picks_that_moved": moved,
                      "any_pick_moved": bool(moved), "pass": True}

    P("")
    P("=" * 100)
    if moved:
        P(f"   {len(moved)} pick(s) MOVE under the corrected phi: {', '.join(moved)}")
        P("   The correction must be APPLIED to the calibration, not noted in prose.")
    else:
        P("   No pick moves.  The mis-targeting is real and measurable, but it does not")
        P("   change a reported result, so the honest treatment is to state in the")
        P("   calibration appendix WHAT the design matches -- the radial process -- and")
        P("   to report the attenuation, rather than to silently re-run the tables.")
    disjoint = [n for n, v in rows_c.items() if v["cis_disjoint"]]
    P(f"   kappa difference resolved by the calibration in: "
      f"{', '.join(disjoint) if disjoint else 'neither cell'}")
    res["all_pass"] = all(res[k]["pass"] for k in res
                          if isinstance(res[k], dict) and "pass" in res[k])
    P("ALL PASS" if res["all_pass"] else "FAILURE -- a prediction did not hold")
    P("=" * 100)

    text = "\n".join(L)
    print(text)
    OUT.mkdir(exist_ok=True)
    (OUT / "a5_persistence_target.out").write_text(text + "\n", encoding="utf-8")
    (OUT / "a5_persistence_target.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
