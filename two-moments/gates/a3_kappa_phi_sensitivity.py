#!/usr/bin/env python3
"""Does the phi mis-mapping change any theta-hat?  Asked, not assumed.

`gates/a3_persistence_artifact.py` showed the persistence targets were matched by
setting phi = rho_S directly, whereas the copula-AR(1) design satisfies
rho_S = (6/pi) arcsin(phi/2) ~ 0.955 phi.  The measured rho_S are +0.3034
(Treasury, n = 2494) and +0.5906 (commercial paper, n = 4396), so the phi that
actually reproduce them are 0.320 and 0.608, not 0.30 and 0.60.

That is a 5-7% error in phi.  It is only worth a line in the paper if it moves
something.  kappa is DECREASING in phi, so understating phi overstates kappa and
makes the selector too permissive -- it would accept a SMALLER theta than it
should, which for CP x VIX is the direction that would weaken the headline.  So
the question is not rhetorical.

FIRST ATTEMPT, RECORDED BECAUSE IT WAS WRONG.  Run at R = 400 -- the R the shipped
calibration used -- this reported that the Treasury pick moves from 0.75 to 0.50
and concluded "the correction must be applied".  But at the SAME phi and n it also
produced kappa = 0.0846 where the shipped run produced 0.0966: a 12% spread from
the seed alone.  A 34% "effect" of moving phi by 0.02 was therefore not an effect.
**kappa is a 95th percentile estimated from R draws, and at R = 400 its own Monte
Carlo error is larger than the thing being measured.**  The comparison is only
meaningful with kappa estimated precisely enough to resolve it, so this version
runs at R = 3000 and reports a bootstrap interval for kappa itself.  A sensitivity
analysis whose noise exceeds its signal answers a different question than the one
asked.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
from dependent_design import DependentDesign          # noqa: E402
from theta_selector import select, max_to_sum         # noqa: E402
from empirical2_cp import load as load_cp             # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "results"
TREASURY = ROOT / "data" / "fred_cache.npz"
SEED = 20260913
R_CAL = 3000
R_SHIPPED = 400          # what the original calibration used
B_BOOT = 2000


def kappa_at(phi, n, rng, R=R_CAL):
    """kappa = p95 of R_n(0) at index 4.0, with a bootstrap CI for the quantile."""
    d = DependentDesign(phi=phi)
    rs = np.empty(R)
    for i in range(R):
        X, Y = d.draw(n, 4.0, rng)
        rs[i] = max_to_sum(X, Y - X @ d.beta, 0.0)
    k = float(np.quantile(rs, .95))
    bs = np.quantile(rs[rng.integers(0, R, (B_BOOT, R))], .95, axis=1)
    lo, hi = np.quantile(bs, [.025, .975])
    se_shipped = float(np.std(
        np.quantile(rs[rng.integers(0, R, (B_BOOT, R_SHIPPED))], .95, axis=1)))
    return k, float(lo), float(hi), se_shipped


def main():
    rng = np.random.default_rng(SEED)
    cases = (("Treasury x S&P 500", 2494, 0.30, 0.320),
             ("CP x VIX / NASDAQ", 4396, 0.60, 0.608))

    print("=" * 92)
    print(f"kappa at the phi USED vs the phi IMPLIED by the measured rho_S")
    print(f"   same design, same n, R = {R_CAL} as the shipped calibration")
    print("=" * 92)
    print(f"   {'application':<22}{'n':>7}{'phi':>7}{'kappa':>9}"
          f"{'95% CI for kappa':>22}{'s.e. at R=400':>15}")
    kaps = {}
    for name, n, pu, pi in cases:
        row = {"n": n}
        for tag, phi in (("used", pu), ("implied", pi)):
            k, lo, hi, se4 = kappa_at(phi, n, rng)
            row[tag] = {"phi": phi, "kappa": k, "ci": [lo, hi], "se_at_R400": se4}
            print(f"   {name if tag == 'used' else '':<22}{n if tag == 'used' else '':>7}"
                  f"{phi:>7.3f}{k:>9.4f}   [{lo:.4f}, {hi:.4f}]{se4:>15.4f}")
        ku, ki = row["used"]["kappa"], row["implied"]["kappa"]
        sep = (row["used"]["ci"][1] < row["implied"]["ci"][0]
               or row["implied"]["ci"][1] < row["used"]["ci"][0])
        row["change_pct"] = (ki / ku - 1) * 100
        row["cis_disjoint"] = bool(sep)
        kaps[name] = row
        print(f"   {'':<22}{'':>7}{'':>7}{'':>9}   change {row['change_pct']:+.1f}%"
              f"   CIs {'DISJOINT' if sep else 'overlap'}")

    print()
    print("=" * 92)
    print("what the selector picks on the applications under each kappa")
    print("=" * 92)
    dXc, Yn, Yv, _ = load_cp()
    d = np.load(TREASURY, allow_pickle=True)
    dXt, Yt = d["dX"], d["Y"]
    apps = (("CP x VIX", dXc, Yv, "CP x VIX / NASDAQ"),
            ("CP x NASDAQ", dXc, Yn, "CP x VIX / NASDAQ"),
            ("Treasury x S&P 500", dXt, Yt, "Treasury x S&P 500"))
    print(f"   {'application':<22}{'th-hat (kappa used)':>22}{'th-hat (kappa impl.)':>23}")
    picks = {}
    for label, X, Y, key in apps:
        resid = Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]
        tu, _ = select(X, resid, kaps[key]["used"]["kappa"])
        ti, _ = select(X, resid, kaps[key]["implied"]["kappa"])
        picks[label] = {"theta_used": tu, "theta_implied": ti,
                        "changed": bool(tu != ti)}
        flag = "   <-- CHANGED" if tu != ti else ""
        print(f"   {label:<22}{tu:>22.2f}{ti:>23.2f}{flag}")

    print()
    moved = [k for k, v in picks.items() if v["changed"]]
    noisy = [n for n, v in kaps.items() if not v["cis_disjoint"]]
    if moved and not noisy:
        print("   The correction MOVES a pick and the kappa difference is resolved")
        print("   by the calibration.  It must be applied, not noted.")
    elif moved:
        print(f"   A pick moves ({', '.join(moved)}) but the two kappa are NOT")
        print("   separated by their own confidence intervals, so the move is not")
        print("   attributable to the phi correction.  What this actually shows is")
        print("   that theta-hat is sensitive to kappa at the margin -- which is a")
        print("   property of the selector worth reporting on its own, and an")
        print("   argument for calibrating kappa at a larger R than 400.")
    else:
        print("   No pick moves.  The phi mis-mapping is immaterial to every")
        print("   reported result, and the honest treatment is one sentence in the")
        print("   calibration appendix giving the copula relation and stating that")
        print("   the shipped phi were set to rho_S directly -- not a silent fix,")
        print("   and not a re-run of the tables either.")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "a3_kappa_phi_sensitivity.json", "w") as f:
        json.dump({"kappa": kaps, "picks": picks, "R_cal": R_CAL}, f, indent=2)
    print(f"\nwrote {OUT / 'a3_kappa_phi_sensitivity.json'}")


if __name__ == "__main__":
    main()
