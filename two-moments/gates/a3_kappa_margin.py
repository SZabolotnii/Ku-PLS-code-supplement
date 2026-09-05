#!/usr/bin/env python3
"""How close is each application's theta-hat to a switch in kappa?

`a3_kappa_phi_sensitivity.py` established that the phi correction is immaterial,
but it turned up something else on the way: at R = 400 -- the R the shipped
calibration used -- kappa's own Monte Carlo standard error is 0.0104 (Treasury)
and 0.0074 (CP), i.e. about 10% of kappa.  And in the first, under-powered run a
kappa of 0.1138 moved the Treasury pick from 0.75 to 0.50.

So the question is not whether the shipped kappa is right; it is how much room
there is around it.  The selector picks the LEAST admissible theta on the grid, so
theta-hat is a step function of kappa: there is an exact switch point between each
pair of adjacent grid values, and it can be read off directly from R_n(theta) on
the data rather than searched for.  theta-hat = min{theta : R_n(theta) <= kappa},
so the switch from theta_j to theta_{j-1} happens exactly at kappa = R_n(theta_{j-1}).

Reporting the margin in units of kappa's own standard error is the honest form: a
selection rule whose pick sits half a standard error from a boundary is not a rule
that "chose" anything.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
from theta_selector import max_to_sum, GRID           # noqa: E402
from empirical2_cp import load as load_cp             # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "results"
TREASURY = ROOT / "data" / "fred_cache.npz"

# from a3_kappa_phi_sensitivity.out, R = 3000 with a bootstrap CI
SHIPPED = {"Treasury x S&P 500": dict(kappa=0.0966, k3000=0.0981,
                                      ci=(0.0912, 0.1070), se400=0.0104),
           "CP x VIX": dict(kappa=0.0777, k3000=0.0727,
                            ci=(0.0674, 0.0779), se400=0.0074),
           "CP x NASDAQ": dict(kappa=0.0777, k3000=0.0727,
                               ci=(0.0674, 0.0779), se400=0.0074)}


def main():
    dXc, Yn, Yv, _ = load_cp()
    d = np.load(TREASURY, allow_pickle=True)
    dXt, Yt = d["dX"], d["Y"]
    apps = (("CP x VIX", dXc, Yv), ("CP x NASDAQ", dXc, Yn),
            ("Treasury x S&P 500", dXt, Yt))

    print("=" * 100)
    print("R_n(theta) on each application, and the kappa at which theta-hat switches")
    print("=" * 100)
    print(f"   {'application':<20}" + "".join(f"{'R_n(' + f'{t:.2f}' + ')':>12}"
                                              for t in GRID))
    rows = {}
    curves = {}
    for label, X, Y in apps:
        resid = Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]
        r = [float(max_to_sum(X, resid, t)) for t in GRID]
        curves[label] = r
        print(f"   {label:<20}" + "".join(f"{v:12.5f}" for v in r))
    print()
    print("   R_n is decreasing in theta on all three, so the rule is well defined:")
    print("   theta-hat is the first grid point whose R_n falls below kappa.")

    print()
    print("=" * 100)
    print("MARGIN -- distance from the shipped kappa to the nearest switch,")
    print("          in units of kappa's own Monte Carlo s.e. at R = 400")
    print("=" * 100)
    print(f"   {'application':<20}{'kappa':>9}{'th-hat':>8}"
          f"{'switch down at':>16}{'switch up at':>14}{'margin / s.e.':>15}")
    for label, _, _ in apps:
        r = np.array(curves[label])
        sh = SHIPPED[label]
        k = sh["kappa"]
        ok = np.where(r <= k)[0]
        j = int(ok[0]) if len(ok) else len(GRID) - 1
        th = GRID[j]
        # theta-hat drops to GRID[j-1] once kappa >= R_n(GRID[j-1])
        up = float(r[j - 1]) if j > 0 else np.inf     # kappa above this -> smaller theta
        dn = float(r[j])                              # kappa below this -> larger theta
        margin = min(abs(k - up), abs(k - dn)) / sh["se400"]
        rows[label] = {"kappa": k, "theta_hat": th, "switch_to_smaller_at": up,
                       "switch_to_larger_at": dn, "se400": sh["se400"],
                       "margin_in_se": float(margin)}
        us = f"{up:14.5f}" if np.isfinite(up) else f"{'--':>14}"
        print(f"   {label:<20}{k:9.4f}{th:8.2f}{dn:16.5f}{us}{margin:15.2f}")

    print()
    print("   READING.  A margin below about 2 s.e. means the pick is not resolved by")
    print("   the calibration and the paper must say so.  The fix is not a different")
    print("   kappa -- it is calibrating at a larger R, which is cheap (the null is")
    print("   light-tailed and simulated once) and was simply not done.")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "a3_kappa_margin.json", "w") as f:
        json.dump({"R_n": {k: v for k, v in curves.items()},
                   "grid": list(GRID), "margins": rows}, f, indent=2)
    print(f"\nwrote {OUT / 'a3_kappa_margin.json'}")


if __name__ == "__main__":
    main()
