#!/usr/bin/env python3
"""Does the SELECTED theta deliver the finding, without hindsight?

Panel C was found at theta = 1 (p = 0.0013) after looking at theta = 0, 0.5 and 1.
The selector, run blind on the same data, picks theta-hat = 0.75 for CP x VIX and
0.50 for the Treasury curve.  A selection rule is only worth shipping if the theta
it picks reaches the same conclusion as the theta a person picked after seeing all
the answers.  This runs the test at the SELECTED value, and at the selected value
for the Treasury application too, where the answer should be 'nothing here'.

Both are checked against the block-bootstrap null of empirical2_blockboot.py, so
the verdict does not rest on the i.i.d. assumption the selector was calibrated under.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from empirical2_cp import load as load_cp, _w   # noqa: E402
from empirical2_blockboot import stat_only, asym_crit, blocks  # noqa: E402

OUT = HERE.parent / "results"
TREASURY = HERE.parent / "data" / "fred_cache.npz"
B = 999


def report(name, X, Y, thetas, rng):
    n = len(Y)
    mb = max(2, int(round(n ** (1 / 3))))
    print(f"\n   {name}   (n = {n}, mean block = {mb})")
    print(f"   {'theta':>18} {'statistic':>12} {'asym p':>9} {'boot p':>9} "
          f"{'boot 95%':>12} {'asym 95%':>12} {'asym/boot':>11}")
    out = {}
    for th, tag in thetas:
        obs = stat_only(X, Y, th)
        null = np.empty(B)
        for b in range(B):
            null[b] = stat_only(X[blocks(n, mb, rng)], Y[blocks(n, mb, rng)], th)
        bp = float((null >= obs).mean())
        bc = float(np.quantile(null, .95))
        ac = asym_crit(X, Y, th, rng)
        # asymptotic p from the plug-in weighted chi^2
        Z = X * (_w(X, th) * Y)[:, None]
        ev = np.clip(np.linalg.eigvalsh(Z.T @ Z / n), 0.0, None)
        ev = ev[ev > 1e-14]
        d = (rng.chisquare(1.0, (20000, ev.size)) * ev).sum(1)
        ap = float((d > obs).mean())
        out[tag] = dict(theta=th, stat=obs, asym_p=ap, boot_p=bp,
                        boot_c95=bc, asym_c95=ac, ratio=ac / bc)
        print(f"   {tag:>18} {obs:12.4g} {ap:9.4f} {bp:9.4f} {bc:12.4g} "
              f"{ac:12.4g} {ac / bc:10.2f}x")
    return out


def main():
    rng = np.random.default_rng(20260831)
    dXc, Yc, Yv, _ = load_cp()
    d = np.load(TREASURY, allow_pickle=True)
    dXt, Yt = d["dX"], d["Y"]

    print("Test at the SELECTED theta, against a dependence-respecting null.")
    print(f"B = {B}.  'asym p' uses the plug-in weighted-chi^2; 'boot p' the block")
    print("bootstrap.  A rule that works has the two agree at theta-hat.")

    res = {}
    res["cp_vix"] = report("CP term structure x VIX  (theta-hat = 0.75)", dXc, Yv,
                           [(0.0, "0.00  BCT"), (0.5, "0.50"),
                            (0.75, "0.75  SELECTED"), (1.0, "1.00")], rng)
    res["treasury"] = report("Treasury curve x S&P 500  (theta-hat = 0.50)", dXt, Yt,
                             [(0.0, "0.00  BCT"), (0.5, "0.50  SELECTED")], rng)

    OUT.mkdir(exist_ok=True)
    with open(OUT / "check_selected_theta.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\nwrote {OUT / 'check_selected_theta.json'}")


if __name__ == "__main__":
    main()
