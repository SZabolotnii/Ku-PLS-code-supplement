#!/usr/bin/env python3
"""The load-bearing table of the paper, at an R that can carry it.

At R = 500 the headline cells were BCT size 0.018 (3.3 s.e. below nominal) and raw
power 0.280 against theta=1's 0.936 at tail index 1.2.  This re-runs the three
tail indices below 2 at R = 1500 (s.e. 0.0056), with the SAME calibrated local
departure c = 0.716, and adds n = 4000 at index 1.5 to check the conservatism does
not simply wash out with sample size -- the question the earlier work in this repo
got wrong in the other direction.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "gates"))
from a1_7_power import cell, LEVEL  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "code"))
from twomoments import Design  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
C = 0.716


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    rng = np.random.default_rng(20260825)
    design = Design(J=20, decay=1.0)
    v = design.beta / np.linalg.norm(design.beta)
    se = np.sqrt(LEVEL * (1 - LEVEL) / R)
    out, ratios = {}, {}
    for n in (1000, 4000):
        import a1_7_power as P
        P.N = n
        a_n = 1.0 / (np.sqrt(n) * np.log(n))
        idx = [1.2, 1.5, 1.8] if n == 1000 else [1.5]
        print("=" * 84)
        print(f"n = {n}, R = {R}, c = {C}, nominal {LEVEL:.0%}, s.e. = {se:.4f}, "
              f"tol [{LEVEL-2*se:.3f}, {LEVEL+2*se:.3f}]")
        print("=" * 84)
        print(f"   {'theta':>9} {'index':>7} {'size':>10} {'power raw':>12} "
              f"{'power s.c.':>12} {'raw vs BCT':>12}")
        base = {}
        for th in (0.0, 0.5, 1.0):
            for index in idx:
                d = cell(design, index, th, C, v, R, rng, a_n)
                if th == 0.0:
                    base[index] = d["power_raw"]
                ratio = d["power_raw"] / base[index] if base.get(index) else float("nan")
                ratios[f"n{n}_th{th}_idx{index}"] = ratio
                flag = "*" if abs(d["size"] - LEVEL) > 2 * se else " "
                tag = "(BCT)" if th == 0 else "     "
                print(f"   {th:4.1f}{tag} {index:7.1f} {d['size']:9.3f}{flag} "
                      f"{d['power_raw']:12.3f} {d['power_sc']:12.3f} {ratio:11.2f}x",
                      flush=True)
                out[f"n{n}_th{th}_idx{index}"] = d
        print()
    print("   * = size outside +-2 s.e. of nominal")
    OUT.mkdir(exist_ok=True)
    with open(OUT / "a1_7_power_confirm.json", "w") as f:
        json.dump({"R": R, "c": C, "rows": out, "ratio_vs_bct": ratios},
                  f, indent=2, default=float)
    print(f"wrote {OUT / 'a1_7_power_confirm.json'}")


if __name__ == "__main__":
    main()
