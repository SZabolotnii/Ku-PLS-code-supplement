"""How often does the test's CG loop hit its 40-step cap, and how large is sqrt(n)*||R||?

Same design and stopping test as gates/a1_7_power.run_stat (n=1000, a_n=n^-1/2/log n),
under H0, for the nine tab:power cells. Read-only on the track.
"""
import sys, json, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from twomoments import Design, operator, cg

N, M_MAX, R = 1000, 40, int(sys.argv[1]) if len(sys.argv) > 1 else 300
a_n = 1.0 / (np.sqrt(N) * np.log(N))
rng = np.random.default_rng(20261112)
design = Design(J=20, decay=1.0)
out = {}
for th in (0.0, 0.5, 1.0):
    for idx in (1.2, 1.5, 1.8):
        capped, ms, rn = 0, [], []
        for _ in range(R):
            X, Y = design.draw(N, idx, rng)
            A, r = operator(X, Y, th)
            hit = False
            for m in range(1, M_MAX + 1):
                bh = cg(A, r, m)
                res = np.linalg.norm(r - A @ bh)
                if res <= a_n:
                    hit = True
                    break
            capped += (not hit)
            ms.append(m)
            rn.append(np.sqrt(N) * res)
        out[f"th{th}_idx{idx}"] = dict(cap_rate=capped / R, m_median=float(np.median(ms)),
                                       m_max=int(max(ms)),
                                       sqrt_n_R_q99=float(np.quantile(rn, 0.99)),
                                       sqrt_n_R_max=float(max(rn)))
        print(f"theta={th} index={idx}: cap {capped}/{R}, median m {np.median(ms)}, "
              f"max m {max(ms)}, sqrt(n)|R| q99 {np.quantile(rn, .99):.2e} max {max(rn):.2e}",
              flush=True)
json.dump({"R": R, "n": N, "a_n": a_n, "m_max": M_MAX, "cells": out},
          open(sys.argv[2] if len(sys.argv) > 2 else
               pathlib.Path(__file__).resolve().parents[1] / "results" / "cap_probe.json",
               "w"), indent=2)
