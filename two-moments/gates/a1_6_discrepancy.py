#!/usr/bin/env python3
"""Prop 6's gap, and a concrete defect it exposed in the stopping rule.

THE GAP.  Prop 6 states consistency by transporting conjugate-gradient
regularisation theory, whose classical statements assume a KNOWN operator with
noisy data.  Here A-hat_theta is itself estimated and CG is nonlinear in both
arguments.  Looking at what the transport actually needs turned up something more
concrete than a missing citation.

THE DEFECT.  `twomoments.discrepancy_stop` stops when

    ||r-hat - A-hat beta-hat_m||  <=  tau * sqrt( tr(A-hat_theta) / n ) ,

i.e. at a threshold set by tr A_theta = E||X||^{2-theta} -- the scale of the
OPERATOR.  The discrepancy principle is supposed to stop at the noise level of the
DATA, and that level is

    || xi-hat ||  ~  sqrt( tr V_theta / n ),   tr V_theta = E[eps^2 ||X||^{2-2theta}],

which is a different object and, at heavy tails, a very different number: at
theta = 0 and tail index below 2 the first is finite while the second is infinite.
Stopping at the wrong scale means stopping at the wrong iteration.

THE FIX, and it unifies with the rest of the paper.  tr V_theta is exactly the
object whose eigenvalues are the weights in Theorem 6', and it is estimable from
the current residuals.  The theta-adapted rule is

    m-hat = min{ m : ||r-hat - A-hat beta-hat_m|| <= tau * sqrt( tr V-hat_theta(m) / n ) },
    tr V-hat_theta(m) = n^{-1} sum_k (Y_k - <beta-hat_m, X_k>)^2 ||X_k||^{2-2theta},

self-consistent, online, and the SAME quantity the inference half already needs.

CHECKED: whether the fix actually improves estimation, across theta and tail index.
A rule that is theoretically better and empirically identical is not worth a
paragraph; one that is better on both is Prop 6's practical content.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
from twomoments import (Design, operator,  # noqa: E402
                        discrepancy_stop_operator_scale, discrepancy_stop)

OUT = HERE.parent / "results"
SEED = 20260908
THETAS = [0.0, 0.5, 1.0, 2.0]
INDICES = [1.5, 2.5, 4.0]


# The fix now LIVES in twomoments.discrepancy_stop; the local copy that was
# here has been deleted so this gate and the re-run tables share one code path.
# The superseded rule survives as discrepancy_stop_operator_scale.


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = np.random.default_rng(SEED)
    design = Design(J=20, decay=1.0)
    beta, nb = design.beta, np.linalg.norm(design.beta)

    print("=" * 92)
    print("median relative slope error, and median stopping index, by rule")
    print(f"   OLD = threshold sqrt(tr Ahat / n)   NEW = threshold sqrt(tr Vhat / n)")
    print(f"   n = 1000, R = {R}")
    print("=" * 92)
    print(f"   {'index':>7} {'theta':>7} |{'OLD err':>10}{'OLD m':>7}"
          f" |{'NEW err':>10}{'NEW m':>7} |{'gain':>8}")
    rows = []
    for index in INDICES:
        for th in THETAS:
            o, ostop, nw, nstop = [], [], [], []
            for _ in range(R):
                X, Y = design.draw(1000, index, rng)
                a1, m1 = discrepancy_stop_operator_scale(X, Y, th)
                a2, m2 = discrepancy_stop(X, Y, th)
                o.append(np.linalg.norm(a1 - beta) / nb); ostop.append(m1)
                nw.append(np.linalg.norm(a2 - beta) / nb); nstop.append(m2)
            eo, en = float(np.median(o)), float(np.median(nw))
            rows.append({"index": index, "theta": th, "old": eo, "new": en,
                         "old_m": float(np.median(ostop)),
                         "new_m": float(np.median(nstop)), "gain": eo / en})
            print(f"   {index:7.1f} {th:7.1f} |{eo:10.4f}{np.median(ostop):7.1f}"
                  f" |{en:10.4f}{np.median(nstop):7.1f} |{eo/en:7.2f}x")
    print()
    print("   gain > 1 means the noise-level threshold estimates better.")
    OUT.mkdir(exist_ok=True)
    with open(OUT / "a1_6_discrepancy.json", "w") as f:
        json.dump({"R": R, "rows": rows}, f, indent=2, default=float)
    print(f"wrote {OUT / 'a1_6_discrepancy.json'}")


if __name__ == "__main__":
    main()
