#!/usr/bin/env python3
"""Persist a derived quantity the manuscript quotes and no artifact contained.

The CoE audit flagged the exponents 2.25 / 2.01 / 1.89 -- the paper's evidence
that cond(C*C) ~ cond(K)^2 -- as unresolved.  They are correct, and they are
DERIVED: a0_3_bounded_feature_family stores cond(K) and the ratio
cond(C*C)/cond(K), and the exponent was computed in the reading of that table and
never written down.  Same class as the persistence targets and the power-ratio
column: a number that exists only in prose has not been checked, however many
times it has been read.

This reads the stored inputs and writes the derived quantity as its own artifact,
so the chain is cond(K), ratio -> exponent -> manuscript, with a file at each step.
It computes nothing new; if it disagrees with the prose, the prose is wrong.
"""
from __future__ import annotations

import json
import math
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "results"
SRC = OUT / "a0_3_bounded_feature_family.json"


def main():
    rows = json.load(open(SRC))["conditioning"]
    print("=" * 78)
    print("cond(C*C) as a power of cond(K), derived from a0_3_bounded_feature_family")
    print("   exponent = log(cond(C*C)) / log(cond(K)),")
    print("   cond(C*C) = ratio_cf * cond(K)   [ratio_cf is what the source stores]")
    print("=" * 78)
    print(f"   {'decay':>8}{'cond(K)':>14}{'ratio_cf':>12}{'cond(C*C)':>16}{'exponent':>11}")
    out = []
    for r in rows:
        cK = r["cond_K"]
        cCF = r["ratio_cf"] * cK
        e = math.log(cCF) / math.log(cK)
        out.append({"decay": r["decay"], "cond_K": cK, "cond_CFC": cCF,
                    "exponent": e,
                    "ratio_theta_1": r["ratio_theta_1.0"],
                    "ratio_theta_2": r["ratio_theta_2.0"]})
        print(f"   j^-{r['decay']:<5.1f}{cK:14.4g}{r['ratio_cf']:12.4g}"
              f"{cCF:16.6g}{e:11.4f}")
    print()
    print("   An exponent near 2 is the claim: the bounded-feature route composes")
    print("   the covariance twice, so it squares the ill-posedness.  Reported to")
    print("   three significant figures in the manuscript.")
    with open(OUT / "a3_cond_exponents.json", "w") as f:
        json.dump({"source": SRC.name, "rows": out}, f, indent=2)
    print(f"\nwrote {OUT / 'a3_cond_exponents.json'}")


if __name__ == "__main__":
    main()
