#!/usr/bin/env python3
"""The artifact for a number that did not have one.

A3-OUTLINE.md's number ledger turned up a claim in EMPIRICAL-FINDINGS.md section 10
with no artifact behind it: "the lag-1 Spearman autocorrelation of ||X_t|| is
+0.303 for the Treasury curve and +0.591 for the commercial-paper curve, matched by
phi = 0.30 and phi = 0.60".  `run_dependence.py` HARDCODES phi = 0.3 and 0.6 and
merely asserts the match in a print statement; the two correlations were computed
somewhere that no longer exists.

The number is load-bearing, which is why it cannot stay in prose: phi sets kappa
(0.0966 / 0.0777), kappa sets theta-hat, and theta-hat produces the headline
p = 0.0017 on CP x VIX.  A calibration constant chosen by an unrecorded computation
is exactly the failure the ledger exists to catch.

This measures both correlations from the cached FRED panels, and -- the part the
original assertion never checked -- inverts the copula-AR(1) design to find which
phi actually reproduces each measured value, rather than assuming phi = rho.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
from dependent_design import spearman_lag1, pareto_ar1  # noqa: E402

ROOT = HERE.parent
OUT, CACHE = ROOT / "results", ROOT / "data"
SEED = 20260912


TREASURY = ROOT / "data" / "fred_cache.npz"


def panels():
    """The panels the APPLICATIONS actually use, not the raw caches.

    Two traps, both hit on the first attempt at this script:
      1. the .npz files already store dX -- differencing them again measures the
         autocorrelation of a second difference, which is a different number;
      2. the CP application runs on cp_nasdaq.npz (n = 4396, the dates common to
         the curve AND both responses), not cp-term.npz (n = 4405), and the
         Treasury application runs on data/fred_cache.npz (n = 2494), not
         data/treasury.npz (n = 6264).
    Getting either wrong produces a plausible number for the wrong object.
    """
    cp = np.load(CACHE / "cp_nasdaq.npz", allow_pickle=True)["dX"].astype(float)
    tr = np.load(TREASURY, allow_pickle=True)["dX"].astype(float)
    return {"Treasury curve": ("data/fred_cache.npz", tr),
            "commercial-paper curve": ("data/cp_nasdaq.npz", cp)}


def norms_from(dX):
    """||dX_t|| for an ALREADY-DIFFERENCED panel."""
    return np.linalg.norm(dX, axis=1)


def main():
    rng = np.random.default_rng(SEED)
    out = {}

    print("=" * 88)
    print("MEASURED lag-1 Spearman autocorrelation of ||dX_t||")
    print("   Spearman, not Pearson: it is invariant to the marginal, so it is")
    print("   comparable across two series with different tail indices -- which a")
    print("   Pearson autocorrelation at these tails would not be.")
    print("=" * 88)
    rows = {}
    for label, (fname, dX) in panels().items():
        nrm = norms_from(dX)
        nrm = nrm[np.isfinite(nrm)]
        rho = spearman_lag1(nrm)
        rows[label] = {"file": fname, "shape": list(dX.shape),
                       "n_increments": int(len(nrm)), "spearman_lag1": float(rho)}
        print(f"   {label:<24} {fname:<40} panel {str(dX.shape):>12}"
              f"   n = {len(nrm):>5}   rho_S = {rho:+.4f}")
    out["measured"] = rows

    print()
    print("=" * 88)
    print("INVERTING THE DESIGN -- which phi reproduces each measured rho_S?")
    print("   The copula-AR(1) design drives a Gaussian AR(1) with parameter phi")
    print("   through Phi and then the inverse Pareto CDF.  Both maps are monotone,")
    print("   so Spearman is preserved EXACTLY and phi maps to rho_S through the")
    print("   Gaussian-copula relation rho_S = (6/pi) arcsin(phi/2), not phi = rho_S.")
    print("   That distinction was never checked when phi = 0.3 / 0.6 were fixed.")
    print("=" * 88)
    grid = np.round(np.arange(0.0, 0.96, 0.05), 2)
    n_sim = 20000
    tab = []
    print(f"   {'phi':>6} {'rho_S (simulated)':>20} {'(6/pi)arcsin(phi/2)':>22}")
    for phi in grid:
        r = np.mean([spearman_lag1(pareto_ar1(n_sim, 2.0, float(phi), rng))
                     for _ in range(3)])
        theory = (6.0 / np.pi) * np.arcsin(phi / 2.0)
        tab.append({"phi": float(phi), "rho_sim": float(r), "rho_theory": float(theory)})
        print(f"   {phi:6.2f} {r:20.4f} {theory:22.4f}")
    out["inversion"] = tab

    print()
    sim = np.array([t["rho_sim"] for t in tab])
    print(f"   {'series':<24} {'rho_S':>9} {'phi used':>10} {'phi implied':>13}"
          f" {'kappa impact':>14}")
    fixes = {}
    for label, used in (("Treasury curve", 0.30), ("commercial-paper curve", 0.60)):
        if label not in rows:
            continue
        r = rows[label]["spearman_lag1"]
        phi_hat = float(np.interp(r, sim, grid))
        fixes[label] = {"rho_S": r, "phi_used": used, "phi_implied": phi_hat}
        flag = "  <-- MISMATCH" if abs(phi_hat - used) > 0.05 else ""
        print(f"   {label:<24} {r:+9.4f} {used:10.2f} {phi_hat:13.3f}"
              f" {'recalibrate' if flag else 'ok':>14}{flag}")
    out["fixes"] = fixes

    print()
    print("   READING.  If phi_implied differs materially from the phi that was")
    print("   used, kappa was calibrated at the wrong persistence and every")
    print("   theta-hat downstream inherits it.  kappa is DECREASING in phi (more")
    print("   persistence -> a larger max-to-sum ratio at the same n), so a phi that")
    print("   is too small makes kappa too LARGE, which makes the selector too")
    print("   PERMISSIVE -- it would accept a smaller theta than it should.")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "a3_persistence_artifact.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT / 'a3_persistence_artifact.json'}")


if __name__ == "__main__":
    main()
