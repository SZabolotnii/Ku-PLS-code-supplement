#!/usr/bin/env python3
"""The check Panel C has to survive: a null that respects serial dependence.

Theorem 6' assumes i.i.d. sampling.  Daily money-market and volatility series are
not i.i.d. -- they are serially dependent and conditionally heteroskedastic -- so
the theta = 1 rejection in Panel C (p = 0.0013 against BCT's p = 0.597) could in
principle be the i.i.d. assumption failing rather than the calibration working.
The first version of this work had the same exposure and did not test it.

THE NULL USED HERE.  A stationary (Politis-Romano) bootstrap that resamples blocks
of X and blocks of Y with INDEPENDENT start indices.  Each series keeps its own
serial dependence and its own marginal tail; only the cross-series relation is
destroyed.  That is exactly H0: no relation between the curve and the response,
imposed without assuming independence within either series.

Reported per theta: the bootstrap p-value, and -- the point of the exercise -- the
bootstrap 95th percentile of the statistic against the asymptotic critical value
the plug-in weighted-chi^2 supplies.  If the asymptotic critical value sits far
ABOVE the bootstrap one, that is the conservatism of Theorem 6' section 3 visible
on real data rather than in a simulation.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from twomoments import operator, cg  # noqa: E402
from empirical2_cp import load, _w   # noqa: E402

OUT = HERE.parent / "results"
SEED = 20260828
B = 999


def stat_only(X, Y, theta, m_max=60):
    n = len(Y)
    A, r = operator(X, Y, theta)
    a_n = 1.0 / (np.sqrt(n) * np.log(n))
    for m in range(1, m_max + 1):
        bh = cg(A, r, m)
        if np.linalg.norm(r - A @ bh) <= a_n:
            break
    return n * float(np.sum((A @ bh) ** 2))


def asym_crit(X, Y, theta, rng, ndraw=20000):
    n = len(Y)
    Z = X * (_w(X, theta) * Y)[:, None]
    ev = np.clip(np.linalg.eigvalsh(Z.T @ Z / n), 0.0, None)
    ev = ev[ev > 1e-14]
    d = (rng.chisquare(1.0, (ndraw, ev.size)) * ev).sum(1)
    return float(np.quantile(d, .95))


def blocks(n, mb, rng):
    idx, i, p = np.empty(n, dtype=np.int64), 0, 1.0 / mb
    while i < n:
        st = rng.integers(0, n)
        L = min(rng.geometric(p), n - i)
        idx[i:i + L] = (st + np.arange(L)) % n
        i += L
    return idx


def main():
    rng = np.random.default_rng(SEED)
    dX, Y, Yv, dates = load()
    n = len(Yv)
    mb = max(2, int(round(n ** (1 / 3))))
    print(f"Panel C -- contemporaneous, Y = VIX log change, n = {n}, "
          f"B = {B}, mean block = {mb}")
    print("H0 imposed by resampling X-blocks and Y-blocks with INDEPENDENT starts:")
    print("each series keeps its serial dependence, the cross-relation is destroyed.\n")
    print(f"   {'theta':>7} {'statistic':>12} {'boot p':>9} {'boot 95%':>12} "
          f"{'asym 95%':>12} {'asym/boot':>11}")
    rows = {}
    for th in (0.0, 0.5, 1.0):
        obs = stat_only(dX, Yv, th)
        null = np.empty(B)
        for b in range(B):
            ix, iy = blocks(n, mb, rng), blocks(n, mb, rng)
            null[b] = stat_only(dX[ix], Yv[iy], th)
        p = float((null >= obs).mean())
        bc = float(np.quantile(null, .95))
        ac = asym_crit(dX, Yv, th, rng)
        rows[str(th)] = dict(stat=obs, boot_p=p, boot_c95=bc, asym_c95=ac,
                             ratio=ac / bc)
        tag = " (BCT)" if th == 0 else "      "
        print(f"   {th:5.1f}{tag} {obs:12.4g} {p:9.4f} {bc:12.4g} {ac:12.4g} "
              f"{ac / bc:10.2f}x")
    print()
    print("   READ.  'boot p' is the honest p-value under serial dependence.")
    print("   'asym/boot' is how far the plug-in weighted-chi^2 critical value sits")
    print("   above the one a dependence-respecting null actually needs: a number")
    print("   well above 1 is the conservatism, measured on data.")
    OUT.mkdir(exist_ok=True)
    with open(OUT / "empirical2_blockboot.json", "w") as f:
        json.dump({"n": n, "B": B, "mean_block": mb, "rows": rows}, f, indent=2)
    print(f"\nwrote {OUT / 'empirical2_blockboot.json'}")


if __name__ == "__main__":
    main()
