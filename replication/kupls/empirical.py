"""Empirical application: yield-curve shocks and equity returns.

Data are pulled from FRED, which serves them over plain HTTPS with no API key,
so this script is self-contained and the application is replicable by anyone.

  X_t : the curve of daily yield CHANGES across 11 maturities (1M -> 30Y), in
        basis points -- a discretised function of maturity.
  Y_t : S&P 500 daily log return, in percent.

  Panel A (contemporaneous)  Y_t     on X_t
  Panel B (predictive)       Y_{t+1} on X_t

The hypothesis tested is H0: beta = 0 in both panels. Read the note in
`core.py` on what the two nulls assert before interpreting Panel B: T_n asks
whether the linear slope is zero, S_n asks whether the shock curve and the
return are independent, and financial data separates those two questions.
"""
from __future__ import annotations

import io
import json
import pathlib
import urllib.request

import numpy as np

from .core import Grid, stat_T, stat_S, hill

MATURITIES = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3",
              "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"]
LABELS = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
CACHE = pathlib.Path("data/fred_cache.npz")


# --------------------------------------------------------------------- data

def _fred(series: str) -> dict[str, float]:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    raw = urllib.request.urlopen(url, timeout=60).read().decode()
    out = {}
    for line in io.StringIO(raw).readlines()[1:]:
        date, value = line.strip().split(",")[:2]
        out[date] = np.nan if value in (".", "") else float(value)
    return out


def load(refresh: bool = False):
    """Yield-change curves, equity returns and dates. Cached after first pull."""
    if CACHE.exists() and not refresh:
        d = np.load(CACHE, allow_pickle=True)
        return d["dX"], d["Y"], d["dates"]

    CACHE.parent.mkdir(exist_ok=True)
    curves = {m: _fred(m) for m in MATURITIES}
    spx = _fred("SP500")

    common = set(spx)
    for m in MATURITIES:
        common &= set(curves[m])
    dates = sorted(common)

    yields = np.array([[curves[m][d] for m in MATURITIES] for d in dates])
    price = np.array([spx[d] for d in dates])
    ok = np.isfinite(yields).all(1) & np.isfinite(price)
    yields, price = yields[ok], price[ok]
    dates = [d for d, keep in zip(dates, ok) if keep]

    dX = np.diff(yields, axis=0) * 100.0        # basis points
    Y = np.diff(np.log(price)) * 100.0          # percent
    dates = np.array(dates[1:])
    np.savez(CACHE, dX=dX, Y=Y, dates=dates)
    return dX, Y, dates


# ------------------------------------------------------------------ testing

class CurveGrid(Grid):
    """The simulation's procedural rule, with the four probe directions spread
    across the maturity curve (1M, 1Y, 5Y, 20Y) instead of bunched at the
    short end where the maturities are nearly collinear."""

    DIRS = (0, 3, 6, 9)

    def build(self, xi, Y, b):
        def mad(v):
            return np.median(np.abs(v - np.median(v))) + 1e-12

        sY, J, pairs = mad(Y), xi.shape[1], []
        for j in self.DIRS:
            sj = mad(xi[:, j])
            for t in self.mags:
                for sgn in (1.0, -1.0):
                    u = np.zeros(J)
                    u[j] = sgn * t / sj
                    pairs.append((0.0, u))
                    for sm in self.s_mags:
                        pairs.append((sgn * sm / sY, u))
        for sm in self.s_mags:
            for sgn in (1.0, -1.0):
                pairs.append((sgn * sm / sY, np.zeros(J)))
        self.s = np.array([p[0] for p in pairs])
        self.U = np.array([p[1] for p in pairs])
        self.w = np.full(len(pairs), 1.0 / len(pairs))
        return self


def pvalue(stat, weights, rng, ndraw=200_000):
    w = weights[weights > 1e-12]
    if w.size == 0:
        return 1.0
    draws = (rng.chisquare(1.0, (ndraw, w.size)) * w).sum(1)
    return float(np.mean(draws >= stat))


def test_panel(X, Y, m, rng, label):
    b0 = np.zeros(X.shape[1])
    T, wT = stat_T(X, Y, b0, m)
    S, wS = stat_S(X, Y, b0, CurveGrid())
    row = dict(panel=label, n=int(len(Y)), m=int(m),
               T=float(T), p_T=pvalue(T, wT, rng),
               S=float(S), p_S=pvalue(S, wS, rng))
    print(f"  {label:22s} m={m}  n={row['n']:5d}  "
          f"T_n={T:9.2f} p={row['p_T']:.4f}   "
          f"S_n={S:7.4f} p={row['p_S']:.4f}")
    return row


def main():
    from scipy import stats as st

    dX, Y, dates = load()
    rng = np.random.default_rng(20260815)
    n = len(Y)
    print(f"sample: {n} trading days, {dates[0]} -> {dates[-1]}\n")

    print("Tail indices (Hill estimator):")
    tails = {}
    for name, v in (("||X_t||", np.linalg.norm(dX, axis=1)), ("|Y_t|", Y)):
        tails[name] = {f"{f:.0%}": round(hill(v, int(f * n)), 3)
                       for f in (0.02, 0.05, 0.10)}
        print(f"  {name:9s} " +
              "  ".join(f"top {a}: {b:.2f}" for a, b in tails[name].items()))
    print("  (BCT require E||X||^4 < inf, i.e. a tail index above 4.)\n")

    Xp, Yp = dX[:-1], Y[1:]
    print("Tests of H0: beta = 0")
    panels = [test_panel(dX, Y, 3, rng, "A contemporaneous"),
              test_panel(Xp, Yp, 3, rng, "B predictive t+1")]

    print("\nSensitivity to the number of CG components:")
    for m in (1, 2, 5):
        panels.append(test_panel(dX, Y, m, rng, "A contemporaneous"))
        panels.append(test_panel(Xp, Yp, m, rng, "B predictive t+1"))

    print("\nWhich channel carries the predictive dependence?")
    nrm = np.linalg.norm(Xp, axis=1)
    channels = {
        "pearson_normX_absY": st.pearsonr(nrm, np.abs(Yp)),
        "spearman_normX_absY": st.spearmanr(nrm, np.abs(Yp)),
        "spearman_normX_signedY": st.spearmanr(nrm, Yp),
        "pearson_meanshift_Y": st.pearsonr(Xp.mean(1), Yp),
    }
    for k, (r, p) in channels.items():
        print(f"  {k:24s} r={r:+.4f}  p={p:.3g}")
    channels = {k: [float(r), float(p)] for k, (r, p) in channels.items()}

    print("\nRobustness: excluding calendar 2020")
    keep = np.array([not str(d).startswith("2020") for d in dates[1:]])
    ex2020 = test_panel(Xp[keep], Yp[keep], 3, rng, "B ex-2020")

    pathlib.Path("results").mkdir(exist_ok=True)
    json.dump(dict(n=int(n), start=str(dates[0]), end=str(dates[-1]),
                   tails=tails, panels=panels, channels=channels,
                   ex2020=ex2020),
              open("results/empirical.json", "w"), indent=2)
    print("\nwrote results/empirical.json")


if __name__ == "__main__":
    main()
