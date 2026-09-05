#!/usr/bin/env python3
"""Find a functional predictor whose norm has tail index in (1,2).

WHY THAT INTERVAL AND NOT ANOTHER.  Theorem 6' fails for the baseline exactly when
tr V_0 = E[eps^2 ||X||^2] = infinity while tr V_theta stays finite.  Under
conditional homoskedasticity that is E||X||^2 = inf, i.e. a tail index of ||X||
BELOW 2 -- and above 1, so that A_theta = E[||X||^{1-theta} U (x) U] still exists
at theta = 1.  Note it must be the PREDICTOR that is heavy: a heavy eps makes
tr V_theta infinite for every theta and helps nobody.

The existing FRED application (11 constant-maturity Treasury yields) has
||X_t|| at index [2.64, 3.94] -- above 2, in the thin window where both tests work.
It stays in the paper, stated as such.  This looks for a second application that
is in the regime the theorem is about.

CANDIDATES, all daily and all served by FRED over plain HTTPS with no API key, so
the replication package stays self-contained:

  credit-oas   ICE BofA option-adjusted spreads across the RATING ladder
               AAA -> CCC.  A genuine discretised function of credit quality, and
               spreads jump in crises rather than diffusing.
  tips-curve   TIPS real yields across maturity -- thinner, included as a control.
  hy-curve     High-yield sub-indices only, the heavy end of the ladder.
  treasury     the incumbent, recomputed here so the comparison is like for like.

Reported per candidate: the dependence-robust tail-index interval of ||dX_t||
(stationary bootstrap, the A0.4 machinery), and whether it lands in (1,2).
"""
from __future__ import annotations

import io
import json
import pathlib
import sys
import urllib.request

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / "data"
OUT = ROOT / "results"
SEED = 20260826
B = 2000

FAMILIES = {
    "treasury": (["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3",
                  "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"], 100.0,
                 "yield curve across maturity, bp (the incumbent)"),
    "credit-oas": (["BAMLC0A1CAAA", "BAMLC0A2CAA", "BAMLC0A3CA", "BAMLC0A4CBBB",
                    "BAMLH0A1HYBB", "BAMLH0A2HYB", "BAMLH0A3HYC"], 100.0,
                   "option-adjusted spread across the rating ladder AAA->CCC, bp"),
    "hy-curve": (["BAMLH0A1HYBB", "BAMLH0A2HYB", "BAMLH0A3HYC"], 100.0,
                 "high-yield sub-indices only, bp"),
    "tips-curve": (["DFII5", "DFII7", "DFII10", "DFII20", "DFII30"], 100.0,
                   "TIPS real yields across maturity, bp (control)"),
    "cp-financial": (["DCPF1M", "DCPF2M", "DCPF3M"], 100.0,
                     "financial commercial-paper rates across maturity, bp -- "
                     "the 2008 funding freeze lives here"),
    "cp-nonfin": (["DCPN30", "DCPN60", "DCPN90"], 100.0,
                  "nonfinancial commercial-paper rates across maturity, bp"),
    "ig-ladder": (["BAMLC0A1CAAAEY", "BAMLC0A2CAAEY", "BAMLC0A3CAEY",
                   "BAMLC0A4CBBBEY"], 100.0,
                  "investment-grade effective yields across the rating ladder, bp"),
    "cp-term": (["RIFSPPFAAD01NB", "RIFSPPFAAD07NB", "RIFSPPFAAD15NB",
                 "RIFSPPFAAD30NB", "RIFSPPFAAD60NB", "RIFSPPFAAD90NB"], 100.0,
                "AA financial commercial-paper rate across term 1/7/15/30/60/90 "
                "days -- a six-point money-market curve spanning 2008"),
}


def fred(series):
    """Full history.  Two endpoints because they disagree about what 'full' means.

    The graph CSV endpoint honours cosd for some series and silently caps others
    at a recent window -- the ICE BofA family returns 786 observations of
    2023-2026 whatever cosd says.  The /data/<ID>.txt endpoint returns the whole
    series.  Try it first and fall back.
    """
    try:
        raw = urllib.request.urlopen(
            f"https://fred.stlouisfed.org/data/{series}.txt", timeout=90).read().decode()
        rows, started = {}, False
        for line in raw.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0][:4].isdigit() and parts[0].count("-") == 2:
                started = True
                rows[parts[0]] = np.nan if parts[1] == "." else float(parts[1])
        if started and len(rows) > 0:
            return rows
    except Exception:
        pass
    # cosd is NOT optional.  Without it the graph endpoint returns only its
    # default recent window -- a first run of this probe pulled 786 observations
    # of a calm 2023-2026 period for the credit series and concluded they were
    # thin-tailed.  Credit spreads are heavy BECAUSE of 2008; excluding it
    # measured the wrong thing entirely.
    url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
           f"&cosd=1900-01-01&coed=2030-01-01")
    raw = urllib.request.urlopen(url, timeout=90).read().decode()
    out = {}
    for line in io.StringIO(raw).readlines()[1:]:
        parts = line.strip().split(",")[:2]
        if len(parts) < 2:
            continue
        d, v = parts
        out[d] = np.nan if v in (".", "") else float(v)
    return out


def load(name):
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{name}.npz"
    if f.exists():
        d = np.load(f, allow_pickle=True)
        return d["dX"], d["dates"]
    ids, mult, _ = FAMILIES[name]
    cols = {s: fred(s) for s in ids}
    common = set.intersection(*(set(c) for c in cols.values()))
    dates = sorted(common)
    M = np.array([[cols[s][d] for s in ids] for d in dates])
    ok = np.isfinite(M).all(1)
    M, dates = M[ok], [d for d, k in zip(dates, ok) if k]
    dX = np.diff(M, axis=0) * mult
    dates = np.array(dates[1:])
    np.savez(f, dX=dX, dates=dates)
    return dX, dates


def hill(x, k):
    v = np.sort(np.abs(x))[::-1]
    if k >= len(v) or v[k] <= 0:
        return np.nan
    return 1.0 / np.mean(np.log(v[:k] / v[k]))


def stationary_boot(x, k, B, rng, mb):
    n = len(x)
    p = 1.0 / mb
    out = np.empty(B)
    for b in range(B):
        idx = np.empty(n, dtype=np.int64)
        i = 0
        while i < n:
            st = rng.integers(0, n)
            L = min(rng.geometric(p), n - i)
            idx[i:i + L] = (st + np.arange(L)) % n
            i += L
        out[b] = hill(x[idx], k)
    return out


def analyse(name, nrm, rng):
    n = len(nrm)
    ks = np.unique(np.linspace(n / 40, n / 10, 40).astype(int))
    hp = np.array([hill(nrm, int(k)) for k in ks])
    k_ref = int(np.median(ks))
    point = hill(nrm, k_ref)
    mb = max(2, int(round(n ** (1 / 3))))
    d = stationary_boot(nrm, k_ref, B, rng, mb)
    d = d[np.isfinite(d)]
    z = 1.959963984540054
    cands = [np.quantile(d, .025), point / (1 + z / np.sqrt(k_ref)), np.nanmin(hp)]
    hi = [np.quantile(d, .975), point / (1 - z / np.sqrt(k_ref)), np.nanmax(hp)]
    lo, hi = float(min(cands)), float(max(hi))
    return dict(name=name, n=n, k_ref=k_ref, point=float(point),
                lo=lo, hi=hi, in_1_2=bool(hi < 2.0 and lo > 1.0),
                below_2=bool(hi < 2.0), straddles_2=bool(lo < 2.0 < hi))


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    print("tail index of ||dX_t||, dependence-robust (stationary bootstrap, "
          f"B = {B})\n")
    print(f"{'family':>12} {'n':>6} {'point':>7} {'reported 95% interval':>24} "
          f"{'verdict':>28}")
    print("-" * 84)
    for name in FAMILIES:
        try:
            dX, dates = load(name)
        except Exception as e:                     # network or series retired
            print(f"{name:>12}   FAILED: {type(e).__name__}: {e}")
            continue
        nrm = np.linalg.norm(dX, axis=1)
        r = analyse(name, nrm, rng)
        r["span"] = f"{dates[0]}..{dates[-1]}"
        r["desc"] = FAMILIES[name][2]
        rows.append(r)
        if r["in_1_2"]:
            verdict = "*** IN (1,2) -- the regime ***"
        elif r["below_2"]:
            verdict = "below 2 but also below 1"
        elif r["straddles_2"]:
            verdict = "straddles 2 -- inconclusive"
        else:
            verdict = "above 2 -- thin window"
        print(f"{name:>12} {r['n']:6d} {r['point']:7.3f} "
              f"[{r['lo']:8.3f},{r['hi']:8.3f} ]{verdict:>28}")
    print()
    for r in rows:
        print(f"  {r['name']:>12}  {r['span']}  --  {r['desc']}")
    OUT.mkdir(exist_ok=True)
    with open(OUT / "probe_heavy_curves.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {OUT / 'probe_heavy_curves.json'}")


if __name__ == "__main__":
    main()
