#!/usr/bin/env python3
"""Second empirical application --- the money-market term structure, 1998-2026.

WHY A SECOND ONE AND WHY THIS ONE.  Gate A0.4 put the incumbent FRED predictor
(11 Treasury constant maturities) at tail index [2.64, 3.94] -- above 2, i.e. in
the thin window where both tests work and the baseline is slightly better.  That
application stays in the paper, stated as such.  Theorem 6' is about the regime
E||X||^2 = inf, tail index of ||X|| between 1 and 2, and this looks for it in an
object of the SAME KIND: a rate curve across term, not a different asset class.

  X_t : daily changes, in bp, of the AA financial commercial-paper rate across
        term 1 / 7 / 15 / 30 / 60 / 90 days -- a six-point money-market curve.

  TWO responses, BOTH DECLARED IN ADVANCE and both reported, so that this is not
  a search over responses until one rejects:
    Y = equity   NASDAQ Composite daily log return, in percent.
    Y = vol      daily log change of VIX, in percent -- the economically tighter
                 link, since money-market funding stress and equity volatility
                 move together, whereas the level of an equity return is mostly
                 noise with respect to the policy-driven CP curve.

  Panel A  contemporaneous   Y_t     on X_t
  Panel B  predictive        Y_{t+1} on X_t

Three things are reported and the third is the one that matters:

  1. THE TAIL INDEX, with the dependence-robust interval of gate A0.4, plus a
     SUBSAMPLE BREAKDOWN.  A referee's first objection to a heavy-tail claim on
     this series is that the heaviness is one episode -- the 2008 funding freeze.
     That objection is answered here rather than waited for.

  2. tr Vhat_theta, the diagnostic of Theorem 6'.  tr V_0 = E[eps^2 ||X||^2] is
     infinite when the index is below 2, so the baseline's plug-in weights have
     nothing to converge to; tr V_1 = E eps^2 does.  On simulated data this shows
     up as a trace that grows with the sample instead of settling.  Here it is
     checked directly, on expanding subsamples.

  3. THE TESTS.  T_n^theta for theta in {0, 0.5, 1} under the overfitting stopping
     rule the theorem requires, with the variance operator built from the
     NULL-IMPOSED residuals Y - <b,X> (not from a fitted betahat -- that mistake
     produced a withdrawn number in this repo already).
"""
from __future__ import annotations

import io
import json
import pathlib
import sys
import urllib.request

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from twomoments import operator, cg  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / "data"
OUT = ROOT / "results"
SEED = 20260827

CURVE = ["RIFSPPFAAD01NB", "RIFSPPFAAD07NB", "RIFSPPFAAD15NB",
         "RIFSPPFAAD30NB", "RIFSPPFAAD60NB", "RIFSPPFAAD90NB"]
TERMS = ["1d", "7d", "15d", "30d", "60d", "90d"]
EQUITY = "NASDAQCOM"
VOL = "VIXCLS"


def fred(series):
    try:
        raw = urllib.request.urlopen(
            f"https://fred.stlouisfed.org/data/{series}.txt", timeout=90).read().decode()
        rows = {}
        for line in raw.splitlines():
            p = line.split()
            if len(p) == 2 and p[0].count("-") == 2 and p[0][:4].isdigit():
                rows[p[0]] = np.nan if p[1] == "." else float(p[1])
        if rows:
            return rows
    except Exception:
        pass
    raw = urllib.request.urlopen(
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        f"&cosd=1900-01-01&coed=2030-01-01", timeout=90).read().decode()
    rows = {}
    for line in io.StringIO(raw).readlines()[1:]:
        p = line.strip().split(",")[:2]
        if len(p) == 2:
            rows[p[0]] = np.nan if p[1] in (".", "") else float(p[1])
    return rows


def load():
    CACHE.mkdir(exist_ok=True)
    f = CACHE / "cp_nasdaq.npz"
    if f.exists():
        d = np.load(f, allow_pickle=True)
        return d["dX"], d["Y"], d["Yv"], d["dates"]
    cols = {s: fred(s) for s in CURVE}
    px, vx = fred(EQUITY), fred(VOL)
    common = set(px) & set(vx)
    for s in CURVE:
        common &= set(cols[s])
    dates = sorted(common)
    M = np.array([[cols[s][d] for s in CURVE] for d in dates])
    P = np.array([px[d] for d in dates])
    V = np.array([vx[d] for d in dates])
    ok = np.isfinite(M).all(1) & np.isfinite(P) & (P > 0) & np.isfinite(V) & (V > 0)
    M, P, V = M[ok], P[ok], V[ok]
    dates = [d for d, k in zip(dates, ok) if k]
    dX = np.diff(M, axis=0) * 100.0                 # bp
    Y = np.diff(np.log(P)) * 100.0                  # percent
    Yv = np.diff(np.log(V)) * 100.0                 # percent
    dates = np.array(dates[1:])
    np.savez(f, dX=dX, Y=Y, Yv=Yv, dates=dates)
    return dX, Y, Yv, dates


# ------------------------------------------------------------------- tail index

def hill(x, k):
    v = np.sort(np.abs(x))[::-1]
    return 1.0 / np.mean(np.log(v[:k] / v[k])) if k < len(v) and v[k] > 0 else np.nan


def sboot(x, k, B, rng, mb):
    n, p, out = len(x), 1.0 / mb, np.empty(B)
    for b in range(B):
        idx, i = np.empty(n, dtype=np.int64), 0
        while i < n:
            st = rng.integers(0, n)
            L = min(rng.geometric(p), n - i)
            idx[i:i + L] = (st + np.arange(L)) % n
            i += L
        out[b] = hill(x[idx], k)
    return out[np.isfinite(out)]


def tail_report(x, rng, label, B=2000):
    n = len(x)
    ks = np.unique(np.linspace(n / 40, n / 10, 40).astype(int))
    hp = np.array([hill(x, int(k)) for k in ks])
    k = int(np.median(ks))
    pt = hill(x, k)
    d = sboot(x, k, B, rng, max(2, int(round(n ** (1 / 3)))))
    z = 1.959963984540054
    lo = min(np.quantile(d, .025), pt / (1 + z / np.sqrt(k)), np.nanmin(hp))
    hi = max(np.quantile(d, .975), pt / (1 - z / np.sqrt(k)), np.nanmax(hp))
    print(f"   {label:<34} n={n:5d}  point={pt:5.3f}  95% [{lo:5.3f}, {hi:5.3f}]"
          + ("   <-- below 2" if hi < 2 else "   straddles 2" if lo < 2 < hi else ""))
    return dict(label=label, n=n, point=float(pt), lo=float(lo), hi=float(hi))


# ------------------------------------------------------------------ the test

def _w(X, theta):
    """Weights with the zero-norm convention of twomoments.operator."""
    n = len(X)
    if not theta:
        return np.ones(n)
    nrm = np.linalg.norm(X, axis=1)
    w = np.zeros(n)
    nz = nrm > 0
    w[nz] = nrm[nz] ** (-theta)
    return w


def run_test(X, Y, theta, rng, m_max=60, ndraw=20000):
    n = len(Y)
    b = np.zeros(X.shape[1])                        # H0: beta = 0
    A, r = operator(X, Y, theta)
    a_n = 1.0 / (np.sqrt(n) * np.log(n))
    for m in range(1, m_max + 1):
        bh = cg(A, r, m)
        if np.linalg.norm(r - A @ bh) <= a_n:
            break
    stat = n * float(np.sum((A @ (bh - b)) ** 2))
    Z = X * (_w(X, theta) * (Y - X @ b))[:, None]   # exact residuals under H0
    ev = np.clip(np.linalg.eigvalsh(Z.T @ Z / n), 0.0, None)
    ev = ev[ev > 1e-14]
    draws = (rng.chisquare(1.0, (ndraw, ev.size)) * ev).sum(1)
    return dict(theta=theta, m=m, stat=stat, tr_V=float(ev.sum()),
                crit05=float(np.quantile(draws, .95)),
                p=float((draws > stat).mean()))


def main():
    rng = np.random.default_rng(SEED)
    dX, Y, Yv, dates = load()
    nrm = np.linalg.norm(dX, axis=1)
    yr = np.array([d[:4] for d in dates]).astype(int)
    nzero = int((nrm == 0).sum())
    print(f"AA financial commercial-paper curve x NASDAQ, {dates[0]} .. {dates[-1]}, "
          f"n = {len(Y)}")
    print(f"   days with dX_t = 0 exactly (curve unchanged): {nzero} "
          f"({nzero / len(Y):.1%}) -- these carry no direction, and the")
    print(f"   zero-norm convention t_theta(0) := 0 drops them from the score.\n")

    print("1. TAIL INDEX OF ||dX_t||, and is it one episode?")
    rows = [tail_report(nrm, rng, "full sample")]
    for lab, msk in (("excluding 2007-2009", (yr < 2007) | (yr > 2009)),
                     ("excluding 2008 alone", yr != 2008),
                     ("2010 onward only", yr >= 2010),
                     ("pre-2007 only", yr < 2007)):
        rows.append(tail_report(nrm[msk], rng, lab))
    print("   responses, for reference -- these must stay ABOVE 2 for theta=1 to help,")
    print("   since tr V_1 = E eps^2 is the one moment the theorem does need:")
    rows.append(tail_report(np.abs(Y), rng, "|Y_t| NASDAQ return"))
    rows.append(tail_report(np.abs(Yv), rng, "|Y_t| VIX log change"))

    print()
    print("2. DOES tr Vhat SETTLE?  Theorem 6' says tr V_0 = E[eps^2 ||X||^2] has")
    print("   nothing to converge to when the index is below 2, while tr V_1 = E eps^2")
    print("   does.  Expanding subsamples, H0: beta = 0, Panel A.")
    print(f"   {'n':>7} " + "".join(f"{'tr V(th=' + str(t) + ')':>17}"
                                    for t in (0.0, 0.5, 1.0)))
    tr_rows = []
    for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
        k = int(frac * len(Y))
        vals = []
        for th in (0.0, 0.5, 1.0):
            w = _w(dX[:k], th)
            Z = dX[:k] * (w * Y[:k])[:, None]
            vals.append(float(np.trace(Z.T @ Z / k)))
        print(f"   {k:7d} " + "".join(f"{v:17.4g}" for v in vals))
        tr_rows.append({"n": k, "tr": vals})
    print("   -> a column that keeps climbing has no population limit; one that")
    print("      settles does.  This is the diagnostic, on real data.")

    print()
    print("3. THE TESTS, H0: beta = 0, under the overfitting stopping rule")
    res = {}
    for panel, (Xp, Yp) in (("A  contemporaneous, Y = equity", (dX, Y)),
                            ("B  predictive,      Y = equity", (dX[:-1], Y[1:])),
                            ("C  contemporaneous, Y = vol   ", (dX, Yv)),
                            ("D  predictive,      Y = vol   ", (dX[:-1], Yv[1:]))):
        print(f"\n   Panel {panel}")
        print(f"   {'theta':>7} {'m':>4} {'statistic':>13} {'crit 5%':>13} "
              f"{'tr Vhat':>12} {'p':>8}")
        for th in (0.0, 0.5, 1.0):
            d = run_test(Xp, Yp, th, rng)
            res[f"{panel[0]}_{th}"] = d
            tag = " (BCT)" if th == 0 else "      "
            print(f"   {th:5.1f}{tag} {d['m']:4d} {d['stat']:13.4g} "
                  f"{d['crit05']:13.4g} {d['tr_V']:12.4g} {d['p']:8.4f}")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "empirical2_cp.json", "w") as f:
        json.dump({"span": [str(dates[0]), str(dates[-1])], "n": int(len(Y)),
                   "tail": rows, "trace": tr_rows,
                   "tests": res}, f, indent=2, default=float)
    print(f"\nwrote {OUT / 'empirical2_cp.json'}")


if __name__ == "__main__":
    main()
