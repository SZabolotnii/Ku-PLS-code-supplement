#!/usr/bin/env python3
"""Table 11: what the selection rule picks on the three panels, and the test there.

WHY THIS EXISTS.  The table was produced by an interactive driver that was not
kept.  Its artifact, results/selector_recalibrated.{json,out}, therefore recorded
numbers that no script in the repository regenerated -- the one gap in the
package a replicator would hit, on a table of the paper.  This closes it.

WHAT IS DETERMINISTIC AND WHAT IS NOT.  The selector is deterministic given the
data: R_n(theta) is a ratio of two sums, theta-hat is the least grid point at or
below kappa, and both reproduce exactly.  The test at theta-hat is not: its
p-values come from 20,000 weighted-chi^2 draws and 999 stationary-bootstrap
replicates, so they carry Monte Carlo error and move with the seed.  The seed is
fixed here, so the table is reproducible from now on; the ORIGINAL run's digits
cannot be recovered, and the report below says which columns are which.

TWO RESIDUAL CONVENTIONS, PRINTED SIDE BY SIDE.  The rule is a PRE-test: it runs
before beta is fitted, under H0: beta = 0, where the residual is Y itself.  That
is what theta_selector_apply.py does and what the lost driver did.  The margin
table (tab:margin, via gates/a3_kappa_margin.py) instead computes R_n on OLS
residuals.  The two curves differ -- on CP x VIX, R_n(0) is 0.605 against 0.468 --
and nothing in the manuscript says they were computed differently.  They agree on
every pick, which is why the discrepancy never surfaced.  Both are printed here,
and the script FAILS if they ever disagree on a pick, because then the paper
would be quoting one convention's margin beside the other's selection.

KAPPA.  The paper's operative pair is the R = 3000 recalibration printed in
tab:margin: 0.0934 for the Treasury panel, 0.0708 for the two commercial-paper
panels (gates/a3_kappa_recalibrate.py).  The shipped R = 400 pair, 0.0966 and
0.0777, is applied too: a pick that moves between them is not resolved by its
calibration, and tab:margin's whole point is that none does.

Writes results/selector_on_applications.{json,out}.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from theta_selector import max_to_sum, GRID              # noqa: E402
from empirical2_cp import load as load_cp                # noqa: E402
from empirical2_blockboot import stat_only, blocks, _w   # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "results"
TREASURY = ROOT / "data" / "fred_cache.npz"

SEED = 20260905
B = 999
NDRAW = 20000

# R = 3000, at the persistence each panel actually shows (a3_kappa_recalibrate)
KAPPA = {"Treasury x S&P 500": 0.0934, "CP x VIX": 0.0708, "CP x NASDAQ": 0.0708}
# R = 400, the pair the first calibration shipped
KAPPA_SHIPPED = {"Treasury x S&P 500": 0.0966, "CP x VIX": 0.0777,
                 "CP x NASDAQ": 0.0777}

lines: list[str] = []


def P(s=""):
    print(s)
    lines.append(s)


def pick(ratios, kappa):
    """theta-hat = the least grid point whose R_n is at or below kappa."""
    ok = np.where(np.asarray(ratios) <= kappa)[0]
    j = int(ok[0]) if len(ok) else len(GRID) - 1
    return GRID[j], not len(ok)


def test_at(X, Y, theta, rng):
    """Statistic, both calibrations, and the ratio of their critical values."""
    n = len(Y)
    mb = max(2, int(round(n ** (1 / 3))))
    obs = stat_only(X, Y, theta)

    null = np.empty(B)
    for b in range(B):
        null[b] = stat_only(X[blocks(n, mb, rng)], Y[blocks(n, mb, rng)], theta)
    boot_p = float((null >= obs).mean())
    boot_c = float(np.quantile(null, 0.95))

    Z = X * (_w(X, theta) * Y)[:, None]
    ev = np.clip(np.linalg.eigvalsh(Z.T @ Z / n), 0.0, None)
    ev = ev[ev > 1e-14]
    draws = (rng.chisquare(1.0, (NDRAW, ev.size)) * ev).sum(1)
    asym_p = float((draws > obs).mean())
    asym_c = float(np.quantile(draws, 0.95))

    return dict(theta=theta, n=n, mean_block=mb, stat=obs,
                asym_p=asym_p, boot_p=boot_p, asym_c95=asym_c, boot_c95=boot_c,
                ratio=asym_c / boot_c)


def main():
    rng = np.random.default_rng(SEED)

    d = np.load(TREASURY, allow_pickle=True)
    dXt, Yt = d["dX"], d["Y"]
    dXc, Yn, Yv, _ = load_cp()
    apps = (("CP x VIX", dXc, Yv),
            ("CP x NASDAQ", dXc, Yn),
            ("Treasury x S&P 500", dXt, Yt))

    P("=" * 100)
    P("R_n(theta) UNDER BOTH RESIDUAL CONVENTIONS")
    P("   null: the rule as specified -- a pre-test under H0: beta = 0, residual = Y")
    P("   ols : residuals of the least-squares fit, as gates/a3_kappa_margin.py uses")
    P("=" * 100)
    head = f"   {'application':<20}{'resid':>6}" + \
           "".join(f"{'R_n(' + f'{t:.2f}' + ')':>11}" for t in GRID)
    P(head)

    ratios, picks = {}, {}
    for label, X, Y in apps:
        r_null = [max_to_sum(X, Y, t) for t in GRID]
        ols = Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]
        r_ols = [max_to_sum(X, ols, t) for t in GRID]
        ratios[label] = {"null": r_null, "ols": r_ols}
        for tag, r in (("null", r_null), ("ols", r_ols)):
            P(f"   {label:<20}{tag:>6}" + "".join(f"{v:11.5f}" for v in r))

        k, ks = KAPPA[label], KAPPA_SHIPPED[label]
        th, fb = pick(r_null, k)
        th_ols, _ = pick(r_ols, k)
        th_ship, _ = pick(r_null, ks)
        picks[label] = {"kappa": k, "kappa_shipped": ks, "theta_hat": th,
                        "theta_hat_ols_residual": th_ols,
                        "theta_hat_shipped_kappa": th_ship, "fallback": bool(fb)}

    P()
    P("=" * 100)
    P("THE PICK, AND WHETHER IT DEPENDS ON EITHER CHOICE")
    P("=" * 100)
    P(f"   {'application':<20}{'kappa':>9}{'th-hat':>9}"
      f"{'th (ols resid)':>16}{'th (kappa R=400)':>18}")
    disagree = []
    for label, _, _ in apps:
        p = picks[label]
        P(f"   {label:<20}{p['kappa']:9.4f}{p['theta_hat']:9.2f}"
          f"{p['theta_hat_ols_residual']:16.2f}{p['theta_hat_shipped_kappa']:18.2f}")
        if p["theta_hat"] != p["theta_hat_ols_residual"]:
            disagree.append(f"{label}: residual convention moves the pick "
                            f"({p['theta_hat']} vs {p['theta_hat_ols_residual']})")
        if p["theta_hat"] != p["theta_hat_shipped_kappa"]:
            disagree.append(f"{label}: the R=400 kappa moves the pick "
                            f"({p['theta_hat']} vs {p['theta_hat_shipped_kappa']})")
    P()
    if disagree:
        P("   FAIL -- a pick is not resolved:")
        for s in disagree:
            P(f"          {s}")
    else:
        P("   ok -- every pick survives both the residual convention and the two")
        P("        calibrations of kappa, so tab:margin's margins and tab:picks'")
        P("        selections describe the same choice.")

    P()
    P("=" * 100)
    P(f"THE TEST AT theta = 0 AND AT theta-hat, seed {SEED}")
    P(f"   Stationary-bootstrap null, B = {B}, mean block ceil(n^(1/3)); asymptotic")
    P(f"   p from the plug-in weighted chi^2, {NDRAW} draws.  These four columns are")
    P("   Monte Carlo quantities: they move with the seed, the picks above do not.")
    P("=" * 100)
    P(f"   {'application':<20}{'theta':>7}{'statistic':>12}{'asym p':>10}"
      f"{'boot p':>10}{'asym 95%':>12}{'boot 95%':>12}{'asym/boot':>11}")
    tests = {}
    for label, X, Y in apps:
        tests[label] = {}
        for th in (0.0, picks[label]["theta_hat"]):
            if f"{th:.2f}" in tests[label]:
                continue
            t = test_at(X, Y, th, rng)
            tests[label][f"{th:.2f}"] = t
            tag = "  <- theta-hat" if th == picks[label]["theta_hat"] else ""
            P(f"   {label:<20}{th:7.2f}{t['stat']:12.4g}{t['asym_p']:10.4f}"
              f"{t['boot_p']:10.4f}{t['asym_c95']:12.4g}{t['boot_c95']:12.4g}"
              f"{t['ratio']:10.2f}x{tag}")

    P()
    P("   READING.  A ratio near one means the plug-in critical value agrees with a")
    P("   dependence-respecting null; far above one is the conservatism the paper is")
    P("   about.  The p-values are resolution-limited: the bootstrap cannot report")
    P(f"   below 1/{B} = {1/B:.4f}, the weighted chi^2 below 1/{NDRAW} = {1/NDRAW}.")

    OUT.mkdir(exist_ok=True)
    (OUT / "selector_on_applications.out").write_text("\n".join(lines) + "\n",
                                                      encoding="utf-8")
    with open(OUT / "selector_on_applications.json", "w") as f:
        json.dump({"seed": SEED, "B": B, "ndraw": NDRAW, "grid": list(GRID),
                   "kappa": KAPPA, "kappa_shipped": KAPPA_SHIPPED,
                   "ratios": ratios, "picks": picks, "tests": tests},
                  f, indent=2, default=float)
    print(f"\nwrote {OUT / 'selector_on_applications.json'}")
    return 1 if disagree else 0


if __name__ == "__main__":
    sys.exit(main())
