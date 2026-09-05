# Replication package — *Functional Partial Least Squares under Two Moments*

Code and stored output for every computed number in

> **Functional Partial Least Squares under Two Moments**
> Serhii Zabolotnii (ORCID 0000-0003-0242-2234)

This is the second paper of the project. The first one — the moment-free
characteristic-function route — is the other half of this repository
(`../replication/` for its Python, `../KuPLS/` for its Lean proofs), and the two
packages are independent: nothing here imports anything there.

## Requirements

`numpy` and `scipy`, nothing else. Verified with Python 3.13, numpy 2.4.6,
scipy 1.17.1.

```bash
pip install -r requirements.txt
```

## Run it

```bash
./run_all.sh              # everything, in dependency order (hours on one core)
./run_all.sh sim          # only the simulation scripts
./run_all.sh empirical    # only the scripts that read data/ (or FRED)
git diff --stat results/  # what moved
```

Every script writes its own `results/<name>.json`, and `run_all.sh` captures its
stdout to the matching `results/<name>.out`. Both are committed here as they
were when the paper was written, so a fresh run is a diff, not a comparison by
eye. Seeds are fixed inside each script. One line per artifact will always
differ: each script closes by printing the absolute path it wrote to, and the
stored copies carry that path rewritten to `two-moments/…`.

Any single script also runs on its own, from anywhere:

```bash
python code/split_rule_tables.py      # Tables 7 and 8
python gates/a1_7_power_confirm.py    # Table 6
```

## Layout

```
code/     the estimator, the two statistics, the selection rule, the designs,
          the empirical panels, and the scripts behind the paper's tables
gates/    the checks that decided the paper's claims -- each one asks a single
          question and prints a verdict, including the ones that failed
results/  every .json and .out as reported in the paper
data/     the exact FRED panels the applications use (see below)
```

`code/twomoments.py`, `code/theta_selector.py` and `code/dependent_design.py`
are libraries; everything else is a script with a `__main__`.

## Which script produces which table

| Table | What | Script | Artifact |
|---|---|---|---|
| 1 | The moment ledger | analytic — no artifact | — |
| 2 | Where the rate and the comparison hold | analytic — no artifact | — |
| 3 | Size under dependence in both components | `gates/a1_8b_mds_boundary.py` | `a1_8b_mds_boundary` |
| 4 | Calibration of $\kappa$ | `gates/a3_kappa_recalibrate.py`, `gates/a3_kappa_phi_sensitivity.py` | `a3_kappa_recalibrate`, `a3_kappa_phi_sensitivity` |
| 5 | Margins of the three selections | `gates/a3_kappa_margin.py` | `a3_kappa_margin` |
| 6 | Size and power below tail index two | `gates/a1_7_power_confirm.py` | `a1_7_power_confirm` |
| 7 | Estimation error at $n=1000$ | `code/split_rule_tables.py` | `split_rule_tables` |
| 8 | Estimation error at tail index 1.5 across $n$ | `code/split_rule_tables.py` | `split_rule_tables` |
| 9 | The two stopping thresholds compared | `code/split_stopping_rule.py` (part D) | `split_stopping_rule` |
| 10 | Commercial paper against VIX, bootstrap null | `code/empirical2_blockboot.py` | `empirical2_blockboot` |
| 11 | What the rule picks on the applications | `code/selector_on_applications.py` | `selector_on_applications` |
| A.1 | Off-diagonal mass under the two hypotheses | `code/section43_hypotheses.py` | `section43_hypotheses` |
| A.2 | Shrinkage factors on the population diagonal | `code/shrink_table.py` | `shrink_table` |

Numbers quoted in the text but not in a table follow the same convention: the
script's name is the artifact's stem. The exceptions are
`run_pilot.py` → `a2_pilot_estimation`, `run_size_confirm.py` →
`a2_size_confirm`, `run_inference_pilot.py` → `a2_pilot_inference`,
`run_dependence.py` → `dependence` and `run_theta_selector.py` →
`theta_selector`; `run_all.sh` writes each one to its shipped name.

### Artifacts whose driver was not kept

Said plainly, because a replication package that hides this is worth less than
one that does not.

Two of the paper's tables were produced by interactive drivers that were not
retained. Both now have one, and the tables print what the shipped script
returns:

- **Table 11** ← `code/selector_on_applications.py`. The deterministic half
  reproduced the lost run exactly: $R_n(\theta)$ agrees to every digit recorded
  in `results/selector_recalibrated.json`, and so does every pick. The test at
  $\hat\theta$ is Monte Carlo — 20,000 weighted-$\chi^2$ draws and 999 bootstrap
  replicates — so its p-values and the critical-value ratio moved when the seed
  did, and the table was updated to the shipped seed. The script also prints
  $R_n$ under both residual conventions in use in the paper (the pre-test's own,
  under $H_0:\beta=0$, and the OLS residuals `gates/a3_kappa_margin.py` uses) and
  fails if they ever disagree on a pick.
- **Table A.2** ← `code/shrink_table.py`. It reproduced the lost run to within
  Monte Carlo error, which is the point: at $4\times10^{6}$ draws each cell has a
  standard error near $9\times10^{-4}$, so the third decimal is noise. The
  $\theta=0$ row, where $d_j\equiv1$ exactly, is the control — it returns a tail
  slope of $-0.0008$, which is what a slope of zero reads at this many draws.
  The table's earlier caption claimed the slope was zero to $4\times10^{-4}$;
  that was a property of one seed, and the caption now states the claim against
  the $\theta=0$ row instead.

One artifact still has no driver: **`results/rule_size_confirm.{json,out}`**
(§6.3, the level of the selection rule under dependence). The same experiment is
re-run by `gates/a3_kappa_recalibrate.py`, whose step 2 reports the rule's size
at both the shipped and the recalibrated $\kappa$.

`results/selector_recalibrated.{json,out}` is kept as the record of the
superseded run; nothing in the paper reads it any more.

## Data

Eight `.npz` panels in `data/`, all built from **FRED** (Federal Reserve Bank of
St. Louis) over plain HTTPS — no key, no subscription. `code/empirical2_cp.py`
and `code/probe_heavy_curves.py` re-pull and re-cache when a file is missing, so
the package is self-contained but not sealed off.

The caches are committed rather than left to a live pull because FRED revises
its series: a pull in a year's time would give a slightly different panel and
therefore slightly different digits, and there would be no way to tell a
revision from a bug. The two panels the paper's applications use are

- `fred_cache.npz` — eleven constant-maturity Treasury yields against S&P 500
  returns, $n=2494$ (the same file as `../replication/data/`, which paper 1
  re-pulls);
- `cp_nasdaq.npz` — the six-maturity commercial-paper curve against VIX and
  NASDAQ, $n = 4396$, the dates common to the curve and both responses.

The others (`treasury.npz`, `cp-term.npz`, `cp-financial.npz`, `credit-oas.npz`,
`hy-curve.npz`, `ig-ladder.npz`, `tips-curve.npz`) are the wider set of curves
screened in `code/probe_heavy_curves.py` before the two applications were
chosen. Appendix D of the paper states how each panel is constructed. Note that
`treasury.npz` ($n = 6264$) is the raw screening curve and is **not** the panel
the Treasury application runs on.

## Reading the code

The docstrings are the author's working notes and they are blunt: each script
opens with the defect or the question that caused it to be written, and several
report that the paper's earlier wording was wrong. They occasionally point at
internal theory notes (`theory/…`) that are not part of this package; the
statements they name are the propositions and lemmas of the paper's appendices.

Three files in the author's tree are deliberately not published here: two text
gates over the manuscript and the claim-to-artifact resolver used to audit it.
They reproduce no number in the paper.

## Licence

MIT — see [`../LICENSE`](../LICENSE).
