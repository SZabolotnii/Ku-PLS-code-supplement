# Replication package — Sections 10 and 11

Code for every number in

> **Moment-Free Inference for Functional Partial Least Squares**
> Serhii Zabolotnii (ORCID 0000-0003-0242-2234)

Two studies: the Monte Carlo of Section 10 and the empirical application of
Section 11. Both tables in the paper are emitted by this code — no number is
transcribed by hand.

The Lean 4 formalisation of the paper's deterministic algebraic core is the
other half of this repository — see [`../README.md`](../README.md).

## Run it

Everything below runs from **this** directory (`replication/`).

```bash
pip install -r requirements.txt
./run_all.sh          # ~10 min for the Monte Carlo, seconds for the rest
```

or piecewise:

```bash
python -m kupls.simulate 1000    # -> results/simulation.json
python -m kupls.empirical        # -> results/empirical.json   (pulls FRED)
python -m kupls.tables           # -> results/table_{sim,empirical}.tex
```

`results/` holds the exact output the paper reports, so you can diff a fresh
run against it.

## What the two statistics are

| | null hypothesis | needs |
|---|---|---|
| `T_n = n‖K̂(β̂_m − b)‖²` | `β = b` through the second-moment normal equation | `E‖X‖⁴ < ∞` for its limit law |
| `S_n = n‖ρ̂ − ρ̂_b‖²` | `ρ(s,u) = φ_X(u+sb)·φ_ε(s)`, i.e. `ε` **independent** of `X` | no moment of `X`, `ε` or `Y` |

**These are different nulls, and it matters.** With `b = 0`, `S_n` is a
characteristic-function test of independence in the ICM tradition, while `T_n`
only asks whether the linear slope vanishes. On data with volatility linkage the
two can disagree, and the disagreement is a difference of hypotheses rather
than of power. Section 11 of the paper turns on this point; see `kupls/core.py`.

Both are calibrated identically and each by its own theory: the eigenvalues of
its plug-in variance operator, with critical values or *p*-values from
`Σ ω̂_j Z_j²`.

## Monte Carlo (Section 10)

α-stable functional predictors on `L²[0,1]`, `α ∈ {1.2, 1.5, 1.8, 2.0}` (2 is the
Gaussian benchmark), `n ∈ {200, 500, 1000, 2000}`, 1000 replications per cell,
`m = 3` CG steps, nominal 5%. Power is against the local alternative
`b₁ ↦ b₁(1 + n^{-1/2})` and is **size-corrected**, without which `T_n`'s
over-rejection would flatter it.

Gaussian errors are used in the main design deliberately: with them the variance
operator `V = E[ε²X⊗X]` fails to exist through `E‖X‖² = ∞` alone, isolating the
mechanism under test. One extra arm uses Cauchy errors, where `E|ε| = ∞`.

What it finds, including the parts that do not favour the proposal:

- `S_n` holds its nominal level in **all twenty cells** (0.043–0.058) and under
  Cauchy errors. `T_n` over-rejects up to eightfold (0.407 at α = 1.2) and does
  **not** improve with `n`.
- `T_n`'s divergence is conspicuous only for α ≤ 1.5 (median ×38 at α = 1.2,
  ×3.3 at α = 1.5, but only +40% at α = 1.8). The paper's own earlier wording
  overstated this and Section 10 corrects it.
- **`T_n` is markedly more powerful wherever fourth moments exist** — 0.733–0.836
  against `S_n`'s 0.287–0.350 at α = 1.8 and 2.0. The ECF statistic buys a valid
  level, not a uniformly better test.
- Under Cauchy errors `T_n`'s *raw* size looks fine (0.036–0.105) because its
  plug-in critical value inflates in step with the statistic; its size-corrected
  power is 0.064–0.080, i.e. none. A correct-looking rejection rate is not
  evidence that a limit exists.

## Empirical application (Section 11)

- `X_t` — daily changes in eleven constant-maturity Treasury yields (1M–30Y), in
  basis points: a discretised function of maturity.
- `Y_t` — S&P 500 daily log return, percent.
- 2494 trading days, 2016-08-16 to 2026-08-13.

Data come from **FRED** (Federal Reserve Bank of St. Louis) over plain HTTPS —
no key, no subscription, no redistribution. The first run caches to
`data/fred_cache.npz`; delete it or pass `refresh=True` to re-pull.

Findings:

- Hill tail-index estimates: `‖X_t‖` ≈ 3.0, `|Y_t|` ≈ 2.5. The **variance is
  finite** — this is not an infinite-variance sample and the paper does not
  claim it is — but BCT need a tail index above 4, and the fourth moment is not
  there. Standard daily financial data sits in exactly the gap the method
  occupies.
- Sample excess kurtosis, recorded under `kurtosis` in `empirical.json`:
  `‖X_t‖` = 13.627, `Y_t` = 16.631. Note which series each is taken on — the
  tail index is estimated on `|Y_t|`, the kurtosis on the signed `Y_t`, and the
  paper quotes them that way. At a tail index near 3 these are sample versions
  of a population quantity that does not exist, which is the point.
- Panel A (contemporaneous): both statistics reject. The ECF construction is not
  inert on real data.
- Panel B (predictive): they disagree — `T_n` p = 0.062, `S_n` p = 0.0021,
  stable across `m` and stronger when 2020 is excluded. This is the different
  nulls at work: `corr(‖X_t‖, |Y_{t+1}|) = +0.20` (p ≈ 1.6e-24) while the signed
  rank correlation is −0.002 (p = 0.93). Today's rate shock predicts tomorrow's
  *magnitude*, not its direction. Independence is violated; a zero linear slope
  is not.
- `S_n` is numerically identical across `m ∈ {1,2,3,5}` — under `H₀: β = b` it
  never touches `β̂_m`, so the early-stopping choice cannot contaminate the test.

## Reproducibility

Seeding is explicit — `np.random.default_rng([seed, 10α, n, eps_code])`. Two
fresh processes give bit-identical output. (An earlier draft derived the seed
from Python's `hash()`, which is salted per process; that would have made the
published table irreproducible. The note is left here because it is an easy
mistake to repeat.)

Requires `numpy` and `scipy` only. Verified with Python 3.13.

## Layout

```
kupls/core.py       sampling, both statistics, calibration, Hill estimator
kupls/simulate.py   the Monte Carlo study
kupls/empirical.py  FRED retrieval and the two specification tests
kupls/tables.py     emits the paper's LaTeX tables from stored results
results/            output as reported in the paper
run_all.sh          end-to-end
```

## Licence

MIT — see [`../LICENSE`](../LICENSE).
