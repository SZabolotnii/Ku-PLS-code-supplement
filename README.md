# Ku-PLS — code supplement

Everything needed to check two papers on functional partial least squares under
weak moment conditions, by Serhii Zabolotnii (ORCID 0000-0003-0242-2234).

| Directory | Belongs to | Contents |
|---|---|---|
| [`KuPLS/`](KuPLS/) | paper 1 | Lean 4 / Mathlib proofs of the deterministic algebraic core, sorry-free with an axiom audit |
| [`replication/`](replication/) | paper 1 | Python replication of the Monte Carlo study and the empirical application; regenerates both of its tables from source |
| [`two-moments/`](two-moments/) | paper 2 | Python replication of the simulations, the two applications and the checks behind every claim |

> **Paper 1 — *Moment-Free Inference for Functional Partial Least Squares*.**
> An empirical-characteristic-function operator, so that a specification test
> needs no moment of the predictor at all.
>
> **Paper 2 — *Functional Partial Least Squares under Two Moments*.**
> A reweighted family $A_\theta=\mathbb E[\lVert X\rVert^{-\theta}X\otimes X]$
> that contains the second-moment baseline at $\theta=0$ and the spatial-sign
> covariance at $\theta=2$, and a rule for choosing $\theta$ from the data.

The three parts are independent. The Lean development needs `elan`/`lake`; each
replication package needs only `numpy` and `scipy`, and neither imports the
other.

## Replication

Paper 2 (see [`two-moments/README.md`](two-moments/README.md) for the map from
each table to its script):

```bash
cd two-moments && pip install -r requirements.txt && ./run_all.sh
git diff --stat results/
```

Paper 1 (details in [`replication/README.md`](replication/README.md), including
the parts of that study which do not favour its own proposal):

```bash
cd replication && pip install -r requirements.txt && ./run_all.sh
```

Both packages ship their stored output, so re-running is a diff rather than a
comparison by eye, and both fix their seeds explicitly: two fresh processes give
bit-identical output. No number in either paper is transcribed by hand.

Data in both packages comes from **FRED** (Federal Reserve Bank of St. Louis)
over plain HTTPS — no key, no subscription, no redistribution restriction.

## Lean development

This Lean 4 / Mathlib development certifies, sorry-free, the *deterministic algebraic* facts on which paper 1's operator theory rests — the empirical-characteristic-function (ECF) Gram operator is symmetric and positive semidefinite, and the conjugate-gradient / Krylov projection identities carry the functional-PLS machinery over to that operator — and, since 2026-09-07, one *measure-theoretic* step of paper 2: the decomposition $r_\theta = A_\theta\beta + \mathbb E[\varepsilon\,w_\theta(X)X]$ behind the Exactness lemma, with the noise term Bochner-integrable **by hypothesis**. The remaining probabilistic asymptotics (concentration, the weighted-$\chi^2$ limit, the martingale CLT) are classical and live in the manuscripts, not here.

## What is certified

| Paper result | Lean lemma (`namespace CFPLS`) | File |
|---|---|---|
| ECF Gram symmetric (Thm 3.1) | `ecfGram_isHermitian` | `KuPLS/Core.lean` |
| Quadratic form $=\lVert\cdot\rVert^2$ | `ecfQuadForm_eq_normSq` | `KuPLS/Core.lean` |
| **ECF Gram PSD — moment-free crux (Thm 3.1)** | `ecfGram_posSemidef` | `KuPLS/Core.lean` |
| Galerkin $\leftrightarrow$ projection (Thm 4.1) | `normalSystem_iff_starProjection` | `KuPLS/Krylov.lean` |
| CG energy $= a^{\mathsf T}Y$ (Thm 4.1) | `energy_eq_dotProduct` | `KuPLS/Krylov.lean` |
| **CG residual $\perp$ Krylov subspace (Thm 4.1)** | `krylov_residual_orthogonal` | `KuPLS/Krylov.lean` |
| **Paper 2, Exactness lemma — decomposition step, repaired hypotheses** | `exactness_decomposition` (`namespace KuPLS`) | `KuPLS/Exactness.lean` |
| Paper 2, pointwise split of $w(X)\,Y\,X$ | `weighted_split` | `KuPLS/Exactness.lean` |
| Negative control: Bochner integral is $0$ off `Integrable` (Mathlib's `integral_undef`) | `integral_is_junk_off_integrable` | `KuPLS/Exactness.lean` |

## Axiom audit

`KuPLS/Audit.lean` runs `#print axioms` on all nine theorems. Every one depends on **only** the three standard Lean/Mathlib axioms — no `sorryAx`, no custom axioms:

```
'CFPLS.ecfGram_posSemidef' depends on axioms: [propext, Classical.choice, Quot.sound]
```

The full transcript is in [`AUDIT.txt`](AUDIT.txt).

### Why the paper-2 module exists, and what it does not certify

On 2026-09-07 a statement-level audit refuted paper 2's Exactness lemma as originally stated: its proof asserted the noise term integrable "because $\|\varepsilon w_\theta(X)X\| = |\varepsilon|\,\|X\|^{1-\theta}$" — an identity, not an integrability proof — and two counterexamples satisfying every stated hypothesis make $r_\theta$ undefined. The repair adds $\mathbb E[|\varepsilon|\,\|X\|^{1-\theta}] < \infty$ as a hypothesis. `Exactness.lean` states the decomposition step with that hypothesis list and proves it from Mathlib's Bochner integral.

Two things it deliberately does **not** do. It does not prove the joint moment *necessary* — the counterexamples do that, and they are reproduced in the audit rather than formalised. And it does not formalise conditional Fubini: the centring of the noise term is taken as a hypothesis, exactly as the paper's own proof takes it once integrability holds.

The reason the hypothesis is carried *explicitly* is a property of Mathlib worth knowing before trusting any green build over an integral: `MeasureTheory.integral_undef` makes the Bochner integral **equal to zero off `Integrable`**. A formalisation that quietly omits integrability does not fail — it compiles and proves a statement about the value $0$. This module routes the proof through `integral_add`, which *demands* both integrability arguments, so the same statement with `hnoise` removed fails with `unsolved goals` and pulls `sorryAx`. That negative control is what makes the green build meaningful.

## Build

- Toolchain: Lean 4 (`leanprover/lean4:v4.26.0`, pinned in `lean-toolchain`).
- Dependency: Mathlib `v4.26.0` (pinned in `lake-manifest.json`).

```bash
# fetch the prebuilt Mathlib cache (avoids a multi-hour Mathlib rebuild)
lake exe cache get
lake build              # builds the development
lake env lean KuPLS/Audit.lean   # reproduces the axiom audit
```

> **Note for a fresh clone.** In the author's working tree `.lake/packages` is a local symlink to a sibling project's Mathlib build and is intentionally *not* committed (see `.gitignore`). On a clean checkout, `lake exe cache get` provisions Mathlib from scratch.

## Layout

```
KuPLS.lean              -- root import
KuPLS/Core.lean         -- ECF Gram: Hermitian, quadratic form, PSD crux
KuPLS/Krylov.lean       -- CG/Krylov projection identities
KuPLS/Audit.lean        -- #print axioms for all six lemmas
lakefile.lean
lean-toolchain
lake-manifest.json
AUDIT.txt               -- captured axiom-audit transcript
replication/            -- paper 1: Monte Carlo + empirical application
two-moments/            -- paper 2: code, gates, results, data
```

## License

MIT — see [`LICENSE`](LICENSE).
