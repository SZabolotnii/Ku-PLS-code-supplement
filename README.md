# Ku-PLS — Lean 4 formal-verification supplement

Machine-checked deterministic core for the paper

> **Moment-Free Inference for Functional Partial Least Squares**
> Serhii Zabolotnii (ORCID 0000-0003-0242-2234)

This repository holds the **formal verification** half of the paper's artifacts. The
Monte Carlo study of Section 10 and the empirical application of Section 11 are
replicated separately, at
[SZabolotnii/Ku-PLS-replication](https://github.com/SZabolotnii/Ku-PLS-replication).

This Lean 4 / Mathlib development certifies, sorry-free, the *deterministic algebraic* facts on which the paper's operator theory rests: the empirical-characteristic-function (ECF) Gram operator is symmetric and positive semidefinite, and the conjugate-gradient / Krylov projection identities that carry the functional-PLS machinery over to that operator. The probabilistic asymptotics (concentration, the weighted-$\chi^2$ limit) are classical and live in the manuscript, not here.

## What is certified

| Paper result | Lean lemma (`namespace CFPLS`) | File |
|---|---|---|
| ECF Gram symmetric (Thm 3.1) | `ecfGram_isHermitian` | `KuPLS/Core.lean` |
| Quadratic form $=\lVert\cdot\rVert^2$ | `ecfQuadForm_eq_normSq` | `KuPLS/Core.lean` |
| **ECF Gram PSD — moment-free crux (Thm 3.1)** | `ecfGram_posSemidef` | `KuPLS/Core.lean` |
| Galerkin $\leftrightarrow$ projection (Thm 4.1) | `normalSystem_iff_starProjection` | `KuPLS/Krylov.lean` |
| CG energy $= a^{\mathsf T}Y$ (Thm 4.1) | `energy_eq_dotProduct` | `KuPLS/Krylov.lean` |
| **CG residual $\perp$ Krylov subspace (Thm 4.1)** | `krylov_residual_orthogonal` | `KuPLS/Krylov.lean` |

## Axiom audit

`KuPLS/Audit.lean` runs `#print axioms` on all six lemmas. Every one depends on **only** the three standard Lean/Mathlib axioms — no `sorryAx`, no custom axioms:

```
'CFPLS.ecfGram_posSemidef' depends on axioms: [propext, Classical.choice, Quot.sound]
```

The full transcript is in [`AUDIT.txt`](AUDIT.txt).

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
lean/
├── KuPLS.lean              -- root import
├── KuPLS/Core.lean         -- ECF Gram: Hermitian, quadratic form, PSD crux
├── KuPLS/Krylov.lean       -- CG/Krylov projection identities
├── KuPLS/Audit.lean        -- #print axioms for all six lemmas
├── lakefile.lean
├── lean-toolchain
├── lake-manifest.json
└── AUDIT.txt               -- captured axiom-audit transcript
```

## License

MIT — see [`LICENSE`](LICENSE).
