import Mathlib.Analysis.InnerProductSpace.Projection.Basic
import Mathlib.Analysis.InnerProductSpace.Projection.FiniteDimensional
import Mathlib.Analysis.InnerProductSpace.Projection.Submodule
import Mathlib.Analysis.Normed.Module.FiniteDimension
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Data.Matrix.Basic

/-!
# CF-functional-PLS — Core deterministic facts

Machine-checked, **moment-free** algebraic backbone for the
characteristic-function (ECF) reformulation of functional partial least squares
(the conjugate-gradient / Krylov machinery of Babii, Carrasco & Tsafack, 2025).

The two deterministic claims certified here are exactly the ones the proof
document leans on as *unconditional* (no moment assumption on the predictor):

* `ecfGram_isHermitian` / `ecfGram_posSemidef` — the finite ECF-frequency Gram
  matrix `Mᵢⱼ = ⟪vᵢ, vⱼ⟫` of the bounded ECF feature vectors
  `vᵢ = (Re e^{i⟨uᵢ,·⟩}, Im e^{i⟨uᵢ,·⟩})` (real-stacked, so it lives in a *real*
  inner-product space) is **symmetric and positive semidefinite**.  No moment of
  `X` is used: the feature maps `x ↦ e^{i⟨u,x⟩}` are bounded by `1`, so the Gram
  is a covariance of bounded vectors and PSD-ness is purely the
  Gram-of-vectors fact.  This is the crux that breaks the finite-4th-moment
  dependency of the baseline covariance operator `K̂`.

* `ecfQuadForm_eq_normSq` — the quadratic form of the Gram equals the squared
  norm of the corresponding linear combination of features, i.e.
  `xᵀ M x = ‖∑ xᵢ vᵢ‖²`.  This is the load-bearing identity behind PSD and
  behind the A-norm minimisation used by conjugate gradient.

The vectors `v : Fin n → E` are *arbitrary* elements of a real inner-product
space; in the application `E = L²(π) ×' L²(π)` is the real-stacked frequency
Hilbert space and `vᵢ` is the (centred) real-stacked ECF feature at frequency
`uᵢ`.  The PSD proof needs nothing about them.
-/

namespace CFPLS

open scoped BigOperators
open Submodule

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable {n : ℕ}

/-- The finite ECF-frequency **Gram matrix** of a family of (real-stacked) ECF
feature vectors `v : Fin n → E`.  In the application `vᵢ` is the bounded,
moment-free feature `u ↦ e^{i⟨uᵢ,·⟩}` stacked into the real Hilbert space; the
construction here is the generic Gram matrix and needs no boundedness. -/
noncomputable def ecfGram (v : Fin n → E) : Matrix (Fin n) (Fin n) ℝ :=
  fun i j => inner ℝ (v i) (v j)

/-- The Gram matrix is **symmetric** (Hermitian over `ℝ`): `Mᵀ = M`. -/
theorem ecfGram_isHermitian (v : Fin n → E) : (ecfGram v).IsHermitian := by
  ext i j
  simp only [Matrix.conjTranspose_apply, ecfGram, star_trivial]
  exact real_inner_comm (v i) (v j)

/-- **Quadratic-form identity.** `xᵀ M x = ‖∑ xᵢ vᵢ‖²`.  This is the engine of
both PSD-ness and the conjugate-gradient A-norm minimisation. -/
theorem ecfQuadForm_eq_normSq (v : Fin n → E) (x : Fin n → ℝ) :
    x ⬝ᵥ (ecfGram v).mulVec x = ‖∑ i, x i • v i‖ ^ 2 := by
  have hlhs : x ⬝ᵥ (ecfGram v).mulVec x
      = ∑ i, ∑ j, x i * x j * inner ℝ (v i) (v j) := by
    simp only [dotProduct, Matrix.mulVec, ecfGram, Finset.mul_sum]
    exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring
  have hrhs : ‖∑ i, x i • v i‖ ^ 2
      = ∑ i, ∑ j, x i * x j * inner ℝ (v i) (v j) := by
    rw [← real_inner_self_eq_norm_sq, sum_inner]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [inner_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [real_inner_smul_left, real_inner_smul_right, mul_assoc]
  rw [hlhs, hrhs]

/-- **Theorem (moment-free Gram PSD).** The ECF-frequency Gram matrix is
**positive semidefinite** — symmetric with nonnegative quadratic form — with
*no* moment assumption on the predictor.  The ECF features are bounded
(`|e^{i⟨u,x⟩}| = 1`), so this is a covariance of bounded vectors and PSD-ness is
unconditional.  This is the moment-free replacement for the finite-4th-moment
positivity of the empirical covariance operator `K̂` in the baseline. -/
theorem ecfGram_posSemidef (v : Fin n → E) : (ecfGram v).PosSemidef := by
  refine Matrix.PosSemidef.of_dotProduct_mulVec_nonneg (ecfGram_isHermitian v) (fun x => ?_)
  have hx : (star x) ⬝ᵥ (ecfGram v).mulVec x = ‖∑ i, x i • v i‖ ^ 2 := by
    have hs : star x = x := rfl
    rw [hs, ecfQuadForm_eq_normSq]
  rw [hx]
  exact pow_two_nonneg _

end CFPLS
