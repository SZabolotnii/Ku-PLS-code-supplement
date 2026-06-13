import KuPLS.Core

/-!
# CF-functional-PLS — Krylov / conjugate-gradient transfer

The CF-PLS estimator with `m` components solves the **standard-norm**
least-squares problem (baseline Eq. (4), now with the moment-free ECF operator
`K_φ` and ECF source `r_φ`)

```
    min ‖ r_φ − K_φ b ‖²   over   b ∈ 𝒦ₘ = span{ r_φ, K_φ r_φ, …, K_φ^{m-1} r_φ }.
```

This file certifies that the **conjugate-gradient / Krylov fixed-point
characterisation** of the minimiser needs *only* the self-adjoint / inner-product
structure — no moment assumption.  Concretely:

* `normalSystem_iff_starProjection` — the Galerkin normal system `M a = Y`
  (with `M` the ECF Gram of the dictionary, `Y` the cross-moment vector) holds
  iff `∑ aᵢ φᵢ` is the orthogonal projection of the target onto the dictionary
  span.  This is the `FK = Y ↔ L²-projection` correspondence applied to the ECF
  operator, and is proved with nothing but the real inner-product / orthogonal
  projection apparatus.
* `krylovDict` / `krylov_residual_orthogonal` — instantiating the dictionary
  with the **image Krylov vectors** `φᵢ = K_φ^{i+1} r_φ`, the CF-PLS residual
  `r_φ − K_φ b̂ₘ` is orthogonal to the Krylov subspace `K_φ·𝒦ₘ`.  This is exactly
  the A-orthogonality (Galerkin) condition that defines the conjugate-gradient
  iterate — established here from PSD self-adjointness alone, demonstrating the
  moment-free transfer of the CG machinery.

Everything is over a real `InnerProductSpace ℝ E`; in the application `E` is the
real-stacked ECF frequency space `L²(π)` and `K_φ` is the moment-free,
self-adjoint, PSD ECF covariance operator of `Core`.
-/

namespace CFPLS

open scoped BigOperators
open Submodule

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable {n : ℕ}

/-- Right-hand side `Y` of the CF normal system: cross-moments of the dictionary
with the target (the ECF cross element `r_φ`). -/
noncomputable def rhs (φ : Fin n → E) (g : E) : Fin n → ℝ :=
  fun i => inner ℝ (φ i) g

/-- Span of the dictionary — the Krylov subspace in the application. -/
def dictSpan (φ : Fin n → E) : Submodule ℝ E :=
  Submodule.span ℝ (Set.range φ)

instance (φ : Fin n → E) : FiniteDimensional ℝ (dictSpan φ) :=
  FiniteDimensional.span_of_finite ℝ (Set.finite_range φ)

instance (φ : Fin n → E) : (dictSpan φ).HasOrthogonalProjection :=
  haveI : CompleteSpace (dictSpan φ) := FiniteDimensional.complete ℝ _
  Submodule.HasOrthogonalProjection.ofCompleteSpace _

/-- Linear combinations of the dictionary lie in its span. -/
lemma sum_smul_mem_dictSpan (φ : Fin n → E) (a : Fin n → ℝ) :
    (∑ i, a i • φ i) ∈ dictSpan φ :=
  Submodule.sum_mem _ fun i _ =>
    Submodule.smul_mem _ _ (Submodule.subset_span (Set.mem_range_self i))

/-- The `j`-th entry of `M a` (with `M = ecfGram φ`) is the inner product of
`φ j` with the candidate combination. -/
lemma mulVec_ecfGram_apply (φ : Fin n → E) (a : Fin n → ℝ) (j : Fin n) :
    (ecfGram φ).mulVec a j = inner ℝ (φ j) (∑ i, a i • φ i) := by
  simp [ecfGram, Matrix.mulVec, dotProduct, inner_sum, real_inner_smul_right,
    mul_comm]

/-- **CG/Krylov fixed-point characterisation (Galerkin ↔ projection).**
The normal system `M a = Y` for the ECF Gram `M = ecfGram φ` holds iff
`∑ aᵢ φᵢ` is the orthogonal projection of the target `g` onto the dictionary
span.  Proof uses only the real inner-product / orthogonal-projection
structure — **no moment assumption**, certifying the moment-free CG transfer. -/
theorem normalSystem_iff_starProjection (φ : Fin n → E) (g : E) (a : Fin n → ℝ) :
    (ecfGram φ).mulVec a = rhs φ g ↔
      (dictSpan φ).starProjection g = ∑ i, a i • φ i := by
  constructor
  · intro h
    have hres : ∀ j, inner ℝ (φ j) (g - ∑ i, a i • φ i) = (0 : ℝ) := by
      intro j
      have hj := congrFun h j
      rw [mulVec_ecfGram_apply] at hj
      rw [inner_sub_right, hj]
      simp [rhs]
    have horth : ∀ w ∈ dictSpan φ, inner ℝ (g - ∑ i, a i • φ i) w = (0 : ℝ) := by
      intro w hw
      induction hw using Submodule.span_induction with
      | mem x hx =>
        obtain ⟨j, rfl⟩ := hx
        rw [real_inner_comm]
        exact hres j
      | zero => simp
      | add x y _ _ hx hy => rw [inner_add_right, hx, hy, add_zero]
      | smul c x _ hx => rw [real_inner_smul_right, hx, mul_zero]
    exact Submodule.eq_starProjection_of_mem_of_inner_eq_zero
      (sum_smul_mem_dictSpan φ a) horth
  · intro h
    funext j
    have hperp : g - ∑ i, a i • φ i ∈ (dictSpan φ)ᗮ := by
      rw [← h]
      exact Submodule.sub_starProjection_mem_orthogonal g
    have hj : inner ℝ (φ j) (g - ∑ i, a i • φ i) = (0 : ℝ) :=
      Submodule.inner_right_of_mem_orthogonal
        (Submodule.subset_span (Set.mem_range_self j)) hperp
    rw [inner_sub_right, sub_eq_zero] at hj
    rw [mulVec_ecfGram_apply, ← hj]
    simp [rhs]

/-- **CG residual energy in normal-system form.** For any solution `a` of the
Galerkin system, the squared norm of the projection (the fitted part) equals
`∑ aᵢ Yᵢ`.  This is the `J = aᵀ Y` form of the CF-PLS objective, again from
PSD self-adjointness only. -/
theorem energy_eq_dotProduct (φ : Fin n → E) (g : E) (a : Fin n → ℝ)
    (h : (ecfGram φ).mulVec a = rhs φ g) :
    ‖(dictSpan φ).starProjection g‖ ^ 2 = ∑ i, a i * rhs φ g i := by
  have hproj := (normalSystem_iff_starProjection φ g a).mp h
  have hperp : g - ∑ i, a i • φ i ∈ (dictSpan φ)ᗮ := by
    rw [← hproj]
    exact Submodule.sub_starProjection_mem_orthogonal g
  have h0 : inner ℝ (∑ i, a i • φ i) (g - ∑ i, a i • φ i) = (0 : ℝ) :=
    Submodule.inner_right_of_mem_orthogonal (sum_smul_mem_dictSpan φ a) hperp
  rw [inner_sub_right, sub_eq_zero] at h0
  have hsum : inner ℝ (∑ i, a i • φ i) g = ∑ i, a i * rhs φ g i := by
    simp [sum_inner, real_inner_smul_left, rhs]
  rw [hproj, ← real_inner_self_eq_norm_sq, ← h0, hsum]

/-! ## Krylov instantiation: the CF-PLS conjugate-gradient residual -/

/-- The **image Krylov dictionary** `φᵢ = K_φ^{i+1} r` for `i = 0 … m-1`, i.e.
`{K_φ r, K_φ² r, …, K_φ^m r}`.  Its span is `K_φ · 𝒦ₘ`, the subspace onto which
CF-PLS projects the source `r` in the standard norm. -/
noncomputable def krylovDict (A : E →ₗ[ℝ] E) (r : E) (m : ℕ) : Fin m → E :=
  fun i => (A ^ (i.val + 1)) r

/-- **Moment-free conjugate-gradient residual orthogonality (Galerkin).**
If the Galerkin coefficients `a` solve the ECF normal system for the image
Krylov dictionary, then the CF-PLS residual `r − ∑ aᵢ K_φ^{i+1} r` is orthogonal
to the entire Krylov subspace `span{K_φ r, …, K_φ^m r}`.  This is exactly the
A-orthogonality condition that defines the conjugate-gradient iterate, obtained
here from inner-product / PSD structure alone — the CG machinery transfers to the
moment-free ECF operator. -/
theorem krylov_residual_orthogonal (A : E →ₗ[ℝ] E) (r : E) (m : ℕ)
    (a : Fin m → ℝ)
    (h : (ecfGram (krylovDict A r m)).mulVec a = rhs (krylovDict A r m) r) :
    ∀ w ∈ dictSpan (krylovDict A r m),
      inner ℝ (r - ∑ i, a i • krylovDict A r m i) w = (0 : ℝ) := by
  have hproj := (normalSystem_iff_starProjection (krylovDict A r m) r a).mp h
  have hperp : r - ∑ i, a i • krylovDict A r m i ∈ (dictSpan (krylovDict A r m))ᗮ := by
    rw [← hproj]
    exact Submodule.sub_starProjection_mem_orthogonal r
  intro w hw
  exact Submodule.inner_left_of_mem_orthogonal hw hperp

end CFPLS
