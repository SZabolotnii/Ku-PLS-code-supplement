/-
# Exactness — the decomposition step of Lemma `lem:exact`, as repaired 2026-09-07

WHAT THIS FILE CERTIFIES.  The lemma reads, after repair,

    Assume (MI), E‖X‖^{2−θ} < ∞ and E[|ε|·‖X‖^{1−θ}] < ∞.
    Then r_θ exists, A_θ β = r_θ exactly, and β is recovered on (ker A_θ)^⊥.

Its proof has exactly one step that a referee refuted on the unrepaired text: the
split of the Bochner integral of `w(X)·Y·X` into a signal term and a noise term.
The unrepaired proof asserted the noise term integrable "because ‖ε w(X) X‖ =
|ε|‖X‖^{1−θ}", which is an identity and not an integrability proof; two
independent counterexamples (a discrete law with `P(N=n)=7·8⁻ⁿ`, and a Pareto law
with `ε|X ~ U(−X², X²)`) satisfy every stated hypothesis and make `r_θ`
undefined.  The repair adds the joint moment as a hypothesis.

This file states that decomposition step with the repaired hypothesis list and
proves it from Mathlib's Bochner integral.  It is stated for a general real
inner-product space `E`, so the Hilbert-space case is an instance, and for a
general weight function `w : E → ℝ`, so the zero convention `w(0)·0 := 0` is a
choice of `w` and not a case split here.

WHAT IT DOES NOT CERTIFY, AND WHY THAT IS THE POINT.  It does not prove that the
joint-moment hypothesis is *necessary* — the counterexamples do that, and they are
reproduced in the audit rather than formalised.  It does not touch (MI) as a
conditional expectation: the step from `E[ε | X] = 0` to `E[ε w(X) X] = 0` is
conditional Fubini, and this file takes the resulting centring as a hypothesis
`hcentred`, exactly as the paper's proof takes it after establishing
integrability.  The Bochner integral in Mathlib is *defined* to be zero off
`Integrable`, so a formalisation that dropped `hnoise` would not fail — it would
silently state something about the value 0.  That is why the hypothesis is
carried explicitly and why `#print axioms` at the end is part of the certificate.

The kernel's earlier six lemmas certified finite-dimensional algebra that was
never in doubt while `Lemma 3.3` of paper 1 was false; this lemma is the one
whose failure the audit actually found, and it is the first measure-theoretic
statement in the kernel.
-/
import Mathlib

open MeasureTheory

namespace KuPLS

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

omit [MeasurableSpace Ω] in
/-- The weighted response-times-predictor term `w(X) · Y · X` with
`Y = ⟪β, X⟫ + ε` splits pointwise into a signal term and a noise term.
Pure algebra; no integrability is involved. -/
theorem weighted_split (w : E → ℝ) (X : Ω → E) (ε : Ω → ℝ) (β : E) :
    (fun ω => (w (X ω) * (inner ℝ β (X ω) + ε ω)) • X ω)
      = fun ω => (w (X ω) * inner ℝ β (X ω)) • X ω + (w (X ω) * ε ω) • X ω := by
  funext ω
  rw [mul_add, add_smul]

/-- **The decomposition step of `lem:exact`, repaired.**

Hypotheses, in the order the lemma states them after repair:
* `hsignal` — the signal term is Bochner integrable; in the paper this follows
  from `E‖X‖^{2−θ} < ∞` via `‖w(X)⟪β,X⟫X‖ ≤ ‖β‖·‖X‖^{2−θ}`;
* `hnoise`  — the noise term is Bochner integrable; in the paper this is
  **the added hypothesis** `E[|ε|·‖X‖^{1−θ}] < ∞`, since
  `‖w(X) ε X‖ = |ε|·‖X‖^{1−θ}`;
* `hcentred` — the noise term has mean zero; in the paper this is (MI) through
  conditional Fubini, which is legitimate only once `hnoise` holds.

Conclusion: `r_θ = E[w(X) Y X]` exists as the integral of an integrable function
and equals `E[w(X)⟪β,X⟫X]`, which is `A_θ β`. -/
theorem exactness_decomposition
    (w : E → ℝ) (X : Ω → E) (ε : Ω → ℝ) (β : E)
    (hsignal : Integrable (fun ω => (w (X ω) * inner ℝ β (X ω)) • X ω) μ)
    (hnoise : Integrable (fun ω => (w (X ω) * ε ω) • X ω) μ)
    (hcentred : ∫ ω, (w (X ω) * ε ω) • X ω ∂μ = 0) :
    Integrable (fun ω => (w (X ω) * (inner ℝ β (X ω) + ε ω)) • X ω) μ ∧
    ∫ ω, (w (X ω) * (inner ℝ β (X ω) + ε ω)) • X ω ∂μ
      = ∫ ω, (w (X ω) * inner ℝ β (X ω)) • X ω ∂μ := by
  rw [weighted_split]
  refine ⟨hsignal.add hnoise, ?_⟩
  rw [integral_add hsignal hnoise, hcentred, add_zero]

/-- **Negative control, stated as a theorem about Mathlib rather than about the
paper.**  Off `Integrable` the Bochner integral is zero by definition.  This is
why `hnoise` cannot be omitted from `exactness_decomposition` and quietly
recovered: without it the "integral" of the noise term is a junk value, and an
identity involving it proves nothing about `r_θ`. -/
theorem integral_is_junk_off_integrable
    {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
    (f : Ω → F) (hf : ¬ Integrable f μ) : ∫ ω, f ω ∂μ = 0 :=
  integral_undef hf

end KuPLS

#print axioms KuPLS.exactness_decomposition
#print axioms KuPLS.weighted_split
