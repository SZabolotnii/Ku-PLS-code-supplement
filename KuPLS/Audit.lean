import KuPLS.Core
import KuPLS.Krylov
import KuPLS.Exactness

/-!
# Axiom audit

`#print axioms` for every certified theorem.  Expected output: only the standard
Lean/Mathlib axioms (`propext`, `Classical.choice`, `Quot.sound`) — no `sorryAx`,
no custom axioms.  These are the *deterministic, moment-free* facts underlying
the CF-functional-PLS gate.
-/

open CFPLS

-- Moment-free Gram PSD (the crux that breaks the 4th-moment dependency)
#print axioms ecfGram_isHermitian
#print axioms ecfQuadForm_eq_normSq
#print axioms ecfGram_posSemidef

-- CG / Krylov transfer (needs only self-adjoint PSD / inner-product structure)
#print axioms normalSystem_iff_starProjection
#print axioms energy_eq_dotProduct
#print axioms krylov_residual_orthogonal

-- Paper 2, lem:exact as repaired 2026-09-07: the FIRST measure-theoretic
-- statement in this kernel.  The decomposition r_theta = A_theta beta + E[noise]
-- with the noise term Bochner-integrable BY HYPOTHESIS -- the hypothesis whose
-- absence two independent referee arms refuted on the unrepaired text.
#print axioms KuPLS.exactness_decomposition
#print axioms KuPLS.weighted_split
#print axioms KuPLS.integral_is_junk_off_integrable
