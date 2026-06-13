import KuPLS.Core
import KuPLS.Krylov

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
