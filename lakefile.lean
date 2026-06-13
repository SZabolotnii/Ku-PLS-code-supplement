import Lake
open Lake DSL

package kupls

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.26.0"

lean_lib KuPLS where
  roots := #[`KuPLS]
