#!/usr/bin/env bash
# Re-run every script of the package, in dependency order, writing each one's
# stdout to the results/*.out file this repository ships and letting the script
# write its own results/*.json.  `git diff results/` is then the whole answer to
# "does it still produce the paper's numbers?".
#
# A few scripts write an artifact whose name is not the script's own -- the
# third field of each line below is the artifact stem, so the diff lands on the
# shipped file rather than beside it.
#
# The Monte Carlo scripts dominate the cost: several run 400 to 3000
# replications of a functional CG fit, and the calibration gates run 2000 to
# 3000.  Each script's docstring states its own design, so a single number can
# be re-checked without running the set.  Expect the full run to take hours on
# one core.
#
# Usage:  ./run_all.sh            everything
#         ./run_all.sh sim        only the simulation scripts
#         ./run_all.sh empirical  only the scripts that read data/ (or FRED)
set -u
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
mkdir -p results

# dir script artifact-stem
SIM="
code   prop5_counterexample        prop5_counterexample
code   pilot_offset                pilot_offset
code   section43_hypotheses        section43_hypotheses
code   split_stopping_rule         split_stopping_rule
code   split_rule_tables           split_rule_tables
code   shrink_table                shrink_table
code   run_pilot                   a2_pilot_estimation
code   run_size_confirm            a2_size_confirm
code   run_inference_pilot         a2_pilot_inference
code   run_dependence              dependence
code   run_theta_selector          theta_selector
gates  a0_1_bounded_test_function  a0_1_bounded_test_function
gates  a0_3_bounded_feature_family a0_3_bounded_feature_family
gates  a0_3b_response_side         a0_3b_response_side
gates  a1_5_consistency            a1_5_consistency
gates  a1_6_discrepancy            a1_6_discrepancy
gates  a1_6b_source_condition      a1_6b_source_condition
gates  a1_7_power                  a1_7_power
gates  a1_7_power_confirm          a1_7_power_confirm
gates  a1_7_residual_identity      a1_7_residual_identity
gates  a1_7_verify                 a1_7_verify
gates  a3_cond_exponents           a3_cond_exponents
gates  a3_dj_bounds                a3_dj_bounds
gates  a3_kappa_phi_sensitivity    a3_kappa_phi_sensitivity
"

# these read data/*.npz, and re-pull from FRED if a cache is missing
EMP="
code   empirical2_cp               empirical2_cp
code   probe_heavy_curves          probe_heavy_curves
code   empirical2_blockboot        empirical2_blockboot
code   theta_selector_apply        theta_selector_apply
code   check_selected_theta        check_selected_theta
code   selector_on_applications    selector_on_applications
gates  a0_4_tail_index             a0_4_tail_index
gates  a1_8_mixing                 a1_8_mixing
gates  a1_8b_mds_boundary          a1_8b_mds_boundary
gates  a3_persistence_artifact     a3_persistence_artifact
gates  a3_kappa_margin             a3_kappa_margin
gates  a3_kappa_recalibrate        a3_kappa_recalibrate
gates  a5_persistence_target       a5_persistence_target
gates  b7_blocklength              b7_blocklength
"

FAILED=""

run_set() {
  local dir name stem t0 rc
  while read -r dir name stem; do
    [ -z "${dir:-}" ] && continue
    printf '%-46s' "$dir/$name.py"
    t0=$SECONDS
    if "$PY" "$dir/$name.py" > "results/$stem.out" 2>&1; then rc=0; else rc=$?; fi
    if [ $rc -eq 0 ]; then
      printf 'ok   %5ds\n' $((SECONDS - t0))
    else
      printf 'FAIL %5ds  (exit %d, see results/%s.out)\n' $((SECONDS - t0)) "$rc" "$stem"
      FAILED="$FAILED $dir/$name"
    fi
  done
}

case "${1:-all}" in
  all)       printf '%s' "$SIM" | run_set; printf '%s' "$EMP" | run_set ;;
  sim)       printf '%s' "$SIM" | run_set ;;
  empirical) printf '%s' "$EMP" | run_set ;;
  *) echo "usage: $0 [all|sim|empirical]" >&2; exit 2 ;;
esac

echo
if [ -n "$FAILED" ]; then
  echo "FAILED:$FAILED"
  exit 1
fi
echo "all scripts completed -- now run:  git diff --stat results/"
