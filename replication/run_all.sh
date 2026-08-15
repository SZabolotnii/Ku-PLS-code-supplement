#!/usr/bin/env bash
# Reproduce every number in the paper's Tables 1 and 2, from scratch.
set -euo pipefail
mkdir -p results data
echo "== Monte Carlo (Section 10) — about 10 minutes =="
python -m kupls.simulate "${1:-1000}"
echo
echo "== Empirical application (Section 11) — pulls FRED over HTTPS =="
python -m kupls.empirical
echo
echo "== LaTeX tables =="
python -m kupls.tables
