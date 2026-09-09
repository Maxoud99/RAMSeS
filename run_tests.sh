#!/usr/bin/env bash
# Run the test suite: every module in its OWN process.
#
#     ./run_tests.sh                      # everything
#     ./run_tests.sh Utils WebUI          # only these paths
#
# The one-process-per-module rule is load-bearing, not tidiness.
# Model_Selection/test_thompson_sampling.py and test_rank_aggregation.py
# install fake "Metrics", "Metrics.Ensemble_GA" and "Metrics.metrics" entries
# into sys.modules at import time, so they can exercise the module under test
# without dragging in the pipeline. Any test sharing that interpreter
# afterwards receives the stubs instead of the real package — which is what
# made test_reward_domain report "Metrics is not a package" when the suite was
# run in one pytest process.
#
# PYTHONPATH carries the repo root PLUS each file's own directory, because the
# suite mixes dotted imports (`from Metrics.metrics import ...`) with bare ones
# (`from Thompson_Sampling import ...`).
#
# pytest is a test-only dependency and is deliberately absent from
# requirements.txt, which is the runtime install. Install it into the env:
#     python -m pip install pytest
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$PWD

PY=${RAMSES_PYTHON:-/raid0_ssd2/maxoud/condaenv/RAMS/bin/python}
if ! "$PY" -c "import pytest" 2>/dev/null; then
  echo "pytest not installed in $PY — run: $PY -m pip install pytest" >&2
  exit 2
fi

# This box has no display; matplotlib's Qt backend aborts with
# "Could not find the Qt platform plugin xcb" without this.
export MPLBACKEND=Agg

TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(.)

pass=0; fail=0; failed=()
while IFS= read -r f; do
  d=$(dirname "$f")
  out=$(PYTHONPATH="$ROOT:$ROOT/$d" timeout 900 "$PY" -m pytest "$f" \
          -q --no-header -p no:cacheprovider 2>&1)
  summary=$(printf '%s\n' "$out" | tail -1)
  if printf '%s\n' "$out" | grep -qE "^(FAILED|ERROR)"; then
    fail=$((fail+1)); failed+=("$f")
    printf "  FAIL  %-58s %s\n" "$f" "$summary"
  else
    pass=$((pass+1))
    printf "  ok    %-58s %s\n" "$f" "$summary"
  fi
done < <(find "${TARGETS[@]}" -name 'test_*.py' \
           -not -path './Mononito/*' -not -path './archive/*' \
           -not -path './Misc/*' | sort)

echo
echo "modules passing: $pass   modules with failures: $fail"
for f in "${failed[@]}"; do echo "   $f"; done
[ "$fail" -eq 0 ]
