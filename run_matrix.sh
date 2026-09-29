#!/bin/sh
# Phase 2 experiment matrix.
#
#   sh run_matrix.sh                 # full matrix (4000 steps)
#   sh run_matrix.sh 800             # a fast, small version of the same matrix
#
# Ordering. The three mandatory arms are run across all six seeds first, then
# the two stretch arms across three seeds. Six is the smallest number of seeds
# for which the exact paired sign-flip test can attain p < 0.05 (2/2^6 = 0.031),
# so seed count on the mandatory comparison is prioritised over arm count. If
# the matrix is interrupted, what survives is the well-powered core experiment.
#
# Runs are executed strictly one at a time: wall-clock is a reported metric, so
# two runs must never share the GPU.
#
# Every run writes runs/<arm>_seed<seed>_s<steps>/ and is skipped if that
# directory already holds a summary.json, so the matrix is resumable.

set -u
cd "$(dirname "$0")"

# Interpreter: $PYTHON if set, else the project venv (Windows or POSIX
# layout), else whatever `python` is on PATH.
if [ -n "${PYTHON:-}" ]; then PY=$PYTHON
elif [ -x ./.venv/Scripts/python.exe ]; then PY=./.venv/Scripts/python.exe
elif [ -x ./.venv/bin/python ]; then PY=./.venv/bin/python
else PY=python; fi
STEPS=${1:-4000}
MANDATORY_SEEDS="0 1 2 3 4 5"
STRETCH_SEEDS="0 1 2"
MANDATORY="baseline attn-first attn-second"
STRETCH="alternating attn-freeze"
# Added in the pre-submission audit: attn-first/attn-second match the
# step-averaged LR multiplier, not the integrated LR, so they also differ in
# per-group LR mass (see schedules.py). These LR-mass-matched versions separate
# timing from budget; six seeds, same as the arms they correct.
CORRECTION="attn-first-lrm attn-second-lrm"

export PYTHONWARNINGS=ignore
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
mkdir -p logs runs

# Single-instance lock. Two matrices sharing the GPU would both slow down and
# silently corrupt the wall-clock numbers this experiment reports, so a second
# instance refuses to start rather than quietly competing for the device.
LOCK=runs/.matrix.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "[matrix] refusing to start: $LOCK exists (another matrix is running)."
  echo "[matrix] if that is stale, remove it with: rm -rf $LOCK"
  exit 1
fi
trap 'rm -rf "$LOCK"' EXIT INT TERM

run_one() {
  arm=$1; seed=$2
  dir="runs/${arm}_seed${seed}_s${STEPS}"
  if [ -f "$dir/summary.json" ]; then
    echo "[matrix] skip $arm seed$seed (already complete)"
    return
  fi
  echo "[matrix] === $arm seed$seed ($STEPS steps) === $(date +%H:%M:%S)"
  $PY -u train.py --arm "$arm" --seed "$seed" --max-steps "$STEPS" \
      --eval-every 100 >> "logs/phase2_${arm}_seed${seed}.log" 2>&1
  echo "[matrix] exit=$? $arm seed$seed $(date +%H:%M:%S)"
}

# --- core experiment: mandatory arms, all seeds -----------------------------
for seed in $MANDATORY_SEEDS; do
  for arm in $MANDATORY; do
    run_one "$arm" "$seed"
  done
done

# --- stretch arms ----------------------------------------------------------
for seed in $STRETCH_SEEDS; do
  for arm in $STRETCH; do
    run_one "$arm" "$seed"
  done
done

# --- budget-corrected directed arms ----------------------------------------
for seed in $MANDATORY_SEEDS; do
  for arm in $CORRECTION; do
    run_one "$arm" "$seed"
  done
done

echo "[matrix] all done $(date +%H:%M:%S)"
