#!/usr/bin/env bash
# Run the frozen confirmation analyses (GRANULARITY_CONFIRMATION.md, tag
# granularity-confirmation-plan-v1) on the GPU host's CPUs.
#
# Usage: run-confirmation-analysis.sh RUN_DIR
# RUN_DIR holds prepared/, parser-verified/, parser-audit.json, sample.json and
# edu-scores/. CODE must be an unmodified `git archive` of the frozen tag.
set -euo pipefail

run="$(cd "$1" && pwd)"
code="${CODE:-/scratch/threads-of-subtlety/sentiment4-7199693/code}"
python="${PYTHON:-/scratch/threads-of-subtlety/.venv/bin/python}"
out="$run/confirm"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0

cd "$code"
trap 'status=$?; printf "%s\n" "$status" > "$run/analysis-exit-status"' EXIT
date -u
"$python" -c "import sys, numpy, scipy, sklearn; print(sys.version.split()[0], numpy.__version__, scipy.__version__, sklearn.__version__)"
sha256sum research/sentiment/confirm_granularity.py research/sentiment/granularity_diagnostics.py \
    research/sentiment/relation_gate.py research/sentiment/GRANULARITY_CONFIRMATION.md
mkdir "$out"
cohort=(--corpus cohort --prepared "$run/prepared" --trees "$run/parser-verified"
        --scores "$run/edu-scores/scores.jsonl" --audit "$run/parser-audit.json"
        --scores-manifest "$run/edu-scores/manifest.json" --sample "$run/sample.json")

"$python" -B -u -m research.sentiment.confirm_granularity primary "${cohort[@]}" \
    --output "$out/primary" > "$out/primary.log" 2>&1

# The remaining analyses are independent of each other. Scramble must follow
# primary on this host because it reproduces primary's predictions.
"$python" -B -u -m research.sentiment.granularity_diagnostics scramble "${cohort[@]}" --permutations 200 \
    --reference-report "$out/primary/report.json" --reference-predictions "$out/primary/predictions.jsonl" \
    --output "$out/scramble" > "$out/scramble.log" 2>&1 &
scramble=$!
"$python" -B -u -m research.sentiment.confirm_granularity curve "${cohort[@]}" \
    --output "$out/curve" > "$out/curve.log" 2>&1 &
curve=$!
"$python" -B -u -m research.sentiment.confirm_granularity others "${cohort[@]}" \
    --baseline-config research/sentiment/replication/results/amazon/baseline-config.json \
    --output "$out/others" > "$out/others.log" 2>&1 &
others=$!
status=0
for pid in "$scramble" "$curve" "$others"; do
    wait "$pid" || status=1
done
date -u
exit "$status"
