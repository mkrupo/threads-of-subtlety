#!/usr/bin/env bash
# Fine-tune the roberta-base baseline on all five outer folds of a cohort, using
# both GTX 1080 Ti cards, then merge the held-out predictions.
#
# Usage: run-roberta-baseline-1080ti.sh RUN_DIR
# RUN_DIR holds code/ (a git archive of the committed baseline) and output/.
# COHORT points at the cohort folder with prepared/ (reviews, manifest, splits,
# provenance). MODEL points at the pinned roberta-base snapshot.
set -euo pipefail

run="$(cd "$1" && pwd)"
cohort="${COHORT:-/scratch/threads-of-subtlety/amazon23-confirm-v1.tFl3lr}"
model="${MODEL:-/data/huggingface/hub/models--FacebookAI--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b}"
python="${PYTHON:-/scratch/threads-of-subtlety/.venv-finetune/bin/python}"
export HF_HOME=/data/huggingface HF_HUB_CACHE=/data/huggingface/hub HUGGINGFACE_HUB_CACHE=/data/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0

cd "$run/code"
trap 'status=$?; printf "%s\n" "$status" > "$run/exit-status"' EXIT
date -u
nvidia-smi --query-gpu=index,uuid,memory.free --format=csv
"$python" -m pip freeze 2>/dev/null > "$run/pip-freeze.txt" || "$python" -c "import importlib.metadata as m; print('\n'.join(sorted(f'{d.metadata[\"Name\"]}=={d.version}' for d in m.distributions())))" > "$run/pip-freeze.txt"
sha256sum research/sentiment/finetune_roberta.py

fold() {
    "$python" -B -u -m research.sentiment.finetune_roberta train --fold "$1" --device cuda:0 \
        --prepared "$cohort/prepared" --model-dir "$model" --output "$run/output"
}
( trap - EXIT; export CUDA_VISIBLE_DEVICES=0; fold 0; fold 2; fold 4 ) > "$run/gpu0.log" 2>&1 &
left=$!
( trap - EXIT; export CUDA_VISIBLE_DEVICES=1; fold 1; fold 3 ) > "$run/gpu1.log" 2>&1 &
right=$!
status=0
wait "$left" || status=1
wait "$right" || status=1
(( status == 0 )) || exit 1
"$python" -B -m research.sentiment.finetune_roberta merge --prepared "$cohort/prepared" --output "$run/output"
date -u
