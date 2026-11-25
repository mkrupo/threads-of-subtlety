#!/usr/bin/env bash
# Segment and parse a prepared cohort on the 2x GTX 1080 Ti host with the parser
# code, checkpoints and environment of the verified Amazon run. This follows the
# launcher used for amazon23-short-v1: PCC segments first on GPU 0, the other
# inventories start when the shared segmentation exists, then every inventory
# runs the converter retry pass and the verifier checks all trees.
#
# Usage: run-cohort-parser-1080ti.sh RUN_DIR
# RUN_DIR must contain sample.json and source/ (the staged Amazon parser source).
set -euo pipefail

run="$(cd "$1" && pwd)"
project=/scratch/threads-of-subtlety
pilot="$project/parser-pilot"
python="$pilot/.venv/bin/python"
first="$run/parser-first-pass"
verified="$run/parser-verified"
stock="$pilot/hub/models--tchewik--isanlp_rst_v3/snapshots/b99b1c0f39435971b9daaa450e8476ef7b1a554a"
export PYTHONPATH="$run/source:$pilot/site:/scratch/rst_similarity/.venv/lib/python3.10/site-packages"
export HF_HOME="$pilot/hf-home" HF_HUB_CACHE="$pilot/hub" HUGGINGFACE_HUB_CACHE="$pilot/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NVIDIA_TF32_OVERRIDE=0
export HF_DATASETS_CACHE=/data/huggingface/datasets UV_CACHE_DIR=/data/uv-cache XDG_CACHE_HOME=/data/cache
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

cd "$run/source"
trap 'status=$?; printf "%s\n" "$status" > "$run/exit-status"' EXIT
date -u
nvidia-smi --query-gpu=index,uuid,memory.free --format=csv
sha256sum "$run/sample.json"

parse() {
    local inventory="$1" output="$2"
    shift 2
    "$python" -B -u -m research.sentiment.parse_sfu run \
        --sample "$run/sample.json" --output "$output" --checkpoints "$project/checkpoints" \
        --stock-snapshot "$stock" --inventory "$inventory" --cuda-device 0 "$@"
}
wait_for_workers() {
    local left_status=0 right_status=0
    wait "$1" || left_status=$?
    wait "$2" || right_status=$?
    if (( left_status || right_status )); then
        printf "GPU worker exits: %s %s\n" "$left_status" "$right_status"
        return 1
    fi
}

mkdir "$first"
( trap - EXIT; export CUDA_VISIBLE_DEVICES=0
  parse deu.rst.pcc "$first"; parse eng.erst.gum "$first" ) > "$run/first-gpu0.log" 2>&1 &
left=$!
( trap - EXIT; export CUDA_VISIBLE_DEVICES=1
  until [[ -s "$first/segmentation.json" ]]; do
      kill -0 "$left" 2>/dev/null || exit 1
      sleep 2
  done
  parse nld.rst.nldt "$first"; parse eng.rst.rstdt "$first" ) > "$run/first-gpu1.log" 2>&1 &
right=$!
wait_for_workers "$left" "$right"

mkdir "$verified"
cp "$first/scenes.jsonl" "$first/segmentation.json" "$verified/"
( trap - EXIT; export CUDA_VISIBLE_DEVICES=0
  parse deu.rst.pcc "$verified" --retry-from "$first"
  parse eng.erst.gum "$verified" --retry-from "$first" ) > "$run/verified-gpu0.log" 2>&1 &
left=$!
( trap - EXIT; export CUDA_VISIBLE_DEVICES=1
  parse nld.rst.nldt "$verified" --retry-from "$first"
  parse eng.rst.rstdt "$verified" --retry-from "$first" ) > "$run/verified-gpu1.log" 2>&1 &
right=$!
wait_for_workers "$left" "$right"

"$python" -B -m research.sentiment.parse_sfu verify \
    --sample "$run/sample.json" --output "$verified" --report "$run/parser-audit.json"
date -u
