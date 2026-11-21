#!/usr/bin/env bash
set -euo pipefail

run="$(cd "$1" && pwd)"
assets="$(cd "$2" && pwd)"
repo="$(git -C "$run" rev-parse --show-toplevel)"
checkpoints="${UNIRST_CHECKPOINTS:-$repo/HC3_project}"
device="${CUDA_DEVICE:-0}"
cd "$run/source"
parser="$assets/parser-venv/bin/python"
export HF_HOME="$assets/hf" HF_HUB_CACHE="$assets/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NVIDIA_TF32_OVERRIDE=0

first="$run/parser-first-pass"
verified="$run/parser-verified"

parse() {
  local inventory="$1" output="$2"
  shift 2
  "$parser" -B -u -m research.sentiment.parse_sfu run \
    --sample "$run/sample.json" --output "$output" \
    --checkpoints "$checkpoints" --stock-snapshot "$assets/stock" \
    --inventory "$inventory" --cuda-device "$device" "$@"
}

if [[ "${3:-}" != --retry-existing && "$device" == -1 ]]; then
  for inventory in deu.rst.pcc nld.rst.nldt eng.erst.gum eng.rst.rstdt; do
    parse "$inventory" "$first"
  done
elif [[ "${3:-}" != --retry-existing ]]; then
  # PCC writes the shared segmentation. Then keep at most two parser
  # processes resident on the GPU.
  parse deu.rst.pcc "$first" &
  pcc=$!
  while [[ ! -s "$first/segmentation.json" ]]; do
    if ! kill -0 "$pcc" 2>/dev/null; then
      wait "$pcc"
      exit 1
    fi
    sleep 2
  done
  parse nld.rst.nldt "$first" &
  nldt=$!
  wait "$pcc"
  parse eng.erst.gum "$first" &
  gum=$!
  wait "$nldt"
  parse eng.rst.rstdt "$first" &
  rstdt=$!
  wait "$gum"
  wait "$rstdt"
fi

mkdir "$verified"
cp "$first/scenes.jsonl" "$first/segmentation.json" "$verified/"
for inventory in deu.rst.pcc nld.rst.nldt eng.erst.gum eng.rst.rstdt; do
  parse "$inventory" "$verified" --retry-from "$first"
done

"$parser" -B -m research.sentiment.parse_sfu verify \
  --sample "$run/sample.json" --output "$verified" \
  --report "$run/parser-audit.json"
