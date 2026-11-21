#!/usr/bin/env bash
set -euo pipefail

run="$(cd "$1" && pwd)"
assets="$(cd "$2" && pwd)"
cd "$run/source"
scorer="$assets/scorer-venv/bin/python"
classifier="$assets/classifier-exact-venv/bin/python"
local_results="research/sentiment/results"
prepared="$run/prepared"
trees="$run/parser-verified"
scores="$run/edu-scores/scores.jsonl"
uniform="$run/edu-mean"
nuclearity="$run/nuclearity"
text_order="$run/relation-text-order"

"$classifier" -B - "$run/parser-audit.json" <<'PY'
import json
from pathlib import Path
import sys
data = json.loads(Path(sys.argv[1]).read_text())
if data["complete_documents_across_inventories"] != data["documents"]:
    raise ValueError("The parser audit does not cover every review in all four inventories")
if any(info["unmapped_observed"] for info in data["inventory_reports"].values()):
    raise ValueError("A predicted label is missing from the frozen coarse maps")
PY

# These report filenames are the original SFU runner's local provenance
# interface. Bind them to this run's generated artifacts in the copied code.
cp "$run/parser-audit.json" "$local_results/sfu-full-parser-audit.json"
"$classifier" -B - "$run/text-controls/run_manifest.json" "$local_results/sfu-text-controls-v1.json" <<'PY'
import json
from pathlib import Path
import sys
source, output = map(Path, sys.argv[1:])
manifest = json.loads(source.read_text())
output.write_text(json.dumps({"provenance": manifest}, indent=2) + "\n", encoding="utf-8")
PY
"$scorer" -B -u -m research.sentiment.score_sfu \
  --scenes "$trees/scenes.jsonl" --model-dir "$assets/cardiff" --output "$run/edu-scores"
cp "$run/edu-scores/manifest.json" "$local_results/sfu-edu-scores-v1.json"
"$classifier" -B -u -m research.sentiment.edu_baseline \
  --prepared-dir "$prepared" --scores "$scores" --output-dir "$uniform"
cp "$uniform/report.json" "$local_results/sfu-edu-mean-v1.json"
"$classifier" -B -u -m research.sentiment.nuclearity \
  --prepared-dir "$prepared" --scores "$scores" --trees-dir "$trees" \
  --uniform-predictions "$uniform/predictions.jsonl" --output-dir "$nuclearity"
cp "$nuclearity/report.json" "$local_results/sfu-nuclearity-v1.json"

common_args=(--prepared-dir "$prepared" --scores "$scores" --trees-dir "$trees"
  --uniform-predictions "$uniform/predictions.jsonl" --nuclearity-predictions "$nuclearity/predictions.jsonl")
"$classifier" -B -u -m research.sentiment.relation_pooling \
  "${common_args[@]}" --output-dir "$text_order"
cp "$text_order/report.json" "$local_results/sfu-relation-pooling-v1.json"
"$classifier" -B -u -m research.sentiment.relation_stability \
  --predictions "$text_order/predictions.jsonl" --manifest "$prepared/manifest.jsonl" \
  --output "$text_order/stability.json"
"$classifier" -B -u -m research.sentiment.relation_pooling \
  "${common_args[@]}" --nucleus-first "$text_order/predictions.jsonl" \
  --output-dir "$run/relation-nucleus-first"
"$classifier" -B -u -m research.sentiment.motif_sfu \
  --prepared-dir "$prepared" --trees-dir "$trees" --output-dir "$run/motifs"
