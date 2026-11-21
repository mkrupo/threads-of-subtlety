#!/usr/bin/env bash
set -euo pipefail

scripts="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run="$(cd "$1" && pwd)"
assets="$(cd "$2" && pwd)"
archive="$(realpath "$3")"
classifier="$assets/classifier-exact-venv/bin/python"

bash "$scripts/stage.sh" "$run" sfu
cd "$run/source"
"$classifier" -B -m research.sentiment.prepare_corpus \
  --archive "$archive" --output-dir "$run/prepared" --seed 42
"$classifier" -B -m research.sentiment.parse_sfu prepare \
  --prepared "$run/prepared" --sample "$run/sample.json"
"$classifier" -B -m research.sentiment.baseline \
  --prepared-dir "$run/prepared" --output-dir "$run/text-controls-exact"
"$classifier" -B -m research.sentiment.verify_artifacts \
  --prepared-dir "$run/prepared" --run-dir "$run/text-controls-exact" \
  --archive "$archive" > "$run/text-controls-exact-independent.json"

bash "$scripts/run-parser.sh" "$run" "$assets"
bash "$scripts/run-classifiers.sh" "$run" "$assets" sfu
