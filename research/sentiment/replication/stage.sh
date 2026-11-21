#!/usr/bin/env bash
set -euo pipefail

scripts="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(git -C "$scripts" rev-parse --show-toplevel)"
run="$(cd "$1" && pwd)"
corpus="$2"
case "$corpus" in
  amazon|uci) ;;
  *) echo "Choose amazon or uci" >&2; exit 2 ;;
esac

source="$run/source"
mkdir "$source"
git -C "$repo" archive 25499ace90e23203da204abb92c32fb0ab8de41b research/sentiment tos |
  tar -x -C "$source"
patch --batch -p1 -d "$source" -i "$scripts/amazon.patch"
cp "$scripts/prepare_$corpus.py" "$source/research/sentiment/"
if [[ "$corpus" == uci ]]; then
  patch --batch -p1 -d "$source" -i "$scripts/uci.patch"
fi
