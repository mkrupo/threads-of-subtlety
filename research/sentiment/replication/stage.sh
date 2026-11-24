#!/usr/bin/env bash
set -euo pipefail

scripts="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(git -C "$scripts" rev-parse --show-toplevel)"
run="$(cd "$1" && pwd)"
corpus="$2"
case "$corpus" in
  sfu|amazon|uci) ;;
  *) echo "Choose sfu, amazon, or uci" >&2; exit 2 ;;
esac

source="$run/source"
mkdir "$source"
git -C "$repo" archive 25499ace90e23203da204abb92c32fb0ab8de41b research/sentiment tos |
  tar -x -C "$source"
if [[ "$corpus" != sfu ]]; then
  patch --batch -p1 -d "$source" -i "$scripts/amazon.patch"
  cp "$scripts/prepare_$corpus.py" "$source/research/sentiment/"
fi
if [[ "$corpus" == uci ]]; then
  patch --batch -p1 -d "$source" -i "$scripts/uci.patch"
fi
