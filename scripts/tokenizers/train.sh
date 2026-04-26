#!/usr/bin/env bash
set -euo pipefail

TRAINER='./scripts/tokenizers/trainer.py'
LINES=${n:-500000}

usage() {
  echo "Usage: $0 LANG SPLIT"
  echo '--'
  echo 'LANG options: arb, pes, spa, ...'
  echo 'SPLIT can be either LANG or eng'
  echo '--'
  echo "Example:"
  echo "n=100000 $0 kor eng"

}

LANG="${1:-}"
SPLIT="${2:-}"
[[ -z "$LANG" || -z "$SPLIT" ]] && usage && exit 1
shift 2

prefix="corpus/high_resource/$LANG-eng/corpus.$LANG-eng"
file="$prefix.$SPLIT"

head -n "$LINES" "$file" | python "$TRAINER" "$@"
