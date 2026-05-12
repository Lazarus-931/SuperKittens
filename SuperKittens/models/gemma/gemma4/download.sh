#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-e4b}"
case "$VARIANT" in
    e2b) REPO="google/gemma-4-E2B-it";  NAME="gemma-4-E2B-it" ;;
    e4b) REPO="google/gemma-4-E4B-it";  NAME="gemma-4-E4B-it" ;;
    26b) REPO="google/gemma-4-26B-it";  NAME="gemma-4-26B-it" ;;
    31b) REPO="google/gemma-4-31B-it";  NAME="gemma-4-31B-it" ;;
    *) echo "unknown variant: $VARIANT (use e2b|e4b|26b|31b)"; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
DEST="$ROOT/SuperKittens/model_weights/$NAME"
mkdir -p "$DEST"

if command -v hf >/dev/null 2>&1; then
    hf download "$REPO" --local-dir "$DEST"
elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$REPO" --local-dir "$DEST" --local-dir-use-symlinks False
else
    echo "neither 'hf' nor 'huggingface-cli' found. install: pip install -U 'huggingface_hub[cli]'"
    exit 1
fi
echo "downloaded → $DEST"
