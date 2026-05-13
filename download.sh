#!/usr/bin/env bash
set -euo pipefail

# SuperKittens model downloader. Registry-based: add new models by editing the
# `resolve` function below.
#
# Usage:
#   ./download.sh                       # list known models
#   ./download.sh gemma4-e2b            # download one
#   ./download.sh gemma4-e2b qwen3-0.6b # download multiple
#   ./download.sh --list                # explicit list

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_ROOT="$ROOT/SuperKittens/model_weights"

resolve() {
    case "$1" in
        gemma4-e2b)         echo "google/gemma-4-E2B-it     gemma-4-E2B-it" ;;
        gemma4-e4b)         echo "google/gemma-4-E4B-it     gemma-4-E4B-it" ;;
        gemma4-26b)         echo "google/gemma-4-26B-it     gemma-4-26B-it" ;;
        gemma4-31b)         echo "google/gemma-4-31B-it     gemma-4-31B-it" ;;
        qwen3-0.6b)         echo "Qwen/Qwen3-0.6B           Qwen3-0.6B" ;;
        qwen3-1.7b)         echo "Qwen/Qwen3-1.7B           Qwen3-1.7B" ;;
        qwen3-8b)           echo "Qwen/Qwen3-8B             Qwen3-8B" ;;
        qwen3-32b)          echo "Qwen/Qwen3-32B            Qwen3-32B" ;;
        deepseek-v3)        echo "deepseek-ai/DeepSeek-V3   DeepSeek-V3" ;;
        deepseek-v2-lite)   echo "deepseek-ai/DeepSeek-V2-Lite  DeepSeek-V2-Lite" ;;
        mamba2-130m)        echo "AntonV/mamba2-130m-hf      mamba2-130m-hf" ;;
        mamba2-2.7b)        echo "AntonV/mamba2-2.7b-hf      mamba2-2.7b-hf" ;;
        *)                  echo "" ;;
    esac
}

KNOWN="gemma4-e2b gemma4-e4b gemma4-26b gemma4-31b qwen3-0.6b qwen3-1.7b qwen3-8b qwen3-32b deepseek-v3 deepseek-v2-lite mamba2-130m mamba2-2.7b"

color() { printf "\033[1;36m%s\033[0m\n" "$*"; }
warn()  { printf "\033[1;33m%s\033[0m\n" "$*"; }
fail()  { printf "\033[1;31m%s\033[0m\n" "$*" >&2; exit 1; }

list_models() {
    echo "available models:"
    for spec in $KNOWN; do
        info=$(resolve "$spec")
        repo=$(echo "$info" | awk '{print $1}')
        printf "  %-22s  %s\n" "$spec" "$repo"
    done
}

if [[ $# -eq 0 ]] || [[ "${1:-}" == "--list" ]] || [[ "${1:-}" == "-l" ]]; then
    list_models
    exit 0
fi

if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
    fail "neither 'hf' nor 'huggingface-cli' is installed. run: pip install -U 'huggingface_hub[cli]'"
fi

for spec in "$@"; do
    info=$(resolve "$spec")
    if [[ -z "$info" ]]; then
        warn "unknown spec: $spec  (run './download.sh --list' to see options)"
        continue
    fi
    repo=$(echo "$info" | awk '{print $1}')
    name=$(echo "$info" | awk '{print $2}')
    dest="$DEST_ROOT/$name"
    mkdir -p "$dest"
    color "↓ $spec  ($repo)  →  $dest"
    if command -v hf >/dev/null 2>&1; then
        hf download "$repo" --local-dir "$dest"
    else
        huggingface-cli download "$repo" --local-dir "$dest" --local-dir-use-symlinks False
    fi
done
echo "done."
