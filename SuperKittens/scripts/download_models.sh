#!/usr/bin/env bash
# SuperKittens model downloader — ONE script, ONE clear list.
#
# Mirrors what `sk <model>` does, as a standalone reference. Downloads land in
# weights/<weight_dir>/ (override with SK_WEIGHTS_ROOT) — exactly where the
# inference registry looks — so `api.load("<variant>")` resolves with no
# symlinks. Each model also gets its tokenizer/config pulled from the canonical
# HF repo (GGUF repos usually omit those).
#
# Usage:
#   scripts/download_models.sh                 # list models and exit
#   scripts/download_models.sh qwen3-4b-q4km   # download one (or several)
#   scripts/download_models.sh --all           # download everything
#
# Requires huggingface-cli (`pip install huggingface_hub`). Gated repos (Gemma)
# need `huggingface-cli login` first.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST_ROOT="${SK_WEIGHTS_ROOT:-$ROOT/weights}"

#  variant            | gguf source repo                          | file (or *)                                       | weight_dir
#  --- Qwen3 (first-party GGUF repos: <repo>-GGUF) -------------------------------------------------------------------------
read -r -d '' MODELS <<'EOF' || true
qwen3-0.6b           | Qwen/Qwen3-0.6B-GGUF                       | Qwen3-0.6B-Q8_0.gguf                              | Qwen3-0.6B
qwen3-1.7b           | Qwen/Qwen3-1.7B-GGUF                       | Qwen3-1.7B-Q8_0.gguf                              | Qwen3-1.7B-GGUF
qwen3-4b             | Qwen/Qwen3-4B-GGUF                         | Qwen3-4B-Q8_0.gguf                                | Qwen3-4B-GGUF
qwen3-4b-q4km        | Qwen/Qwen3-4B-GGUF                         | Qwen3-4B-Q4_K_M.gguf                              | Qwen3-4B-GGUF
qwen3-8b             | Qwen/Qwen3-8B-GGUF                         | Qwen3-8B-Q8_0.gguf                                | Qwen3-8B-GGUF
qwen3-8b-q4km        | Qwen/Qwen3-8B-GGUF                         | Qwen3-8B-Q4_K_M.gguf                              | Qwen3-8B-GGUF
qwen3-14b            | Qwen/Qwen3-14B-GGUF                        | Qwen3-14B-Q8_0.gguf                               | Qwen3-14B-GGUF
qwen3-14b-q4km       | Qwen/Qwen3-14B-GGUF                        | Qwen3-14B-Q4_K_M.gguf                             | Qwen3-14B-GGUF
qwen3-32b            | Qwen/Qwen3-32B-GGUF                        | Qwen3-32B-Q8_0.gguf                               | Qwen3-32B-GGUF
mistral-7b-v0.3      | bartowski/Mistral-7B-Instruct-v0.3-GGUF   | Mistral-7B-Instruct-v0.3-Q4_K_M.gguf              | Mistral-7B-Instruct-v0.3-GGUF
nemotron-nano-8b     | bartowski/nvidia_Llama-3.1-Nemotron-Nano-8B-v1-GGUF | nvidia_Llama-3.1-Nemotron-Nano-8B-v1-Q4_K_M.gguf | Llama-3.1-Nemotron-Nano-8B-v1-GGUF
gemma4-12b-unified   | google/gemma-4-12B-it-GGUF                 | gemma-4-12B-it-Q4_K_M.gguf                        | gemma-4-12B-it-GGUF
gemma4-e2b           | google/gemma-4-E2B-it                      | *                                                 | gemma-4-E2B-it
gemma4-e4b           | google/gemma-4-E4B-it                      | *                                                 | gemma-4-E4B-it
mamba2-130m          | AntonV/mamba2-130m-hf                      | *                                                 | mamba2-130m-hf
deepseek-v2-lite     | deepseek-ai/DeepSeek-V2-Lite               | *                                                 | DeepSeek-V2-Lite
EOF

# canonical repo (for tokenizer/config) keyed by variant
declare -A CANON=(
  [qwen3-0.6b]=Qwen/Qwen3-0.6B [qwen3-1.7b]=Qwen/Qwen3-1.7B [qwen3-4b]=Qwen/Qwen3-4B
  [qwen3-4b-q4km]=Qwen/Qwen3-4B [qwen3-8b]=Qwen/Qwen3-8B [qwen3-8b-q4km]=Qwen/Qwen3-8B
  [qwen3-14b]=Qwen/Qwen3-14B [qwen3-14b-q4km]=Qwen/Qwen3-14B [qwen3-32b]=Qwen/Qwen3-32B
  [mistral-7b-v0.3]=mistralai/Mistral-7B-Instruct-v0.3
  [nemotron-nano-8b]=nvidia/Llama-3.1-Nemotron-Nano-8B-v1
  [gemma4-12b-unified]=google/gemma-4-12B-it )

if [[ $# -eq 0 ]]; then
  echo "Available models (pass one or more, or --all):"
  echo "$MODELS" | awk -F'|' 'NF{printf "  %-20s %s\n",$1,$2}'
  exit 0
fi

command -v huggingface-cli >/dev/null 2>&1 || { echo "ERROR: pip install huggingface_hub" >&2; exit 1; }

WANT=("$@"); ALL=0; [[ "${1:-}" == "--all" ]] && ALL=1
want(){ [[ $ALL -eq 1 ]] && return 0; for w in "${WANT[@]}"; do [[ "$w" == "$1" ]] && return 0; done; return 1; }

echo "$MODELS" | while IFS='|' read -r v repo file dir; do
  v="$(echo "$v"|xargs)"; [[ -z "$v" ]] && continue
  repo="$(echo "$repo"|xargs)"; file="$(echo "$file"|xargs)"; dir="$(echo "$dir"|xargs)"
  want "$v" || continue
  out="$DEST_ROOT/$dir"; mkdir -p "$out"
  echo "==> $v  ($repo)"
  if [[ "$file" == "*" ]]; then huggingface-cli download "$repo" --local-dir "$out"
  elif [[ -f "$out/$file" ]]; then echo "    have $file"
  else huggingface-cli download "$repo" "$file" --local-dir "$out"; fi
  canon="${CANON[$v]:-}"
  if [[ -n "$canon" ]]; then
    for t in tokenizer.json tokenizer_config.json config.json generation_config.json; do
      [[ -f "$out/$t" ]] || huggingface-cli download "$canon" "$t" --local-dir "$out" 2>/dev/null || true
    done
  fi
  echo "    -> $out"
done
echo "Done."
