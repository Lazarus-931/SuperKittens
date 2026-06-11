#!/bin/bash
# SK GPU forward for the 3 parity prompts on amelia (run detached via caffeinate).
# Usage: run_sk.sh [gpu|cpu] [prompt_index|all] [extra runner args...]
exec >> ~/sk-diffg-s1/sk_run.log 2>&1
set -x
MODE=${1:-gpu}
WHICH=${2:-all}
shift 2 || true
cd ~/sk-diffg-s1
export PYTHONPATH=~/sk-diffg-s1
for i in 1 2 3; do
  if [ "$WHICH" != "all" ] && [ "$WHICH" != "$i" ]; then continue; fi
  date
  sysctl vm.swapusage
  /usr/bin/python3 -m SuperKittens.models.gemma.diffusion.runner \
    --gguf ~/diffgemma-gguf/diffusiongemma-26B-A4B-it-Q4_K_M.gguf \
    --prompt-ids inputs/p${i}_prompt.i32 --canvas-ids inputs/p${i}_canvas.i32 \
    --out sk_${MODE}_p${i}.bin --mode ${MODE} "$@"
  echo "SK_${MODE}_P${i}_RC=$?"
done
date
echo SK_RUN_DONE
