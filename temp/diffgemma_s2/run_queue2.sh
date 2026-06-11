#!/bin/zsh
# run_queue2.sh — Gate-3 remaining prompts + Gate-4 reference cli, strictly
# sequential (one heavy process at a time on the 16 GB host).
LAB=~/sk-diffg-s2b
GGUF=~/diffgemma-gguf/diffusiongemma-26B-A4B-it-Q4_K_M.gguf

$LAB/tools/run_gate3.sh pq_s16 $LAB/gen/pqcli_prompt.i32 16 1234
$LAB/tools/run_gate3.sh p2_s16 $LAB/gen/p2cli_prompt.i32 16 1234

# Gate 4 reference: uninstrumented cli (no DG_EB_DUMP in env), same prompt,
# same seed and S as the SK p1 run. Same watchdog rules as run_gate3.sh —
# the CPU cli is ~30 s/step so 3600 s is generous.
caffeinate -is ~/llamacpp-diffg/build/bin/llama-diffusion-cli -m $GGUF \
  -p "What is the capital of France?" \
  --diffusion-eb-max-steps 16 --seed 1234 -n 256 \
  > $LAB/gen/cli_p1_s16.log 2>&1 &
CPID=$!
SECS=0
while kill -0 $CPID 2>/dev/null; do
  sleep 90; SECS=$((SECS+90))
  swap=$(sysctl -n vm.swapusage | awk '{print $6}' | tr -d M)
  free=$(df -m /System/Volumes/Data | tail -1 | awk '{print $4}')
  echo "t=$SECS swap=${swap}M diskfree=${free}M" >> $LAB/gen/cli_p1_s16.mem.log
  if (( ${swap%.*} > 3500 )); then echo "WATCHDOG swap KILL" >> $LAB/gen/cli_p1_s16.log; kill -9 $CPID; break; fi
  if (( free < 4000 )); then echo "WATCHDOG disk KILL" >> $LAB/gen/cli_p1_s16.log; kill -9 $CPID; break; fi
  if (( SECS > 3600 )); then echo "WATCHDOG timeout KILL" >> $LAB/gen/cli_p1_s16.log; kill -9 $CPID; break; fi
done
echo QUEUE2_DONE
