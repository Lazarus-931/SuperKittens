#!/bin/zsh
# run_gate3.sh <name> <prompt.i32> <steps> <seed> [extra generate.py args...]
# One SK e2e generation, detached-safe. Watchdog: system swap > 3.5 GB,
# root-disk free < 4 GB, or wall > 3600 s (post-cleanup amelia has ~12 GB
# free; tighter-than-Stage-1 thresholds abort long before the host is at risk).
LAB=~/sk-diffg-s2b
GGUF=~/diffgemma-gguf/diffusiongemma-26B-A4B-it-Q4_K_M.gguf
name=$1; pids=$2; steps=$3; seed=$4; shift 4
out=$LAB/gen/$name
mkdir -p $out
cd $LAB
PYTHONPATH=$LAB caffeinate -is python3 -m SuperKittens.models.gemma.diffusion.generate \
  --gguf $GGUF --prompt-ids $pids --out-dir $out --steps $steps --seed $seed \
  --sc-embt $LAB/dg_embT_f16.bin "$@" > $out/run.log 2>&1 &
PID=$!
echo $PID > $out/pid
SECS=0
while kill -0 $PID 2>/dev/null; do
  sleep 90; SECS=$((SECS+90))
  swap=$(sysctl -n vm.swapusage | awk '{print $6}' | tr -d M)
  free=$(df -m /System/Volumes/Data | tail -1 | awk '{print $4}')
  rss=$(ps -o rss= -p $PID 2>/dev/null)
  echo "t=$SECS rss_kb=$rss swap=${swap}M diskfree=${free}M freepct=$(memory_pressure -Q 2>/dev/null | awk -F': ' '/percentage/{print $2}')" >> $out/mem.log
  if (( ${swap%.*} > 3500 )); then echo "WATCHDOG swap=${swap}M KILL" >> $out/run.log; kill -9 $PID; exit 3; fi
  if (( free < 4000 )); then echo "WATCHDOG diskfree=${free}M KILL" >> $out/run.log; kill -9 $PID; exit 4; fi
  if (( SECS > 3600 )); then echo "WATCHDOG timeout KILL" >> $out/run.log; kill -9 $PID; exit 5; fi
done
wait $PID
echo "EXIT $?" >> $out/run.log
