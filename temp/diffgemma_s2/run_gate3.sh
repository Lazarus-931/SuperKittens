#!/bin/zsh
# run_gate3.sh <name> <prompt.i32> <steps> <seed> [extra generate.py args...]
# One SK e2e generation, detached-safe, with the Stage-1 watchdog rules:
# kill on system swap > 5.5 GB, root-disk free < 400 MB, or wall > 3600 s.
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
  sleep 30; SECS=$((SECS+30))
  swap=$(sysctl -n vm.swapusage | awk '{print $6}' | tr -d M)
  free=$(df -m /System/Volumes/Data | tail -1 | awk '{print $4}')
  if (( ${swap%.*} > 5500 )); then echo "WATCHDOG swap=${swap}M KILL" >> $out/run.log; kill -9 $PID; exit 3; fi
  if (( free < 400 )); then echo "WATCHDOG diskfree=${free}M KILL" >> $out/run.log; kill -9 $PID; exit 4; fi
  if (( SECS > 3600 )); then echo "WATCHDOG timeout KILL" >> $out/run.log; kill -9 $PID; exit 5; fi
done
wait $PID
echo "EXIT $?" >> $out/run.log
