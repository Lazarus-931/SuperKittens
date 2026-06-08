#!/bin/bash
# DeepSeek-V2-Lite decode memory-pressure probe (run ON A MINI, e.g. derek).
# Confirms the swap-death pathology: model resident > usable RAM -> OS compressor + swap.
#
#   ssh derek 'bash -s' < temp/profile_mem.sh <PID>
#
# Emits one snapshot of vm_stat / swapusage / process state. Loop it across a
# decode run to watch swapins climb on the critical path.
PID="${1:-}"
echo "=== $(date) ==="
vm_stat | grep -E "Pages free|Pages wired|stored in compressor|occupied by compressor|Swapins|Swapouts|Pageins"
sysctl vm.swapusage
[ -n "$PID" ] && ps -o pid,etime,%cpu,rss,state -p "$PID"
