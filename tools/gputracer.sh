#!/usr/bin/env bash
# gputracer.sh — capture and/or report Apple .gputrace findings for a SK kernel.
#
# Wraps tools/gputracer.py. Two modes:
#
#   Parse an existing capture (runs anywhere — pure stdlib):
#       tools/gputracer.sh kernel.gputrace [--json] [--raw]
#
#   Capture a kernel's trace then report (run on a GPU host, e.g. a mini;
#   this script exports MTL_CAPTURE_ENABLED=1, which Metal requires):
#       tools/gputracer.sh capture out.gputrace \
#           --src SuperKittens/kernels/norm/rmsnorm.metal --fn rmsnorm \
#           --grid 4096,1,1 --tg 256,1,1 --buf 16384 --buf 16384 --buf 64
#
# A .gputrace is a command-intent recording (kernel/grid/threadgroup/buffers),
# not a timing log — pair it with SuperKittens/benchmark/harness for tok/s & GB/s.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-$(command -v python3 || command -v python)}"
TRACER="$SCRIPT_DIR/gputracer.py"

if [[ ! -f "$TRACER" ]]; then
  echo "gputracer.py not found next to this script ($TRACER)" >&2
  exit 1
fi

usage() {
  sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

case "${1:-}" in
  ""|-h|--help)
    usage 0 ;;
  capture)
    shift
    # Metal only records a .gputrace when this is set in the process environment.
    export MTL_CAPTURE_ENABLED=1
    exec "$PY" "$TRACER" capture "$@" ;;
  *)
    # parse mode: first arg is a .gputrace bundle path
    exec "$PY" "$TRACER" "$@" ;;
esac
