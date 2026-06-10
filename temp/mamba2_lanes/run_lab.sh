#!/bin/bash
# derek runner: env mirrors ~/sk-mamba-prefill/run_prof.sh (runtime metal
# compile via SK_METAL_SRC_FALLBACK; weights reused read-only from
# sk-mamba-prefill). Usage: bash run_lab.sh <script.py> [env overrides first].
set -e
cd ~/sk-mamba-lanes
export SK_DYLIB=${SK_DYLIB:-$PWD/build/libsk.dylib}
export SK_METAL_SRC_FALLBACK=$(cat .fallback_mamba.txt)
export PYTHONPATH=$PWD
source ~/sk-venv/bin/activate 2>/dev/null || true
export SNAP=${SNAP:-$HOME/sk-mamba-prefill/SuperKittens/model_weights/mamba2-130m-hf}
python "$@"
