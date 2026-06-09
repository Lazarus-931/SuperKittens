#!/bin/bash
set -e
cd ~/sk-mamba-ssdchunk
export SK_DYLIB=$PWD/build/libsk.dylib
export SK_METAL_SRC_FALLBACK=$(cat .fallback_mamba.txt)
export PYTHONPATH=$PWD
source ~/sk-venv/bin/activate 2>/dev/null || true
export SNAP=$PWD/SuperKittens/model_weights/mamba2-130m-hf
python temp/mamba2_ssdchunk/ab_ssdchunk.py
echo "===== SKIP=ssd TTFT (separate process; SK_MAMBA_SKIP is static-read) ====="
SK_MAMBA_SKIP=ssd python temp/mamba2_ssdchunk/ttft_skip.py
echo DONE_AB
