#!/bin/bash
set -e
cd ~/sk-mamba-lanes
MOD=temp/mamba2_lanes/mod_src
BASE=temp/mamba2_lanes/base_src
DST=SuperKittens/models/ssm/mamba2
mkdir -p "$MOD"
cp "$DST"/launcher.c++ "$DST"/launcher.h "$DST"/mamba2_model.h "$MOD"/
echo "=== BASE build (pristine main launcher/model.h) ==="
cp "$BASE"/launcher.c++ "$BASE"/launcher.h "$BASE"/mamba2_model.h "$DST"/
bash build_dylib.sh
mv build/libsk.dylib build/libsk_base.dylib
echo "=== MOD build (batched-lane prefill) ==="
cp "$MOD"/launcher.c++ "$MOD"/launcher.h "$MOD"/mamba2_model.h "$DST"/
bash build_dylib.sh
echo BUILD_BOTH_OK
