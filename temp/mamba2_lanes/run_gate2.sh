#!/bin/bash
set -e
cd ~/sk-mamba-lanes
L=temp/mamba2_lanes
SK_DYLIB=$PWD/build/libsk_base.dylib TAG=base bash $L/run_lab.sh $L/gate_single.py
SK_DYLIB=$PWD/build/libsk.dylib      TAG=new  bash $L/run_lab.sh $L/gate_single.py
python3 - << "PY"
import numpy as np
a = np.load("temp/mamba2_lanes/single_base_logits.npy")
b = np.load("temp/mamba2_lanes/single_new_logits.npy")
ta = open("temp/mamba2_lanes/single_base_toks.txt").read()
tb = open("temp/mamba2_lanes/single_new_toks.txt").read()
print("logits byte-identical:", a.tobytes() == b.tobytes())
print("tokens identical:", ta == tb)
print("GATE2_PASS" if (a.tobytes() == b.tobytes() and ta == tb) else "GATE2_FAIL")
PY
