#!/bin/zsh
# gate1b_gpu.sh — Gate 1 GPU leg: SK Metal forward with SC active vs the
# reference dumps produced by gate1_sc.py (same canvas1/L0 inputs).
set -e
LAB=~/sk-diffg-s2b
GGUF=~/diffgemma-gguf/diffusiongemma-26B-A4B-it-Q4_K_M.gguf
G1=$LAB/gate1
cd $LAB
export PYTHONPATH=$LAB

echo "== GPU zero-SC control =="
caffeinate -is python3 -m SuperKittens.models.gemma.diffusion.runner \
  --gguf $GGUF --prompt-ids ~/sk-diffg-s1/inputs/p1_prompt.i32 \
  --canvas-ids $G1/canvas1.i32 --out $G1/G1_nosc.f32 --mode gpu

echo "== GPU SC-active (temp_inv = 1/0.8) =="
caffeinate -is python3 -m SuperKittens.models.gemma.diffusion.runner \
  --gguf $GGUF --prompt-ids ~/sk-diffg-s1/inputs/p1_prompt.i32 \
  --canvas-ids $G1/canvas1.i32 --out $G1/G1_sc.f32 --mode gpu \
  --sc-logits $G1/L0.f32 --sc-temp-inv 1.25 \
  --sc-embt $LAB/dg_embT_f16.bin --dump-dir $G1/dump_gpu --dump-names sc_sig

python3 - <<'EOF'
import json
import numpy as np

G1 = "/Users/amelia/sk-diffg-s2b/gate1"
C, V, D = 256, 262144, 2816


def stats(name, a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    rel = float(np.sqrt(((a - b) ** 2).mean()) / (np.sqrt((b ** 2).mean()) or 1.0))
    cos = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    print(json.dumps({"name": name, "rel_rms": round(rel, 6), "cos": round(cos, 8)}))


def agree(name, a, b):
    am = (a.argmax(1) == b.argmax(1)).mean()
    print(json.dumps({"name": name, "argmax_agree": float(am)}))


sig_gpu = np.load(f"{G1}/dump_gpu/sc_sig.-1.npy")
sig_ref = np.fromfile(f"{G1}/sc_sig_r001.f32", np.float32).reshape(C, D)
stats("sc_sig GPU-vs-ref", sig_gpu, sig_ref)
L1_sc = np.fromfile(f"{G1}/L1_sc.f32", np.float32).reshape(C, V)
L1_nosc = np.fromfile(f"{G1}/L1_nosc.f32", np.float32).reshape(C, V)
G_sc = np.fromfile(f"{G1}/G1_sc.f32", np.float32).reshape(C, V)
G_nosc = np.fromfile(f"{G1}/G1_nosc.f32", np.float32).reshape(C, V)
agree("logits GPU-vs-ref SC-active", G_sc, L1_sc)
agree("logits GPU-vs-ref zero-SC (envelope)", G_nosc, L1_nosc)
stats("logits GPU-vs-ref SC-active", G_sc, L1_sc)
stats("logits GPU-vs-ref zero-SC (envelope)", G_nosc, L1_nosc)
EOF
echo GATE1B_DONE
