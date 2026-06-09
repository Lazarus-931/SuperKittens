"""Probe the T>1 nondeterminism: CB error status + per-layer divergence.

Phase A: 12 fresh seq>1 T=128 forwards; print next each time (stderr carries
any 'ds forward: command buffer ERROR' lines from the patched launcher).
Phase B: 3 fresh tok-by-tok T=128 (trusted path determinism check).
Phase C: with SK_DS_DBGL set by the caller this would change timing; instead
the caller runs a second pass with SK_DS_DBGL=1 to localize the first
diverging layer (per-layer L2 to stderr).
"""
import os, sys
import numpy as np

ROOT = os.path.expanduser("~/sk-ds-pfhang-r2")
sys.path.insert(0, ROOT)
from SuperKittens.inference.registry import load

SNAP = os.path.expanduser("~/ds_fit/DeepSeek-V2-Lite")
GGUF = os.path.expanduser("~/qwen-gguf/DeepSeek-V2-Lite.Q4_K_M.gguf")

m = load("deepseek-v2-lite", snapshot=SNAP, gguf=GGUF, seq_max=512, cache_max=512)

sent = ("Artificial intelligence is a branch of computer science that studies "
        "intelligent agents, learning, reasoning, and perception. ") * 16
ids = np.asarray(m.tokenizer.encode(sent, bos=True), dtype=np.int32)[:128]

n_seq = int(os.environ.get("N_SEQ", "12"))
outs = []
for i in range(n_seq):
    m.reset()
    nxt = int(m.forward(ids))
    outs.append(nxt)
    print(f"[seq T=128 run {i}] next={nxt}", flush=True)
print(f"[seq summary] {outs} distinct={sorted(set(outs))}", flush=True)

if not os.environ.get("SK_DS_DBGL"):
    for i in range(3):
        m.reset()
        nxt = None
        for tid in ids:
            nxt = int(m.forward(np.asarray([tid], dtype=np.int32)))
        print(f"[tbt T=128 run {i}] next={nxt}", flush=True)
print("PROBE_DONE", flush=True)
