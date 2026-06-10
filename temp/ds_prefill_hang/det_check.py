"""Pin down the long-T next-token divergence seen in ab_ttft.py.

Checks, in one process / one handle:
  1. input ids: are the first-128 tokens of (sentence*8) and (sentence*16)
     encodings identical (rules out an input mismatch between repro and ab)?
  2. determinism: 5 fresh seq>1 T=128 forwards — same next every time?
  3. tok-by-tok vs seq>1 (MoE-MMA on AND per-slot via runtime toggle) next +
     24-greedy continuation text at T=128 and T=256.
"""
import os, sys, ctypes
import numpy as np

ROOT = os.path.expanduser("~/sk-ds-pfhang-r2")
sys.path.insert(0, ROOT)
from SuperKittens.inference.registry import load
import SuperKittens.models.deepseek.deepseek as dsmod

SNAP = os.path.expanduser("~/ds_fit/DeepSeek-V2-Lite")
GGUF = os.path.expanduser("~/qwen-gguf/DeepSeek-V2-Lite.Q4_K_M.gguf")

m = load("deepseek-v2-lite", snapshot=SNAP, gguf=GGUF, seq_max=512, cache_max=512)
lib = dsmod._load()
lib.sk_deepseek_set_moe_mma.argtypes = [ctypes.c_int]
lib.sk_deepseek_set_moe_mma.restype = None

sent = ("Artificial intelligence is a branch of computer science that studies "
        "intelligent agents, learning, reasoning, and perception. ")
ids8  = np.asarray(m.tokenizer.encode(sent * 8,  bos=True), dtype=np.int32)
ids16 = np.asarray(m.tokenizer.encode(sent * 16, bos=True), dtype=np.int32)
print(f"[ids] len8={len(ids8)} len16={len(ids16)} "
      f"first128_equal={bool((ids8[:128] == ids16[:128]).all())}", flush=True)

def seq_next(ids, mma=-1):
    lib.sk_deepseek_set_moe_mma(mma)
    m.reset()
    r = int(m.forward(np.asarray(ids, dtype=np.int32)))
    lib.sk_deepseek_set_moe_mma(-1)
    return r

def tbt_next(ids):
    m.reset()
    nxt = None
    for tid in ids:
        nxt = int(m.forward(np.asarray([tid], dtype=np.int32)))
    return nxt

def cont(first, n=24):
    out = [first]
    for _ in range(n):
        out.append(int(m.forward(np.asarray([out[-1]], dtype=np.int32))))
    return out

print("[determinism] 5 fresh seq>1 T=128 forwards (ids16[:128]):",
      [seq_next(ids16[:128]) for _ in range(5)], flush=True)
print("[determinism] 3 fresh seq>1 T=128 forwards (ids8[:128]): ",
      [seq_next(ids8[:128]) for _ in range(3)], flush=True)
print("[order-effect] seq T=16, T=64, then T=128 (repro order, ids8):",
      seq_next(ids8[:16]), seq_next(ids8[:64]), seq_next(ids8[:128]), flush=True)

for T in (128, 256):
    ids = ids16[:T]
    a = tbt_next(ids);            ca = cont(a)
    b = seq_next(ids, mma=1);     cb = cont(b)
    c = seq_next(ids, mma=0)      # per-slot MoE path, next only
    print(f"--- T={T} ---", flush=True)
    print(f"  next: tbt={a} seq_mma={b} seq_perslot={c}", flush=True)
    print(f"  tbt_text:  {m.tokenizer.decode(ca, skip_special=True)!r}", flush=True)
    print(f"  seq_text:  {m.tokenizer.decode(cb, skip_special=True)!r}", flush=True)
print("DET_DONE", flush=True)
