"""Correctness gate + single-stream TTFT A/B: token-by-token vs T>1 prefill.

ONE handle, one process (16 GB box discipline). Phases:
  1. correctness: for 3 prompts, token-by-token prefill -> next + 32 greedy vs
     T>1 single-forward prefill -> next + 32 greedy (fresh reset, same handle).
  2. TTFT A/B at T=128/256: time-to-first-token = consume T prompt tokens and
     produce the first next-token. tok-by-tok = T seq=1 forwards; T>1 = one
     forward(T). 2 warmups + 7 reps, median.
"""
import os, sys, time
import numpy as np

ROOT = os.path.expanduser("~/sk-ds-pfhang-r2")
sys.path.insert(0, ROOT)
from SuperKittens.inference.registry import load

SNAP = os.path.expanduser("~/ds_fit/DeepSeek-V2-Lite")
GGUF = os.path.expanduser("~/qwen-gguf/DeepSeek-V2-Lite.Q4_K_M.gguf")

def median(xs):
    s = sorted(xs); n = len(s)
    return s[n//2] if n % 2 else 0.5*(s[n//2-1]+s[n//2])

t0 = time.perf_counter()
m = load("deepseek-v2-lite", snapshot=SNAP, gguf=GGUF, seq_max=512, cache_max=512)
print(f"[load] {time.perf_counter()-t0:.1f}s", flush=True)

def prefill_tbt(ids):
    m.reset()
    nxt = None
    for tid in ids:
        nxt = m.forward(np.asarray([tid], dtype=np.int32))
    return int(nxt)

def prefill_seq(ids):
    m.reset()
    return int(m.forward(np.asarray(ids, dtype=np.int32)))

def cont32(first):
    out = [first]
    for _ in range(32):
        out.append(int(m.forward(np.asarray([out[-1]], dtype=np.int32))))
    return out

print("=== PHASE 1: correctness (tok-by-tok vs T>1, next + 32 greedy) ===", flush=True)
probes = [
    "The capital of France is",
    "Artificial intelligence is a branch of computer science that",
    "In the beginning God created the heavens and the earth",
]
for txt in probes:
    ids = np.asarray(m.tokenizer.encode(txt, bos=True), dtype=np.int32)
    a = cont32(prefill_tbt(ids))
    b = cont32(prefill_seq(ids))
    match = a == b
    nmatch = sum(x == y for x, y in zip(a, b))
    print(f"  [T={len(ids)}] identical={match} ({nmatch}/33)  {txt[:40]!r}", flush=True)
    if not match:
        print(f"    tbt: {a}", flush=True)
        print(f"    seq: {b}", flush=True)
        print(f"    tbt_text: {m.tokenizer.decode(a, skip_special=True)!r}", flush=True)
        print(f"    seq_text: {m.tokenizer.decode(b, skip_special=True)!r}", flush=True)

print("=== PHASE 2: TTFT A/B (2 warmup + 7 reps, median) ===", flush=True)
text = ("Artificial intelligence is a branch of computer science that studies "
        "intelligent agents, learning, reasoning, and perception. ") * 16
ids_full = np.asarray(m.tokenizer.encode(text, bos=True), dtype=np.int32)

results = {}
for T in (128, 256):
    ids = ids_full[:T]
    for name, fn in (("tok-by-tok", prefill_tbt), ("seq>1", prefill_seq)):
        for _ in range(2):
            fn(ids)
        reps = []
        for _ in range(7):
            t0 = time.perf_counter()
            nxt = fn(ids)
            reps.append(time.perf_counter() - t0)
        med = median(reps) * 1e3
        results[(T, name)] = med
        print(f"  [TTFT T={T}] {name:10s} median={med:8.2f} ms  next={nxt}  "
              f"reps_ms={[round(x*1e3,1) for x in reps]}", flush=True)

for T in (128, 256):
    a = results[(T, "tok-by-tok")]; b = results[(T, "seq>1")]
    print(f"[SUMMARY T={T}] tok-by-tok {a:.1f} ms -> seq>1 {b:.1f} ms = {a/b:.2f}x "
          f"({(a-b)/a*100:.1f}% faster)", flush=True)
print("AB_DONE", flush=True)
