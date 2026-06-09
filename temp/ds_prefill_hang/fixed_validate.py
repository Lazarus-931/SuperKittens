"""Post-fix validation on the FIXED build (one handle, one process).

  A. drift gate: 12 fresh seq>1 T=128 forwards -> next must be stable AND equal
     to the tok-by-tok truth from the SAME process.
  B. correctness: T=6/11/11 (short) and T=64/128/256 (long): tok-by-tok vs seq>1
     next + 32-greedy continuation, token-identical required.
  C. TTFT A/B (2 warmup + 7 reps median): tok-by-tok vs seq>1, T=128/256.
  D. decode stream: 64 greedy tokens after a tok-by-tok prefill (compare with the
     base-build run of the same to prove decode is byte-identical).
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

m = load("deepseek-v2-lite", snapshot=SNAP, gguf=GGUF, seq_max=512, cache_max=512)
print(f"[dylib] {os.environ.get('SK_DYLIB','?')}", flush=True)

def tbt(ids):
    m.reset()
    nxt = None
    for tid in ids:
        nxt = int(m.forward(np.asarray([tid], dtype=np.int32)))
    return nxt

def seq(ids):
    m.reset()
    return int(m.forward(np.asarray(ids, dtype=np.int32)))

def cont(first, n=32):
    out = [first]
    for _ in range(n):
        out.append(int(m.forward(np.asarray([out[-1]], dtype=np.int32))))
    return out

sent = ("Artificial intelligence is a branch of computer science that studies "
        "intelligent agents, learning, reasoning, and perception. ") * 16
ids_full = np.asarray(m.tokenizer.encode(sent, bos=True), dtype=np.int32)

print("=== A. drift gate (12 fresh seq T=128) ===", flush=True)
truth = tbt(ids_full[:128])
runs = [seq(ids_full[:128]) for _ in range(12)]
ok = all(r == truth for r in runs)
print(f"  tbt_truth={truth} seq_runs={runs} STABLE_AND_CORRECT={ok}", flush=True)

print("=== B. correctness: next + 32-greedy, tbt vs seq ===", flush=True)
probes = [
    ("capital",  "The capital of France is"),
    ("ai",       "Artificial intelligence is a branch of computer science that"),
    ("genesis",  "In the beginning God created the heavens and the earth"),
]
for name, txt in probes:
    ids = np.asarray(m.tokenizer.encode(txt, bos=True), dtype=np.int32)
    a = cont(tbt(ids)); b = cont(seq(ids))
    print(f"  [short {name} T={len(ids)}] identical={a==b} "
          f"({sum(x==y for x,y in zip(a,b))}/33)", flush=True)
for T in (64, 128, 256):
    ids = ids_full[:T]
    a = cont(tbt(ids)); b = cont(seq(ids))
    print(f"  [long T={T}] identical={a==b} ({sum(x==y for x,y in zip(a,b))}/33)", flush=True)
    if a != b:
        print(f"    tbt: {a}\n    seq: {b}", flush=True)
        print(f"    tbt_text: {m.tokenizer.decode(a, skip_special=True)!r}", flush=True)
        print(f"    seq_text: {m.tokenizer.decode(b, skip_special=True)!r}", flush=True)
    else:
        print(f"    text: {m.tokenizer.decode(b, skip_special=True)!r}", flush=True)

print("=== C. TTFT A/B (2 warmup + 7 reps, median) ===", flush=True)
results = {}
for T in (128, 256):
    ids = ids_full[:T]
    for name, fn in (("tok-by-tok", tbt), ("seq>1", seq)):
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

print("=== D. decode stream (tbt prefill + 64 greedy) ===", flush=True)
ids = np.asarray(m.tokenizer.encode("The capital of France is", bos=True), dtype=np.int32)
stream = cont(tbt(ids), n=64)
print(f"  DECODE_STREAM={stream}", flush=True)
print("VALIDATE_DONE", flush=True)
