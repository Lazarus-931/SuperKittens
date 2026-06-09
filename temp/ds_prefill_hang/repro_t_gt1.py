"""Repro probe: does DeepSeek-V2-Lite MLA T>1 prefill hang on main?

Runs T=16/64/128 single-forward prefills (fresh reset each) on ONE handle.
SK_DS_DEBUG=1 in the env prints the last-row logits top10 per forward, which
doubles as the finite-logits check. A hang would trip the outer `timeout`.
"""
import os, sys, time
import numpy as np

ROOT = os.path.expanduser("~/sk-ds-pfhang-r2")
sys.path.insert(0, ROOT)
from SuperKittens.inference.registry import load

SNAP = os.path.expanduser("~/ds_fit/DeepSeek-V2-Lite")
GGUF = os.path.expanduser("~/qwen-gguf/DeepSeek-V2-Lite.Q4_K_M.gguf")

t0 = time.perf_counter()
m = load("deepseek-v2-lite", snapshot=SNAP, gguf=GGUF, seq_max=512, cache_max=512)
print(f"[load] {time.perf_counter()-t0:.1f}s", flush=True)

text = ("Artificial intelligence is a branch of computer science that studies "
        "intelligent agents, learning, reasoning, and perception. ") * 8
ids_full = np.asarray(m.tokenizer.encode(text, bos=True), dtype=np.int32)
print(f"[tok] {len(ids_full)} prompt tokens available", flush=True)

for T in (16, 64, 128):
    ids = ids_full[:T]
    m.reset()
    t0 = time.perf_counter()
    nxt = m.forward(ids)
    dt = (time.perf_counter() - t0) * 1e3
    print(f"[prefill T={T}] OK next={nxt} {dt:.1f} ms", flush=True)

# Short greedy continuation off the T=128 prefill to show the cache is sane.
cont = [int(nxt)]
for _ in range(15):
    cont.append(m.forward(np.asarray([cont[-1]], dtype=np.int32)))
print(f"[cont after T=128] {m.tokenizer.decode(cont, skip_special=True)!r}", flush=True)
print("REPRO_DONE no hang", flush=True)
