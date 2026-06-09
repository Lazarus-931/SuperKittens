"""E2B decode coherence + tok/s harness. Run on amelia (CLT-only, runtime metal-compile)."""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.expanduser("~/sk-gemma-opt"))

import SuperKittens.api as api

VARIANT = os.environ.get("SK_VARIANT", "gemma4-e2b")
N_TOK   = int(os.environ.get("SK_NTOK", "48"))
PROMPT  = os.environ.get("SK_PROMPT", "The capital of France is")

t0 = time.time()
m = api.load(VARIANT)
print(f"[load] {VARIANT} in {time.time()-t0:.1f}s", flush=True)

tok = m.tokenizer
ids = tok.encode(PROMPT, bos=True) if hasattr(tok, "encode") else None
if ids is None:
    # fallback: chat-style
    ids = tok.chat([{"role": "user", "content": PROMPT}], bos=True)
ids = np.array(ids, dtype=np.int32)
print(f"[prompt] {PROMPT!r} -> {len(ids)} tokens", flush=True)

# Warmup forward (prefill) — also primes PSOs.
m.reset()
t0 = time.time()
first = int(m.forward(ids)[0])
prefill_s = time.time() - t0
print(f"[prefill] {prefill_s*1000:.0f} ms, first tok={first}", flush=True)

out = [first]
last = first
# Decode timing: time N_TOK single-token forwards.
tdec0 = time.time()
for i in range(N_TOK - 1):
    last = int(m.forward(np.array([last], dtype=np.int32))[0])
    out.append(last)
dec_s = time.time() - tdec0
toks = N_TOK - 1
print(f"[decode] {toks} tok in {dec_s:.3f}s -> {toks/dec_s:.2f} tok/s", flush=True)

text = tok.decode(out, skip_special=True) if hasattr(tok, "decode") else str(out)
print(f"[gen ids] {out[:40]}", flush=True)
print(f"[gen text] {text!r}", flush=True)
