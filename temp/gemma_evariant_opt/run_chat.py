"""E2B/E4B chat coherence + decode tok/s. Run on amelia."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("~/sk-gemma-opt"))
import SuperKittens.api as api

VARIANT = os.environ.get("SK_VARIANT", "gemma4-e2b")
N_TOK   = int(os.environ.get("SK_NTOK", "64"))
PROMPT  = os.environ.get("SK_PROMPT", "Explain in two sentences why the sky is blue.")
REPS    = int(os.environ.get("SK_REPS", "3"))

t0 = time.time()
m = api.load(VARIANT)
print(f"[load] {VARIANT} in {time.time()-t0:.1f}s", flush=True)
tok = m.tokenizer
eos_ids = getattr(tok, "eos_ids", None)
eos = getattr(tok, "eos_id", None)
ids = tok.chat([{"role": "user", "content": PROMPT}], bos=True)
ids = np.array(ids, dtype=np.int32)
print(f"[prompt] {PROMPT!r} -> {len(ids)} tokens; eos={eos} eos_ids={eos_ids}", flush=True)

# Coherence pass (greedy, with EOS stop).
m.reset()
out = m.generate(ids, max_new_tokens=N_TOK, temperature=0.0, eos_id=eos, eos_ids=eos_ids)
text = tok.decode(out, skip_special=True)
print(f"[coherence n={len(out)}] {text!r}", flush=True)

# Decode speed: fixed N_TOK, no early stop, multi-rep min.
best = 0.0
for r in range(REPS):
    m.reset()
    first = int(m.forward(ids)[0])
    last = first; n = 1
    t = time.time()
    for _ in range(N_TOK - 1):
        last = int(m.forward(np.array([last], dtype=np.int32))[0]); n += 1
    dt = time.time() - t
    tps = (N_TOK - 1) / dt
    best = max(best, tps)
    print(f"[rep{r}] {N_TOK-1} tok in {dt:.3f}s -> {tps:.2f} tok/s", flush=True)
print(f"[BEST decode] {best:.2f} tok/s", flush=True)
