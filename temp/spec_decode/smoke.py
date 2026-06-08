"""Smoke + correctness: load 0.6B, exercise the spec-decode ABI, and verify
the all-rows LM head matches per-position single-token forwards (the property
spec-decode verify relies on). Runs on an 8GB Mac (0.6B-only)."""
import os
import _env  # noqa: F401  (sets SK_DYLIB/SK_METALLIB)
import numpy as np
import SuperKittens as sk
import SuperKittens.models.qwen.qwen  # register adapter

# Local 8GB-Mac correctness run: snapshot override points at a 0.6B Q8_0 gguf.
_snap = os.environ.get("SK_SNAP_06B")
_kw = dict(cache_max=512)
if _snap:
    _kw["snapshot"] = _snap
m = sk.load("qwen3-0.6b", **_kw)
print("loaded qwen3-0.6b; vocab=", m.cfg.vocab_size, "n_layers=", m.cfg.n_layers)

for name in ("get_logits_rows", "get_pos", "set_pos", "set_lm_head_all_rows"):
    print(f"has {name}:", hasattr(m, name))

ids = np.arange(5, 25, dtype=np.int32)  # arbitrary 20-token "prompt"
m.reset()
print("pos before prefill:", m.get_pos())
m.forward(ids)
print("pos after prefill:", m.get_pos())
rows = m.get_logits_rows(len(ids))
print("logits_rows shape:", rows.shape, "dtype:", rows.dtype)
arg = int(np.argmax(rows[-1].astype(np.float32)))
print("last-row argmax:", arg)

# --- Correctness: all-rows LM head == per-position single-token decode ---
# Build a reference chain: prefill prompt, then step-decode K tokens, recording
# the argmax produced at each position. Then re-run the same [prompt + K] as one
# all-rows forward and confirm row j's argmax matches the j-th reference token.
m.set_lm_head_all_rows(True)
prompt = np.arange(100, 110, dtype=np.int32)
m.reset()
m.forward(prompt)
chain = [int(np.argmax(m.get_logits_rows(len(prompt))[-1].astype(np.float32)))]
K = 5
for _ in range(K):
    m.forward(np.array([chain[-1]], np.int32))
    chain.append(int(np.argmax(m.get_logits_rows(1)[-1].astype(np.float32))))
# chain[i] = argmax after consuming prompt + chain[:i]
# One verify forward over [prompt..., chain[0..K-1]] must reproduce chain[1..K]
# at the corresponding rows (relative rows len(prompt)-1 .. len(prompt)+K-1).
m.reset()
verify_in = np.concatenate([prompt, np.array(chain[:K], np.int32)])
m.forward(verify_in)
vrows = m.get_logits_rows(len(verify_in)).astype(np.float32)
ok = True
# verify_in = [prompt..., chain[0..K-1]]. Row r predicts the token after position
# r; the row at index (len(prompt)-1)+j consumed prompt+chain[:j], so its argmax
# must equal the per-position decode token chain[j].
for j in range(K):
    row = len(prompt) - 1 + j
    a = int(np.argmax(vrows[row]))
    if a != chain[j]:
        ok = False
        print(f"  MISMATCH row {row}: all-rows={a} ref={chain[j]}")
print("all-rows verify matches per-position decode:", ok)

m.close()
print("SMOKE OK" if ok else "SMOKE FAIL")
