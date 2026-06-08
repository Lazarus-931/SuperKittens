"""Smoke test: load 0.6B-Q8_0, exercise the spec-decode ABI, single forward."""
import _env  # noqa: F401  (sets SK_DYLIB/SK_METALLIB)
import numpy as np
import SuperKittens as sk
import SuperKittens.models.qwen.qwen  # register adapter

m = sk.load("qwen3-0.6b", cache_max=512)
print("loaded qwen3-0.6b; vocab=", m.cfg.vocab_size, "n_layers=", m.cfg.n_layers)

# ABI presence
for name in ("get_logits_rows", "get_pos", "set_pos"):
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
m.close()
print("SMOKE OK")
