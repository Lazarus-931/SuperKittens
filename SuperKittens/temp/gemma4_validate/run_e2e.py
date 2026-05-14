"""End-to-end gemma4 chat test."""
import sys, time
import numpy as np

print("[*] importing SuperKittens", flush=True)
import SuperKittens as sk
import SuperKittens.models.gemma.gemma4  # registers
print("[*] registered keys:", list(sk.MODEL_REGISTRY.keys()), flush=True)

t0 = time.time()
print("[*] loading gemma4-e2b ...", flush=True)
m = sk.load("gemma4-e2b")
print(f"[*] load took {time.time()-t0:.1f}s", flush=True)
print(f"[*] tokenizer family={m.tokenizer.family} bos={m.tokenizer.bos_id} eos={m.tokenizer.eos_id} vocab={m.tokenizer.vocab_size}", flush=True)

# --- argmax probe ---
m.reset()
out = m.forward(np.array([10979, 236888], dtype=np.int32))
print(f"[probe] forward([10979, 236888]) argmax = {int(out[0])}  (expect 10979)", flush=True)

# --- chat 1 ---
m.reset()
t0 = time.time()
r1 = m.chat("What is the capital of France?", max_new_tokens=32)
dt = time.time() - t0
print(f"[chat1] dt={dt:.1f}s  tok/s={32/dt:.2f}", flush=True)
print(f"[chat1] {r1!r}", flush=True)

# --- chat 2 ---
m.reset()
t0 = time.time()
r2 = m.chat("Write one sentence about cats.", max_new_tokens=32)
dt = time.time() - t0
print(f"[chat2] dt={dt:.1f}s  tok/s={32/dt:.2f}", flush=True)
print(f"[chat2] {r2!r}", flush=True)
