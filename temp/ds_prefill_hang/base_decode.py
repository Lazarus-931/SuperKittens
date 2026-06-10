"""BASE-build decode stream (T=1-only dispatches) for cross-build identity."""
import os, sys
import numpy as np

ROOT = os.path.expanduser("~/sk-ds-pfhang-r2")
sys.path.insert(0, ROOT)
from SuperKittens.inference.registry import load

SNAP = os.path.expanduser("~/ds_fit/DeepSeek-V2-Lite")
GGUF = os.path.expanduser("~/qwen-gguf/DeepSeek-V2-Lite.Q4_K_M.gguf")

m = load("deepseek-v2-lite", snapshot=SNAP, gguf=GGUF, seq_max=512, cache_max=512)
print(f"[dylib] {os.environ.get('SK_DYLIB','?')}", flush=True)

ids = np.asarray(m.tokenizer.encode("The capital of France is", bos=True), dtype=np.int32)
m.reset()
nxt = None
for tid in ids:
    nxt = int(m.forward(np.asarray([tid], dtype=np.int32)))
stream = [nxt]
for _ in range(64):
    stream.append(int(m.forward(np.asarray([stream[-1]], dtype=np.int32))))
print(f"  DECODE_STREAM={stream}", flush=True)
print("BASE_DECODE_DONE", flush=True)
