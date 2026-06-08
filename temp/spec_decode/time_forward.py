"""Per-seq forward timing on 0.6B: confirm seq=K verify cost vs seq=1, and check
the gemm_mma path is active (SK_NO_GEMM_MMA toggles it). Local sanity only —
the real amortization claim is on 4B-Q4_K_M; here we just confirm the seq>1 path
isn't pathologically slow (which would explain the 0.003x synthetic number)."""
import _env  # noqa: F401
import os, time
import numpy as np
import SuperKittens as sk
import SuperKittens.models.qwen.qwen  # noqa

m = sk.load("qwen3-0.6b", cache_max=512)
rng = np.random.default_rng(1)
prompt = rng.integers(10, 4000, size=16).astype(np.int32)


def time_seq(seq, iters=20):
    # warm
    m.reset(); m.forward(prompt)
    chunk = rng.integers(10, 4000, size=seq).astype(np.int32)
    m.forward(chunk)
    # measure: each iter does one seq-token forward at current_pos>0
    ts = []
    for _ in range(iters):
        m.reset(); m.forward(prompt)
        t0 = time.perf_counter()
        m.forward(chunk)
        m.get_logits_rows(seq)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3  # ms


print(f"GEMM_MMA on={'SK_NO_GEMM_MMA' not in os.environ}")
for seq in (1, 2, 4, 6, 8):
    print(f"seq={seq}: {time_seq(seq):.2f} ms")
m.close()
