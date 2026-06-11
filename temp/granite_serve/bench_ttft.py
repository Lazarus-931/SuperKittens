"""WIN 2 A/B: all-lanes TTFT, token-by-token lockstep prefill vs one batched
batch=N forward. Same batch=N handle, interleaved A/B, median of REPS.
Also checks decode aggregate after each prefill mode (gate: within ±2%).

Env: SK_REPO/SK_DYLIB/SK_METALLIB/SK_GRANITE_GGUF, N, T, REPS, WARMUP,
DEC_CHECK (decode steps for the after-prefill check), OUT_JSON.
"""
import json
import os
import sys
import time

import numpy as np

REPO = os.environ.get("SK_REPO", os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, REPO)
os.environ.setdefault("SK_DYLIB", os.path.join(REPO, "build", "libsk.dylib"))
os.environ.setdefault("SK_METALLIB", os.path.join(REPO, "build", "libsk.metallib"))

from SuperKittens.models.granite.granite import Granite, Config  # noqa: E402

GGUF      = os.environ["SK_GRANITE_GGUF"]
N         = int(os.environ.get("N", "8"))
T         = int(os.environ.get("T", "128"))
REPS      = int(os.environ.get("REPS", "7"))
WARMUP    = int(os.environ.get("WARMUP", "2"))
DEC_CHECK = int(os.environ.get("DEC_CHECK", "32"))
GAP       = float(os.environ.get("GAP", "0.3"))


def prefill_A(m, mat):
    """Token-by-token lockstep; returns (ttft_seconds, first_tokens)."""
    m.reset()
    t0 = time.perf_counter()
    for t in range(T):
        cur = m.forward_batched(mat[:, t])
    return time.perf_counter() - t0, cur


def prefill_B(m, prompts):
    m.reset()
    t0 = time.perf_counter()
    cur = m.prefill_batched(prompts)
    return time.perf_counter() - t0, cur


def decode_check(m, cur):
    t0 = time.perf_counter()
    for _ in range(DEC_CHECK):
        cur = m.forward_batched(cur)
    return N * DEC_CHECK / (time.perf_counter() - t0)


def main():
    cfg = Config(batch=N, seq_max=max(T, 64), cache_max=T + DEC_CHECK + 16)
    m = Granite(cfg)
    m.load_gguf(GGUF)
    rng = np.random.default_rng(13)
    prompts = [rng.integers(100, 50000, size=T).astype(np.int32)
               for _ in range(N)]
    mat = np.stack(prompts)

    for _ in range(WARMUP):
        prefill_A(m, mat); time.sleep(GAP)
        prefill_B(m, prompts); time.sleep(GAP)

    a_ms, b_ms, dec_a, dec_b = [], [], [], []
    first_match = True
    for r in range(REPS):
        ta, ca = prefill_A(m, mat)
        dec_a.append(decode_check(m, ca)); time.sleep(GAP)
        tb, cb = prefill_B(m, prompts)
        dec_b.append(decode_check(m, cb)); time.sleep(GAP)
        first_match &= bool(np.array_equal(ca, cb))
        a_ms.append(ta * 1e3); b_ms.append(tb * 1e3)
        print(f"rep {r}: A {ta*1e3:9.2f} ms | B {tb*1e3:9.2f} ms | "
              f"dec A {dec_a[-1]:7.2f} B {dec_b[-1]:7.2f} agg tok/s",
              flush=True)

    a_med = float(np.median(a_ms)); b_med = float(np.median(b_ms))
    da = float(np.median(dec_a)); db = float(np.median(dec_b))
    print(f"[win2] N={N} T={T}: A(tbt)={a_med:.2f} ms B(batched)={b_med:.2f} ms "
          f"speedup={a_med/b_med:.3f}x first-tokens-match={first_match}")
    print(f"[win2] decode-after-prefill: A {da:.2f} vs B {db:.2f} agg tok/s "
          f"(ratio {db/da:.4f}; gate ±2%)")
    out = os.environ.get("OUT_JSON",
                         os.path.join(os.path.dirname(__file__),
                                      f"win2_n{N}_t{T}.json"))
    with open(out, "w") as f:
        json.dump(dict(N=N, T=T, a_ms=a_ms, b_ms=b_ms, a_med=a_med,
                       b_med=b_med, speedup=a_med / b_med,
                       dec_a=dec_a, dec_b=dec_b,
                       dec_ratio=db / da, first_match=first_match), f)
    print(f"[win2] wrote {out}")


if __name__ == "__main__":
    main()
