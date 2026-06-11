"""WIN 1 A/B: aggregate decode tok/s, N-lane lockstep vs Nx sequential
single-stream. One batch=N handle; A-side decode runs the untouched
single-stream ABI (sk_granite_forward, lane 0) so it IS the production
baseline. Interleaved A/B per rep (thermal), median of REPS.

Env: SK_REPO/SK_DYLIB/SK_METALLIB/SK_GRANITE_GGUF, N, T (prefill len),
STEPS (decode steps), REPS, WARMUP, OUT_JSON.
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

GGUF   = os.environ["SK_GRANITE_GGUF"]
N      = int(os.environ.get("N", "8"))
T      = int(os.environ.get("T", "32"))
STEPS  = int(os.environ.get("STEPS", "64"))
REPS   = int(os.environ.get("REPS", "7"))
WARMUP = int(os.environ.get("WARMUP", "2"))
GAP    = float(os.environ.get("GAP", "0.3"))


def run_A(m, prompts):
    """N sequential single-stream requests; returns (decode_seconds, tokens)."""
    total = 0.0
    toks = 0
    for p in prompts:
        m.reset()
        cur = m.forward(p)  # production single-forward prefill (untimed)
        t0 = time.perf_counter()
        for _ in range(STEPS):
            cur = m.forward(np.array([cur], dtype=np.int32))
        total += time.perf_counter() - t0
        toks += STEPS
    return total, toks


def run_B(m, prompts):
    """One lockstep batched decode over N lanes; returns (seconds, tokens)."""
    m.reset()
    mat = np.stack(prompts)
    for t in range(T):  # lockstep tbt prefill (untimed)
        cur = m.forward_batched(mat[:, t])
    t0 = time.perf_counter()
    for _ in range(STEPS):
        cur = m.forward_batched(cur)
    dt = time.perf_counter() - t0
    return dt, N * STEPS


def main():
    cfg = Config(batch=N, seq_max=max(T, 64), cache_max=T + STEPS + 16)
    m = Granite(cfg)
    m.load_gguf(GGUF)
    rng = np.random.default_rng(11)
    prompts = [rng.integers(100, 50000, size=T).astype(np.int32)
               for _ in range(N)]

    for _ in range(WARMUP):
        run_A(m, prompts); time.sleep(GAP)
        run_B(m, prompts); time.sleep(GAP)

    a_tps, b_tps = [], []
    for r in range(REPS):
        ta, na = run_A(m, prompts); time.sleep(GAP)
        tb, nb = run_B(m, prompts); time.sleep(GAP)
        a_tps.append(na / ta)
        b_tps.append(nb / tb)
        print(f"rep {r}: A {na/ta:8.2f} agg tok/s | B {nb/tb:8.2f} agg tok/s",
              flush=True)

    a_med = float(np.median(a_tps)); b_med = float(np.median(b_tps))
    print(f"[win1] N={N} T={T} STEPS={STEPS}: A(seq)={a_med:.2f} "
          f"B(batched)={b_med:.2f} speedup={b_med/a_med:.3f}x")
    out = os.environ.get("OUT_JSON",
                         os.path.join(os.path.dirname(__file__),
                                      f"win1_n{N}.json"))
    with open(out, "w") as f:
        json.dump(dict(N=N, T=T, STEPS=STEPS, a_tps=a_tps, b_tps=b_tps,
                       a_med=a_med, b_med=b_med, speedup=b_med / a_med), f)
    print(f"[win1] wrote {out}")


if __name__ == "__main__":
    main()
