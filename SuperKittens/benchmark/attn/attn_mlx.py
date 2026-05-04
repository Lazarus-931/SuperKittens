#!/usr/bin/env python3
"""MLX baseline for attention — mirrors configs in bench_attn.cpp."""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx

SEQLENS = [256, 512, 1024, 2048]
WARMUP  = 5
ITERS   = 30


def _bench(fn, *args) -> float:
    for _ in range(WARMUP):
        mx.eval(fn(*args))
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        mx.eval(fn(*args))
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


def attn_noncausal(Q, K, V):
    return mx.fast.scaled_dot_product_attention(Q, K, V, scale=1.0 / math.sqrt(Q.shape[-1]))


def attn_causal(Q, K, V):
    return mx.fast.scaled_dot_product_attention(Q, K, V, scale=1.0 / math.sqrt(Q.shape[-1]), mask="causal")


def main() -> None:
    print(f"{'kernel':<22}  {'seq':>6}  {'ms':>8}  {'TFLOPS':>8}")
    print(f"{'----------------------':<22}  {'------':>6}  {'--------':>8}  {'--------':>8}")

    configs = [
        ("fa_causal_64",    64,  True,  attn_causal),
        ("fa_noncausal_64", 64,  False, attn_noncausal),
        ("mha_causal",      128, True,  attn_causal),
        ("mha_noncausal",   128, False, attn_noncausal),
    ]

    for name, dim, causal, fn in configs:
        for seq in SEQLENS:
            mx.random.seed(42)
            Q = mx.random.normal((1, 1, seq, dim)).astype(mx.float16)
            K = mx.random.normal((1, 1, seq, dim)).astype(mx.float16)
            V = mx.random.normal((1, 1, seq, dim)).astype(mx.float16)
            mx.eval(Q, K, V)

            ms = _bench(fn, Q, K, V)

            flops = (2.0 * dim + 2.5) * seq * seq * (0.5 if causal else 1.0)
            tflops = flops / (ms * 1e9)
            print(f"{name:<22}  {seq:>6}  {ms:>8.3f}  {tflops:>8.4f}")
        print()


if __name__ == "__main__":
    main()
