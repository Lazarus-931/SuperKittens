#!/usr/bin/env python3


import sys
import time
import math

import mlx.core as mx


def causal_attention(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    """Manual causal scaled dot-product attention."""
    d = q.shape[-1]
    scores = mx.matmul(q, mx.swapaxes(k, -1, -2)) * (1.0 / math.sqrt(d))
    q_len = q.shape[-2]
    k_len = k.shape[-2]
    mask = mx.arange(q_len)[:, None] >= mx.arange(k_len)[None, :]
    scores = mx.where(mask, scores, -1e30)
    probs = mx.softmax(scores, axis=-1)
    return mx.matmul(probs, v)


def causal_attention_fast(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    """MLX fast SDPA with causal mask (if available)."""
    d = q.shape[-1]
    n = q.shape[-2]
    mask = mx.tril(mx.ones((n, n)))
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=1.0 / math.sqrt(d), mask=mask
    )


def bench(fn, q, k, v, warmup=5, iters=20):
    for _ in range(warmup):
        mx.eval(fn(q, k, v))

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn(q, k, v))
        times.append((time.perf_counter() - t0) * 1e6)

    times.sort()
    return times[len(times) // 2]  # median us


if __name__ == "__main__":
    seq   = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    d     = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    heads = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 20

    shape = (1, heads, seq, d)
    Q = mx.random.normal(shape).astype(mx.float16)
    K = mx.random.normal(shape).astype(mx.float16)
    V = mx.random.normal(shape).astype(mx.float16)
    mx.eval(Q, K, V)

    # Manual causal
    t_manual = bench(lambda q, k, v: causal_attention(q, k, v), Q, K, V, iters=iters)

    # Fast SDPA with mask
    try:
        t_fast = bench(lambda q, k, v: causal_attention_fast(q, k, v), Q, K, V, iters=iters)
    except Exception:
        t_fast = None

    # FLOPs (causal: half the QK^T ops)
    flops = (2.0 * d + 2.5) * seq * seq * heads
    tflops_manual = flops / (t_manual * 1e6)
    tflops_fast = flops / (t_fast * 1e6) if t_fast else 0.0

    out = f"{t_manual:.0f},{tflops_manual:.2f}"
    if t_fast:
        out += f",{t_fast:.0f},{tflops_fast:.2f}"
    print(out)
