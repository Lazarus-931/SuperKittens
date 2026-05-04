#!/usr/bin/env python3
"""MLX baseline for conv — mirrors configs in bench_conv.cpp."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx

WARMUP = 5
ITERS  = 20

# (H, W, C_in, C_out, K) — same as bench_conv.cpp
SHAPES = [
    (56,  56,  64,  64,  3),
    (28,  28, 128, 128,  3),
    (14,  14, 256, 256,  3),
    (56,  56,  64, 128,  1),
]


def _bench_conv2d(H, W, C_in, C_out, K) -> float:
    N = 1
    x = mx.random.normal((N, H, W, C_in)).astype(mx.float16)
    w = mx.random.normal((C_out, K, K, C_in)).astype(mx.float16)
    b = mx.random.normal((C_out,)).astype(mx.float16)
    mx.eval(x, w, b)

    def run():
        return mx.conv2d(x, w, stride=(1, 1), padding=(0, 0)) + b

    for _ in range(WARMUP):
        mx.eval(run())

    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        mx.eval(run())
        times.append((time.perf_counter() - t0) * 1e6)

    times.sort()
    return times[len(times) // 2]


def main() -> None:
    print(f"{'shape':<40}  {'us':>8}  {'GFLOPS':>8}")
    print(f"{'----------------------------------------':<40}  {'--------':>8}  {'--------':>8}")

    for H, W, C_in, C_out, K in SHAPES:
        us = _bench_conv2d(H, W, C_in, C_out, K)
        H_out, W_out = H - K + 1, W - K + 1
        flops = 2.0 * H_out * W_out * K * K * C_in * C_out
        gflops = flops / (us * 1e3)
        label = f"H={H} W={W} C_in={C_in} C_out={C_out} K={K}"
        print(f"{label:<40}  {us:>8.1f}  {gflops:>8.1f}")


if __name__ == "__main__":
    main()
