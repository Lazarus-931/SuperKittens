#!/usr/bin/env python3
"""MLX baseline for activations — mirrors configs in act_bench.cpp."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx
from kernels.activation.baseline.baseline import gelu, silu, relu

WARMUP = 5
ITERS  = 20

CONFIGS = [
    (512,  1024), (1024, 1024), (2048, 1024),
    (512,  4096), (1024, 4096), (2048, 4096),
]

ACTIVATIONS = {"gelu": gelu, "silu": silu, "relu": relu}


def _bench(fn, rows, cols) -> float:
    x = mx.random.normal((rows, cols)).astype(mx.float16)
    mx.eval(x)
    for _ in range(WARMUP):
        mx.eval(fn(x))
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        mx.eval(fn(x))
        times.append((time.perf_counter() - t0) * 1e6)
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    print(f"{'kernel':<6}  {'rows':>6} {'cols':>6}  {'us':>8}  {'GB/s':>8}")
    print(f"{'------':<6}  {'------':>6} {'------':>6}  {'--------':>8}  {'--------':>8}")

    for name, fn in ACTIVATIONS.items():
        for rows, cols in CONFIGS:
            us = _bench(fn, rows, cols)
            gb = (2.0 * rows * cols * 2) / 1e9  # read + write, fp16
            print(f"{name:<6}  {rows:>6} {cols:>6}  {us:>8.1f}  {gb / (us * 1e-6):>8.1f}")
        print()


if __name__ == "__main__":
    main()
