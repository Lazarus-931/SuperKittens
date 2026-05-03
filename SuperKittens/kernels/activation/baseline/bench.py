"""
Activation benchmark — MLX baselines.
"""

import sys
import time
import statistics
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.activation.baseline.baseline import gelu, silu, relu


def bench_mlx(fn, rows, cols, dtype=mx.float16, warmup=5, iters=20):
    x = mx.random.normal((1, rows, cols), dtype=dtype) * 0.5
    mx.eval(x)

    for _ in range(warmup):
        mx.eval(fn(x))

    mx.synchronize()
    times = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(fn(x))
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)

    med = statistics.median(times)
    gb = (2.0 * rows * cols * 2) / 1e9  # read + write, fp16
    gbps = gb / (med * 1e-6)
    return med, gbps


def main():
    configs = [
        (512, 1024), (1024, 1024), (2048, 1024),
        (512, 4096), (1024, 4096), (2048, 4096),
    ]

    activations = {"gelu": gelu, "silu": silu, "relu": relu}

    for name, fn in activations.items():
        print(f"\n--- {name} ---")
        print(f"{'rows':>6} {'cols':>6}   {'us':>8}   {'GB/s':>8}")
        for rows, cols in configs:
            us, gbps = bench_mlx(fn, rows, cols)
            print(f"{rows:6} {cols:6}   {us:8.1f}   {gbps:8.1f}")


if __name__ == "__main__":
    main()
