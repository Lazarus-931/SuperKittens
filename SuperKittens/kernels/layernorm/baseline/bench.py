#!/usr/bin/env python3
"""layernorm bench.py — SuperKittens vs MLX benchmark"""
import sys, time, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline.layernorm_mlx import layernorm, rmsnorm

import mlx.core as mx

def bench(fn, *args, iters=20):
    for _ in range(5): mx.eval(fn(*args))
    mx.synchronize()
    times = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(fn(*args))
        mx.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    med = statistics.median(times)
    return med

def main():
    print("=" * 60)
    print("LayerNorm / RMSNorm — SuperKittens vs MLX")
    print("=" * 60)

    for name, fn in [("layernorm", layernorm), ("rmsnorm", rmsnorm)]:
        for shape in [(1024, 1024), (2048, 2048), (4096, 4096)]:
            x = mx.random.normal(shape, dtype=mx.float16) * 0.5
            w = mx.random.normal((shape[1],), dtype=mx.float16)
            b = mx.random.normal((shape[1],), dtype=mx.float16) if fn == layernorm else None
            mx.eval(x, w, b if b is not None else mx.zeros(1))

            args = (fn, x, w, b) if fn == layernorm else (fn, x, w)
            med = bench(*args)

            elem = shape[0] * shape[1]
            bw = (2 * elem * 2) / (med * 1e3)  # read + write fp16, GB/s
            print(f"  {name:12s}  {shape[0]:5d}x{shape[1]:5d}  MLX={med:.1f}us  {bw:.1f} GB/s")

    # SuperKittens Metal numbers (from benchmark harness)
    print()
    print("  Metal layernorm:  59 GB/s (half4, SIMD-group, zero-barrier)")
    print("  Metal rmsnorm:    pending C++ bench")

if __name__ == "__main__":
    main()
