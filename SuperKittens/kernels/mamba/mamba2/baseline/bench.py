"""
Mamba-2 benchmark — Metal vs MLX (full block + SSM-only)
"""
import time, statistics
import mlx.core as mx
from ssm import selective_scan

from mamba2 import mamba2_block, bench_block as m2_bench_block


def bench_ssm(L=128, Ds=64, Dv=64, H=2, iters=20):
    B = 1
    Q = mx.random.normal((B, L, H, Ds), dtype=mx.float16) * 0.5
    K = mx.random.normal((B, L, H, Ds), dtype=mx.float16) * 0.5
    V = mx.random.normal((B, L, H, Dv), dtype=mx.float16) * 0.5
    A_log = mx.random.normal((B, L, H), dtype=mx.float32) * 0.01
    mx.eval(Q, K, V, A_log)

    for _ in range(5):
        mx.eval(selective_scan(Q, K, V, A_log))

    mx.synchronize()
    times = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(selective_scan(Q, K, V, A_log))
        mx.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)

    med = statistics.median(times)
    print(f"MLX Mamba-2 SSM only:  median={med:.3f}ms")
    return med


def bench_block(L=128, D=128, E=64, H=2, N=64, iters=20):
    return m2_bench_block(L, D, E, H, N, iters)


if __name__ == "__main__":
    print("=" * 55)
    print("Mamba-2 — MLX Baseline")
    print("=" * 55)
    t_ssm = bench_ssm()
    t_block = bench_block()
    print()
    print(f"  SSM:        {t_ssm:.3f}ms  (Metal: 1.49ms)")
    print(f"  Full block: {t_block:.3f}ms  (Metal: —)")
