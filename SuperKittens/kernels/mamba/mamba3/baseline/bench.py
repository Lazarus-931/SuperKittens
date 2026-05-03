"""
Mamba-3 benchmark — Metal vs MLX (SISO + MIMO, full block + SSM-only)
"""
import time, statistics
import mlx.core as mx
from ssm import mamba3_ssm
from mamba3 import mamba3_block, _make_block_inputs


def bench_ssm(L=128, DQ=64, DV=64, H=2, CS=32, iters=20):
    B = 1
    q = mx.random.normal((B, H, L, DQ), dtype=mx.float16) * 0.5
    k = mx.random.normal((B, H, L, DQ), dtype=mx.float16) * 0.5
    v = mx.random.normal((B, H, L, DV), dtype=mx.float16) * 0.5
    a = mx.random.normal((B, H, L), dtype=mx.float16) * 0.01
    b = mx.random.normal((B, H, L), dtype=mx.float16) * 0.01
    angle = mx.zeros((B, H, L, DQ // 2), dtype=mx.float16)
    mx.eval(q, k, v, a, b, angle)

    for _ in range(5):
        mx.eval(mamba3_ssm(q, k, v, a, b, angle, chunk_size=CS))

    mx.synchronize()
    times = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(mamba3_ssm(q, k, v, a, b, angle, chunk_size=CS))
        mx.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)
    return statistics.median(times)


def bench_block(L=128, D=128, DQ=64, DV=64, H=2, CS=32, iters=20):
    B = 1
    x, ipw, ipb, nq, nk, opw, opb, aw = _make_block_inputs(B, L, D, DQ, DV, H)
    mx.eval(x, ipw, ipb, nq, nk, opw, opb, aw if aw is not None else mx.zeros(1))

    for _ in range(5):
        mx.eval(mamba3_block(x, ipw, ipb, nq, nk, opw, opb, aw,
                             DQ=DQ, DV=DV, H=H, CS=CS))

    mx.synchronize()
    times = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(mamba3_block(x, ipw, ipb, nq, nk, opw, opb, aw,
                             DQ=DQ, DV=DV, H=H, CS=CS))
        mx.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)
    return statistics.median(times)


def main():
    print("=" * 65)
    print("Mamba-3 — MLX Baseline  (SISO + MIMO)")
    print("=" * 65)

    # MIMO: multi-dimensional per head  (the standard benchmark)
    print("\n── MIMO  (DQ=64, DV=64, H=4) ──")
    t_ssm = bench_ssm(L=128, DQ=64, DV=64, H=4)
    t_block = bench_block(L=128, D=128, DQ=64, DV=64, H=4)
    print(f"  SSM:       {t_ssm:.3f}ms  (Metal: 0.91ms)")
    print(f"  Full block: {t_block:.3f}ms  (Metal: 1.1ms)")

    # SISO: scalar per head  (faster, limited to L ≤ 1024)
    print("\n── SISO  (DQ=4, DV=4, H=4) ──")
    t_ssm_s = bench_ssm(L=128, DQ=4, DV=4, H=4)
    t_block_s = bench_block(L=128, D=128, DQ=4, DV=4, H=4)
    print(f"  SSM:       {t_ssm_s:.3f}ms")
    print(f"  Full block: {t_block_s:.3f}ms")


if __name__ == "__main__":
    main()
