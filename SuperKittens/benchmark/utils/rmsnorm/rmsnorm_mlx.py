"""RMSNorm MLX benchmark."""
import sys, time, statistics
from pathlib import Path
import mlx.core as mx

BASELINE = Path(__file__).resolve().parents[3] / "kernels" / "utils" / "layernorm" / "baseline"
sys.path.insert(0, str(BASELINE))
from layernorm import rmsnorm


def bench(fn, args=(), iters=20):
    for _ in range(5):
        mx.eval(fn(*args))
    mx.synchronize()
    times = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(fn(*args))
        mx.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return statistics.median(times)


def main():
    print("=" * 45)
    print("RMSNorm — MLX Baseline")
    print("=" * 45)

    shapes = [(512, 1024), (1024, 1024), (2048, 2048), (4096, 4096),
              (1024, 4096), (2048, 8192)]
    print(f"{'shape':>12}  {'us':>8}  {'GB/s':>8}")
    for r, d in shapes:
        x = mx.random.normal((r, d), dtype=mx.float16) * 0.5
        w = mx.random.normal((d,), dtype=mx.float16)
        mx.eval(x, w)
        us = bench(rmsnorm, (x, w))
        bw = (2 * r * d * 2) / (us * 1e3)
        print(f"  {r:>5}x{d:<5}  {us:>8.1f}  {bw:>8.1f}")


if __name__ == "__main__":
    main()
