"""SwiGLU MLX benchmark."""
import sys, time, statistics
from pathlib import Path
import mlx.core as mx

BASELINE = Path(__file__).resolve().parents[2] / "kernels" / "swiglu" / "baseline"
sys.path.insert(0, str(BASELINE))
from swiglu import swiglu


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
    print("=" * 55)
    print("SwiGLU — MLX Baseline")
    print("=" * 55)

    shapes = [
        (512, 2048), (1024, 4096), (2048, 8192),
        (512, 1024), (2048, 4096), (4096, 14336),
    ]
    print(f"{'shape':>16}  {'us':>8}  {'GB/s':>8}")
    for rows, d2 in shapes:
        x = mx.random.normal((rows, d2), dtype=mx.float16) * 0.5
        mx.eval(x)
        us = bench(lambda: swiglu(x))
        bw = (rows * d2 * 3) / (us * 1e3)  # GB/s: read (rows,2d) + write (rows,d)
        print(f"  {rows}x{d2:<10}  {us:>8.1f}  {bw:>8.1f}")


if __name__ == "__main__":
    main()
