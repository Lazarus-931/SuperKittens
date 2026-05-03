"""
RoPE benchmark — Metal vs MLX
"""

import math
import sys
import time
import statistics
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.rotary.baseline import rope as mlx_rope


def rope_cpu(q: np.ndarray, k: np.ndarray, base: float = 10000.0) -> tuple[np.ndarray, np.ndarray]:
    """CPU reference."""
    *_, seq, dim = q.shape
    hd = dim // 2
    freq = 1.0 / (base ** (np.arange(0, hd, dtype=np.float32) * 2.0 / dim))
    theta = np.arange(seq, dtype=np.float32)[:, None] * freq[None, :]
    cos = np.cos(theta).astype(np.float32)
    sin = np.sin(theta).astype(np.float32)

    q_out = np.zeros_like(q)
    k_out = np.zeros_like(k)
    q0, q1 = q[..., :hd], q[..., hd:]
    k0, k1 = k[..., :hd], k[..., hd:]

    q_out[..., :hd] = q0 * cos - q1 * sin
    q_out[..., hd:] = q0 * sin + q1 * cos
    k_out[..., :hd] = k0 * cos - k1 * sin
    k_out[..., hd:] = k0 * sin + k1 * cos
    return q_out, k_out


def err_stats(got: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    diff = np.abs(got - ref)
    l2 = np.linalg.norm(diff.ravel()) / (np.linalg.norm(ref.ravel()) + 1e-12)
    return float(l2), float(diff.max())


def bench_mlx(q: mx.array, k: mx.array, warmup=5, iters=20) -> float:
    for _ in range(warmup):
        mx.eval(mlx_rope(q, k))
    mx.synchronize()
    times = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(mlx_rope(q, k))
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(times)  # us


def main():
    configs = [
        (1, 8, 512, 64),
        (1, 8, 1024, 64),
        (1, 8, 2048, 64),
        (1, 8, 512, 128),
        (1, 8, 1024, 128),
        (1, 8, 2048, 128),
    ]

    print(f"{'B':>3} {'H':>3} {'L':>5} {'d':>4}  {'MLX(us)':>10}  {'GB/s':>8}")
    print("-" * 52)

    for B, H, L, d in configs:
        mx.random.seed(42)
        shape = (B, H, L, d)
        q_mlx = mx.random.normal(shape, dtype=mx.float16) * 0.5
        k_mlx = mx.random.normal(shape, dtype=mx.float16) * 0.5
        mx.eval(q_mlx, k_mlx)

        t = bench_mlx(q_mlx, k_mlx)

        # Verify accuracy
        q_np, k_np = np.array(q_mlx), np.array(k_mlx)
        q_ref, k_ref = rope_cpu(q_np, k_np)
        q_rot, k_rot = mlx_rope(q_mlx, k_mlx)
        l2_q, _ = err_stats(np.array(q_rot), q_ref)
        l2_k, _ = err_stats(np.array(k_rot), k_ref)

        # bytes: read Q+K (fp16) + write Q+K (fp16) = 4 * B*H*L*d bytes
        gb = 4.0 * B * H * L * d / 1e9
        gbps = gb / (t * 1e-6)

        print(f"{B:3} {H:3} {L:5} {d:4}  {t:10.1f}  {gbps:8.2f}  l2_q={l2_q:.6f} l2_k={l2_k:.6f}")


if __name__ == "__main__":
    main()
