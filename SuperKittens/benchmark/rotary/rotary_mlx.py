#!/usr/bin/env python3
"""MLX baseline for RoPE — mirrors configs in bench_rotary.cpp."""
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

# (B, H, L, d) — same as bench_rotary.cpp
CONFIGS = [
    (1, 8,  512,  64),
    (1, 8, 1024,  64),
    (1, 8, 2048,  64),
    (1, 8,  512, 128),
    (1, 8, 1024, 128),
    (1, 8, 2048, 128),
]


def _rope(q: mx.array, k: mx.array) -> tuple[mx.array, mx.array]:
    # mx.fast.rope expects (batch, seq, n_heads, head_dim); we have (B, H, L, d)
    # transpose to (B, L, H, d), apply, transpose back
    dims = q.shape[-1]
    q_t = mx.transpose(q, (0, 2, 1, 3))
    k_t = mx.transpose(k, (0, 2, 1, 3))
    q_r = mx.fast.rope(q_t, dims, traditional=False, base=10000.0, scale=1.0, offset=0)
    k_r = mx.fast.rope(k_t, dims, traditional=False, base=10000.0, scale=1.0, offset=0)
    return mx.transpose(q_r, (0, 2, 1, 3)), mx.transpose(k_r, (0, 2, 1, 3))


def _bench(B, H, L, d) -> float:
    mx.random.seed(42)
    q = mx.random.normal((B, H, L, d)).astype(mx.float16)
    k = mx.random.normal((B, H, L, d)).astype(mx.float16)
    mx.eval(q, k)

    for _ in range(WARMUP):
        mx.eval(_rope(q, k))

    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        mx.eval(_rope(q, k))
        times.append((time.perf_counter() - t0) * 1e6)
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    print(f"{'B':>3} {'H':>3} {'L':>5} {'d':>4}  {'us':>10}  {'GB/s':>8}")
    print("-" * 44)

    for B, H, L, d in CONFIGS:
        us = _bench(B, H, L, d)
        # read Q+K + write Q+K = 4 tensors × B*H*L*d × 2 bytes (fp16)
        gb = 4.0 * B * H * L * d * 2 / 1e9
        print(f"{B:>3} {H:>3} {L:>5} {d:>4}  {us:>10.1f}  {gb / (us * 1e-6):>8.2f}")


if __name__ == "__main__":
    main()
