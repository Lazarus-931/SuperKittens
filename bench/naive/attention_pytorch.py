"""
Naive attention in PyTorch — baseline reference.
Computes: O = softmax(Q @ K^T / sqrt(d)) @ V
"""

import torch
import torch.nn.functional as F
import time
import argparse


def naive_attention(Q, K, V):
    """Standard attention — materializes full N×N scores matrix."""
    d = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / (d ** 0.5)
    attn = F.softmax(scores, dim=-1)
    return attn @ V


def benchmark(B, N, d, dtype=torch.float16, device="mps", iters=20):
    Q = torch.randn(B, N, d, dtype=dtype, device=device)
    K = torch.randn(B, N, d, dtype=dtype, device=device)
    V = torch.randn(B, N, d, dtype=dtype, device=device)

    # Warmup
    for _ in range(3):
        _ = naive_attention(Q, K, V)
    if device == "mps":
        torch.mps.synchronize()

    # Timed
    times = []
    for _ in range(iters):
        if device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        O = naive_attention(Q, K, V)
        if device == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    times.sort()
    median_us = times[len(times) // 2]
    flops = 4 * B * N * N * d  # 2 matmuls of (N,d)x(d,N) and (N,N)x(N,d)
    gflops = flops / (median_us * 1e3)

    print(f"Naive Attention (PyTorch on {device})")
    print(f"  B={B} N={N} d={d} dtype={dtype}")
    print(f"  Median: {median_us:.0f} us")
    print(f"  GFLOPS: {gflops:.1f}")
    return median_us


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--N", type=int, default=512)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    benchmark(args.B, args.N, args.d, device=args.device)
