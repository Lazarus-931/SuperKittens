"""
Attention in Triton — naive and fused variants.
Runs on CUDA only (Triton doesn't support Metal).
Use this to compare against SuperKittens when testing on NVIDIA hardware.

Naive: separate matmuls + softmax
Fused: single kernel, Q*K^T + softmax + scores*V in one pass (not FlashAttention)
"""

import torch
import triton
import triton.language as tl
import time
import argparse


# ── Naive (Triton) — just for reference, same as PyTorch ──

def naive_attention_triton(Q, K, V):
    d = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / (d ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    return attn @ V


# ── Fused Attention Kernel (Triton) ──
# Single kernel: loads Q row, streams K/V, computes softmax online
# This is NOT FlashAttention — it still does O(N) work per Q row in one block

@triton.jit
def fused_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    N: tl.constexpr, d: tl.constexpr,
    stride_qn, stride_kn, stride_vn, stride_on,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)

    # Load one Q row (1 x d)
    q_offsets = row * stride_qn + tl.arange(0, d)
    q = tl.load(Q_ptr + q_offsets)

    # Accumulate softmax and weighted V
    m_prev = float("-inf")
    l_prev = 0.0
    acc = tl.zeros([d], dtype=tl.float32)

    for start in range(0, N, BLOCK_N):
        # Load K block (BLOCK_N x d)
        k_offsets = (start + tl.arange(0, BLOCK_N))[:, None] * stride_kn + tl.arange(0, d)[None, :]
        k = tl.load(K_ptr + k_offsets, mask=(start + tl.arange(0, BLOCK_N))[:, None] < N)

        # Compute scores: (1 x d) @ (d x BLOCK_N) = (BLOCK_N,)
        scores = tl.sum(q[None, :] * k, axis=1) / tl.sqrt(float(d))

        # Online softmax
        m_new = tl.maximum(m_prev, tl.max(scores, axis=0))
        correction = tl.exp(m_prev - m_new)
        p = tl.exp(scores - m_new)
        l_new = correction * l_prev + tl.sum(p, axis=0)

        # Load V block and accumulate
        v_offsets = (start + tl.arange(0, BLOCK_N))[:, None] * stride_vn + tl.arange(0, d)[None, :]
        v = tl.load(V_ptr + v_offsets, mask=(start + tl.arange(0, BLOCK_N))[:, None] < N)

        acc = correction * acc + tl.sum(p[:, None] * v, axis=0)
        m_prev = m_new
        l_prev = l_new

    # Normalize
    acc = acc / l_prev

    # Store output row
    o_offsets = row * stride_on + tl.arange(0, d)
    tl.store(O_ptr + o_offsets, acc.to(tl.float16))


def fused_attention(Q, K, V):
    B_N, d = Q.shape[0] * Q.shape[1] if Q.dim() == 3 else Q.shape[0], Q.shape[-1]
    Q_flat = Q.reshape(-1, d)
    K_flat = K.reshape(-1, d) if K.dim() == 3 else K
    V_flat = V.reshape(-1, d) if V.dim() == 3 else V
    N = Q_flat.shape[0]
    O = torch.empty_like(Q_flat)

    BLOCK_N = min(64, N)
    grid = (N,)
    fused_attention_kernel[grid](
        Q_flat, K_flat, V_flat, O,
        N, d,
        Q_flat.stride(0), K_flat.stride(0), V_flat.stride(0), O.stride(0),
        BLOCK_N=BLOCK_N,
    )
    return O.reshape(Q.shape)


def benchmark(B, N, d, dtype=torch.float16, iters=20):
    device = "cuda"
    Q = torch.randn(B, N, d, dtype=dtype, device=device)
    K = torch.randn(B, N, d, dtype=dtype, device=device)
    V = torch.randn(B, N, d, dtype=dtype, device=device)

    # ── Naive ──
    torch.cuda.synchronize()
    for _ in range(3):
        _ = naive_attention_triton(Q, K, V)
    torch.cuda.synchronize()

    times_naive = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = naive_attention_triton(Q, K, V)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_naive.append((t1 - t0) * 1e6)

    times_naive.sort()
    med_naive = times_naive[len(times_naive) // 2]

    # ── Fused ──
    torch.cuda.synchronize()
    for _ in range(3):
        _ = fused_attention(Q, K, V)
    torch.cuda.synchronize()

    times_fused = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = fused_attention(Q, K, V)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_fused.append((t1 - t0) * 1e6)

    times_fused.sort()
    med_fused = times_fused[len(times_fused) // 2]

    flops = 4 * B * N * N * d
    print(f"B={B} N={N} d={d}")
    print(f"  Naive:  {med_naive:.0f} us, {flops / (med_naive * 1e3):.1f} GFLOPS")
    print(f"  Fused:  {med_fused:.0f} us, {flops / (med_fused * 1e3):.1f} GFLOPS")
    print(f"  Speedup: {med_naive / med_fused:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--N", type=int, default=512)
    parser.add_argument("--d", type=int, default=64)
    args = parser.parse_args()

    benchmark(args.B, args.N, args.d)
