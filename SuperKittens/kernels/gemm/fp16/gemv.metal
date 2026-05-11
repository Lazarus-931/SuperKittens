//
// v2_gemv.metal — Qwen3 decode matvec (M=1) for fp16
//
// Specialization of gemm_fp16 for M=1: y[1xN] = x[1xK] @ W[KxN] (transB=false).
// SK's gemm_fp16 uses BM=32; at M=1 it wastes 31/32 rows, dispatches needless tiles.
//
// Design: each threadgroup computes BN consecutive output columns. 128 threads
// (4 simdgroups × 32 lanes). x is broadcast through threadgroup memory; each
// thread accumulates one column dot product using vectorized half8 loads of W.
//
// Grid: (ceil(N/BN), 1, 1). Threadgroup: (BN, 1, 1) → BN must == 128.
//
#include <metal_stdlib>
using namespace metal;

enum : uint { BN = 128, BK = 256 };

[[host_name("gemv_fp16_m1")]]
[[kernel]]
void gemv_fp16_m1(
    device const half* x   [[buffer(0)]],   // (1, K)
    device const half* W   [[buffer(1)]],   // (K, N), row-major
    device       half* y   [[buffer(2)]],   // (1, N)
    constant uint&     N   [[buffer(3)]],
    constant uint&     K   [[buffer(4)]],
    uint  gid  [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]])
{
    const uint col0 = gid * BN;
    const uint col  = col0 + tid;          // this thread owns column `col`

    threadgroup half xs[BK];
    float acc = 0.0f;

    // Vectorized W load helpers — we load via half8 along the K dimension.
    // W is (K,N) row-major: W[k,col] = W[k*N + col]. Strided in K, so we walk
    // BK rows × broadcast x[k], accumulating into our column.
    //
    // Bulk path: K processed in chunks of BK; tail handled scalar.
    const uint K_main = (K / BK) * BK;

    for (uint k0 = 0; k0 < K_main; k0 += BK) {
        // Cooperatively load x[k0 : k0+BK] into shared memory via half8 loads.
        // 128 threads × 1 half8 = 128 halves per pass; we need BK=256 → 2 passes.
        for (uint i = tid; i < BK / 2; i += BN) {
            reinterpret_cast<threadgroup half2*>(xs)[i] =
                reinterpret_cast<const device half2*>(x + k0)[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each thread walks BK rows of column `col`, accumulating.
        // W stride along K is N. Use half8 loads when col is 8-aligned & N%8==0.
        // For Qwen3 N is 10240, 27392, 5120, 8192 — all %8==0. col may not be 8-aligned
        // (tid varies), so scalar in N: each thread reads its own column scalar from W.
        if (col < N) {
            // Unroll inner K by 8 for ILP.
            for (uint kk = 0; kk < BK; kk += 8) {
                float a0 = float(W[(k0+kk+0)*N + col]) * float(xs[kk+0]);
                float a1 = float(W[(k0+kk+1)*N + col]) * float(xs[kk+1]);
                float a2 = float(W[(k0+kk+2)*N + col]) * float(xs[kk+2]);
                float a3 = float(W[(k0+kk+3)*N + col]) * float(xs[kk+3]);
                float a4 = float(W[(k0+kk+4)*N + col]) * float(xs[kk+4]);
                float a5 = float(W[(k0+kk+5)*N + col]) * float(xs[kk+5]);
                float a6 = float(W[(k0+kk+6)*N + col]) * float(xs[kk+6]);
                float a7 = float(W[(k0+kk+7)*N + col]) * float(xs[kk+7]);
                acc += (a0+a1+a2+a3+a4+a5+a6+a7);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Tail
    if (col < N) {
        for (uint k = K_main; k < K; k++) {
            acc += float(x[k]) * float(W[k*N + col]);
        }
        y[col] = half(acc);
    }
}
