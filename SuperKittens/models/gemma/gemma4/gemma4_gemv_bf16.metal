// gemma4_gemv_bf16.metal — M=1 matvec fast-paths for the bf16 decode hot path.
//
// Three kernels:
//   gemma4_gemv_bf16_m1     : y[1,N] = x[1,K] @ W[K,N]    (transB=0)  bf16 → bf16
//   gemma4_gemv_t_bf16_m1   : y[1,N] = x[1,K] @ W[N,K]^T  (transB=1)  bf16 → bf16
//   gemma4_gemv_bf16_fp32_m1: y[1,N] = x[1,K] @ W[K,N]    (transB=0)  bf16 → fp32
//
// Used in Gemma4 decode to skip the 32×64 tile GEMM at M=1 (which wastes 31/32
// rows). Mirrors qwen3's fp16 gemv_m1 / gemv_t_m1 (commit b77880f / 3f9f285).
//
#include <metal_stdlib>
using namespace metal;

namespace meow { namespace gemma4 { namespace gemv_bf16 {
enum : uint { BN = 128, BK = 256 };
} } }
using namespace meow::gemma4::gemv_bf16;

// --- y = x @ W, W is (K,N) row-major. One thread per output column. -------
[[host_name("gemma4_gemv_bf16_m1")]]
[[kernel]]
void gemma4_gemv_bf16_m1(
    device const bfloat* x [[buffer(0)]],   // (1, K)
    device const bfloat* W [[buffer(1)]],   // (K, N), row-major
    device       bfloat* y [[buffer(2)]],   // (1, N)
    constant uint&       N [[buffer(3)]],
    constant uint&       K [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const uint col0 = gid * BN;
    const uint col  = col0 + tid;

    threadgroup bfloat xs[BK];
    float acc = 0.0f;

    const uint K_main = (K / BK) * BK;
    for (uint k0 = 0; k0 < K_main; k0 += BK) {
        // Cooperative load x[k0:k0+BK] into TG memory.
        for (uint i = tid; i < BK; i += BN) {
            xs[i] = x[k0 + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (col < N) {
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
    if (col < N) {
        for (uint k = K_main; k < K; k++) {
            acc += float(x[k]) * float(W[(uint64_t)k*N + col]);
        }
        y[col] = bfloat(acc);
    }
}

// Same as above but writes fp32 outputs (used for PLE proj where bf16-trunc
// of the GEMM accumulator hurt accuracy on the high-gamma channels).
[[host_name("gemma4_gemv_bf16_fp32_m1")]]
[[kernel]]
void gemma4_gemv_bf16_fp32_m1(
    device const bfloat* x [[buffer(0)]],   // (1, K)
    device const bfloat* W [[buffer(1)]],   // (K, N), row-major
    device       float*  y [[buffer(2)]],   // (1, N) fp32
    constant uint&       N [[buffer(3)]],
    constant uint&       K [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const uint col0 = gid * BN;
    const uint col  = col0 + tid;

    threadgroup bfloat xs[BK];
    float acc = 0.0f;

    const uint K_main = (K / BK) * BK;
    for (uint k0 = 0; k0 < K_main; k0 += BK) {
        for (uint i = tid; i < BK; i += BN) xs[i] = x[k0 + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (col < N) {
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
    if (col < N) {
        for (uint k = K_main; k < K; k++) {
            acc += float(x[k]) * float(W[(uint64_t)k*N + col]);
        }
        y[col] = acc;
    }
}

// --- y = x @ W^T, W is (N,K) row-major. One thread per output row. --------
[[host_name("gemma4_gemv_t_bf16_m1")]]
[[kernel]]
void gemma4_gemv_t_bf16_m1(
    device const bfloat* x [[buffer(0)]],   // (1, K)
    device const bfloat* W [[buffer(1)]],   // (N, K) row-major
    device       bfloat* y [[buffer(2)]],   // (1, N)
    constant uint&       N [[buffer(3)]],
    constant uint&       K [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const uint row0 = gid * BN;
    const uint row  = row0 + tid;

    threadgroup bfloat xs[BK];
    float acc = 0.0f;

    const uint K_main = (K / BK) * BK;
    for (uint k0 = 0; k0 < K_main; k0 += BK) {
        for (uint i = tid; i < BK; i += BN) xs[i] = x[k0 + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < N) {
            device const bfloat* wrow = W + (uint64_t)row * K + k0;
            for (uint kk = 0; kk < BK; kk += 8) {
                float a0 = float(wrow[kk+0]) * float(xs[kk+0]);
                float a1 = float(wrow[kk+1]) * float(xs[kk+1]);
                float a2 = float(wrow[kk+2]) * float(xs[kk+2]);
                float a3 = float(wrow[kk+3]) * float(xs[kk+3]);
                float a4 = float(wrow[kk+4]) * float(xs[kk+4]);
                float a5 = float(wrow[kk+5]) * float(xs[kk+5]);
                float a6 = float(wrow[kk+6]) * float(xs[kk+6]);
                float a7 = float(wrow[kk+7]) * float(xs[kk+7]);
                acc += (a0+a1+a2+a3+a4+a5+a6+a7);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < N) {
        device const bfloat* wrow = W + (uint64_t)row * K;
        for (uint k = K_main; k < K; k++) {
            acc += float(x[k]) * float(wrow[k]);
        }
        y[row] = bfloat(acc);
    }
}
