// 2D-tile M=1 GEMV for transB=1: y[1,N] = x[1,K] @ W[N,K]^T.
// W[N,K] row-major (ld=K); same layout as gemv_t_fp16_m1 — drop-in PSO swap.
// v9 config from temp/gemv_2d_tile lab: SG_ROWS=16, TOR=4, TOC=8, smem-staged x.
//
// WHY: x is reused across TOR=4 outputs per thread (reduces W-reread amortization
// vs. 1-out-per-thread baseline) and staged in smem once per TG (one global load
// of x per TG instead of once per row). Lab result on K=4096 N=151936: 1.71×.

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

[[host_name("gemv_t_fp16_2dtile_m1")]]
[[kernel, max_total_threads_per_threadgroup(512)]]
void gemv_t_fp16_2dtile_m1(
    device const half* x  [[buffer(0)]],   // (1, K)
    device const half* W  [[buffer(1)]],   // (N, K) row-major, ld=K
    device       half* y  [[buffer(2)]],   // (1, N)
    constant uint&     N  [[buffer(3)]],
    constant uint&     K  [[buffer(4)]],
    uint3 tg  [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]])
{
    constexpr uint SG_ROWS = 16;
    constexpr uint SG_COLS = 32;
    constexpr uint TOR     = 4;
    constexpr uint TOC     = 8;
    constexpr uint K_PER_ITER = SG_COLS * TOC;   // 256
    constexpr uint OUT_ROWS_PER_TG = SG_ROWS * TOR;  // 64

    const uint tx = tid.x;       // 0..31 (K lane)
    const uint ty = tid.y;       // 0..SG_ROWS-1 (row group)

    uint row_base = tg.x * OUT_ROWS_PER_TG + ty * TOR;
    if (row_base >= N) return;
    if (row_base + TOR > N) row_base = N - TOR;

    threadgroup half x_smem[K_PER_ITER];

    float acc[TOR];
    #pragma unroll
    for (uint r = 0; r < TOR; ++r) acc[r] = 0.0f;

    const uint K_main = (K / K_PER_ITER) * K_PER_ITER;
    const uint tid_lin = ty * SG_COLS + tx;
    const uint nthreads = SG_ROWS * SG_COLS;

    for (uint k0 = 0; k0 < K_main; k0 += K_PER_ITER) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid_lin; i < K_PER_ITER; i += nthreads) {
            x_smem[i] = x[k0 + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float xv[TOC];
        #pragma unroll
        for (uint c = 0; c < TOC; ++c) xv[c] = (float)x_smem[tx * TOC + c];

        #pragma unroll
        for (uint r = 0; r < TOR; ++r) {
            device const half* wrow = W + (size_t)(row_base + r) * K + k0 + tx * TOC;
            float s = 0.0f;
            #pragma unroll
            for (uint c = 0; c < TOC; ++c) s += (float)wrow[c] * xv[c];
            acc[r] += s;
        }
    }

    if (K_main < K) {
        #pragma unroll
        for (uint c = 0; c < TOC; ++c) {
            uint k = K_main + tx * TOC + c;
            if (k < K) {
                float xv = (float)x[k];
                #pragma unroll
                for (uint r = 0; r < TOR; ++r) {
                    acc[r] += (float)W[(size_t)(row_base + r) * K + k] * xv;
                }
            }
        }
    }

    #pragma unroll
    for (uint r = 0; r < TOR; ++r) {
        float v = acc[r];
        v += simd_shuffle_down(v, 16);
        v += simd_shuffle_down(v, 8);
        v += simd_shuffle_down(v, 4);
        v += simd_shuffle_down(v, 2);
        v += simd_shuffle_down(v, 1);
        acc[r] = v;
    }
    if (tx == 0) {
        #pragma unroll
        for (uint r = 0; r < TOR; ++r) y[row_base + r] = (half)acc[r];
    }
}
