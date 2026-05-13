// gemv_t.metal — M=1 matvec for transB=1: y[1,N] = x[1,K] @ W[N,K]^T
//
// y[n] = sum_k x[k] * W[n, k]  (W is N×K row-major; each row is one output's
// weight vector). This is the LM-head / output-projection pattern when weights
// are stored as (vocab, d_model) rather than (d_model, vocab).
//
// One threadgroup of 128 threads processes 128 consecutive output rows; each
// thread sums its own row's dot product. x is cooperatively staged into TG
// memory in BK=256 tiles.
//
#include <metal_stdlib>
using namespace metal;

enum : uint { GVT_BN = 128, GVT_BK = 256 };

[[host_name("gemv_t_fp16_m1")]]
[[kernel]]
void gemv_t_fp16_m1(
    device const half* x [[buffer(0)]],   // (1, K)
    device const half* W [[buffer(1)]],   // (N, K) row-major
    device       half* y [[buffer(2)]],   // (1, N)
    constant uint&     N [[buffer(3)]],
    constant uint&     K [[buffer(4)]],
    uint  gid  [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]])
{
    const uint row0 = gid * GVT_BN;
    const uint row  = row0 + tid;

    threadgroup half xs[GVT_BK];
    float acc = 0.0f;

    const uint K_main = (K / GVT_BK) * GVT_BK;
    for (uint k0 = 0; k0 < K_main; k0 += GVT_BK) {
        for (uint i = tid; i < GVT_BK / 2; i += GVT_BN) {
            reinterpret_cast<threadgroup half2*>(xs)[i] =
                reinterpret_cast<const device half2*>(x + k0)[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < N) {
            // Each thread reads its own row contiguously: W[row, k0..k0+BK].
            // Use half8 vector loads for ILP.
            device const half* wrow = W + (uint64_t)row * K + k0;
            for (uint kk = 0; kk < GVT_BK; kk += 8) {
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
        device const half* wrow = W + (uint64_t)row * K;
        for (uint k = K_main; k < K; k++) {
            acc += float(x[k]) * float(wrow[k]);
        }
        y[row] = half(acc);
    }
}
