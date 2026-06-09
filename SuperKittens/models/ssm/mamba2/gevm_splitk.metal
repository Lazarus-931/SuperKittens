//
//  gevm_splitk.metal — split-K M=1 matvec for small-N decode projections.
//
//  y[1,N] = x[1,K] @ W[K,N] (row-major, transB=0). Same math as gemv_fp16_m1.
//  WHY: gemv_fp16_m1 launches ceil(N/128) threadgroups. For mamba2 out_proj
//  (N=d_model=768 → 6 TGs/layer) that under-occupies the GPU and the matvec
//  runs at ~56 GB/s vs ~101 for the larger in_proj (N=3352, 27 TGs). Splitting
//  the K reduction across KS groups gives 6*KS TGs so memory latency is hidden;
//  a tiny reduce pass sums the KS partials.
//
//  Pass 1 grid: (ceil(N/BN), KS, 1), tg (BN,1,1). Each (col-tile, ks) TG sums
//  its K-slice into partial[ks*N + col].
//  Pass 2 grid: (ceil(N/256),1,1), tg(256,1,1). Sums KS partials → y.

#include <metal_stdlib>
using namespace metal;

enum : uint { GSK_BN = 128, GSK_BK = 256 };

[[host_name("gevm_splitk_p1")]]
[[kernel]]
void gevm_splitk_p1(
    device const half*  x        [[buffer(0)]],   // (1,K)
    device const half*  W        [[buffer(1)]],   // (K,N) row-major
    device       float* partial  [[buffer(2)]],   // (KS,N) fp32
    constant uint&      N        [[buffer(3)]],
    constant uint&      K        [[buffer(4)]],
    constant uint&      KS       [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]])
{
    const uint tid = tid2.x;
    const uint col = gid.x * GSK_BN + tid;
    const uint ks  = gid.y;                 // 0..KS-1

    // K-slice [k_lo, k_hi) for this group; aligned to GSK_BK boundaries to keep
    // the coalesced x-load path. Last group absorbs the remainder + tail.
    const uint per   = ((K / GSK_BK) + KS - 1u) / KS * GSK_BK;   // multiple of BK
    const uint k_lo  = ks * per;
    uint       k_hi  = k_lo + per;
    const bool last  = (ks == KS - 1u);
    if (last) k_hi = K;
    if (k_lo >= K) { if (col < N) partial[(size_t)ks * N + col] = 0.0f; return; }
    if (k_hi > K)  k_hi = K;

    threadgroup half xs[GSK_BK];
    float acc = 0.0f;

    const uint k_main = k_lo + ((k_hi - k_lo) / GSK_BK) * GSK_BK;
    for (uint k0 = k_lo; k0 < k_main; k0 += GSK_BK) {
        for (uint i = tid; i < GSK_BK / 2; i += GSK_BN)
            reinterpret_cast<threadgroup half2*>(xs)[i] =
                reinterpret_cast<const device half2*>(x + k0)[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (col < N) {
            for (uint kk = 0; kk < GSK_BK; kk += 8) {
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
        for (uint k = k_main; k < k_hi; ++k)
            acc += float(x[k]) * float(W[k*N + col]);
        partial[(size_t)ks * N + col] = acc;
    }
}

[[host_name("gevm_splitk_p2")]]
[[kernel]]
void gevm_splitk_p2(
    device const float* partial [[buffer(0)]],   // (KS,N)
    device       half*  y       [[buffer(1)]],   // (1,N)
    constant uint&      N       [[buffer(2)]],
    constant uint&      KS      [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= N) return;
    float acc = 0.0f;
    for (uint ks = 0; ks < KS; ++ks) acc += partial[(size_t)ks * N + gid];
    y[gid] = half(acc);
}
