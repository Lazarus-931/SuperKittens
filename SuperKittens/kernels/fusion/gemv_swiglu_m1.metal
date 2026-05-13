// gemv_swiglu_m1.metal — Fused gate+up+SiLU·mul matvec for M=1 decode.
//
//   y[1, N] = silu(x[1, K] @ W_gate[K, N]) * (x[1, K] @ W_up[K, N])
//
// Replaces 3 separate dispatches (gate-gemv, up-gemv, silu_mul) with one.
// Same structure as gemv_fp16_m1: 128 threads × BK=256 cooperative x-tiling.
//
#include <metal_stdlib>
using namespace metal;

enum : uint { SWIGLU_BN = 128, SWIGLU_BK = 256 };

[[host_name("gemv_swiglu_fp16_m1")]]
[[kernel]]
void gemv_swiglu_fp16_m1(
    device const half* x      [[buffer(0)]],   // (1, K)
    device const half* W_gate [[buffer(1)]],   // (K, N) row-major
    device const half* W_up   [[buffer(2)]],   // (K, N) row-major
    device       half* y      [[buffer(3)]],   // (1, N) = silu(x@W_gate) * (x@W_up)
    constant uint&     N      [[buffer(4)]],
    constant uint&     K      [[buffer(5)]],
    uint  gid  [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]])
{
    const uint col0 = gid * SWIGLU_BN;
    const uint col  = col0 + tid;

    threadgroup half xs[SWIGLU_BK];
    float acc_g = 0.0f;
    float acc_u = 0.0f;

    const uint K_main = (K / SWIGLU_BK) * SWIGLU_BK;

    for (uint k0 = 0; k0 < K_main; k0 += SWIGLU_BK) {
        // Cooperative x load: 128 threads × 1 half2 per pass; need 256 halves → 2 passes.
        for (uint i = tid; i < SWIGLU_BK / 2; i += SWIGLU_BN) {
            reinterpret_cast<threadgroup half2*>(xs)[i] =
                reinterpret_cast<const device half2*>(x + k0)[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (col < N) {
            for (uint kk = 0; kk < SWIGLU_BK; kk += 8) {
                float x0 = float(xs[kk+0]);
                float x1 = float(xs[kk+1]);
                float x2 = float(xs[kk+2]);
                float x3 = float(xs[kk+3]);
                float x4 = float(xs[kk+4]);
                float x5 = float(xs[kk+5]);
                float x6 = float(xs[kk+6]);
                float x7 = float(xs[kk+7]);

                uint base = (k0+kk) * N + col;
                float g0 = float(W_gate[base + 0*N]) * x0;
                float g1 = float(W_gate[base + 1*N]) * x1;
                float g2 = float(W_gate[base + 2*N]) * x2;
                float g3 = float(W_gate[base + 3*N]) * x3;
                float g4 = float(W_gate[base + 4*N]) * x4;
                float g5 = float(W_gate[base + 5*N]) * x5;
                float g6 = float(W_gate[base + 6*N]) * x6;
                float g7 = float(W_gate[base + 7*N]) * x7;
                acc_g += (g0+g1+g2+g3+g4+g5+g6+g7);

                float u0 = float(W_up[base + 0*N]) * x0;
                float u1 = float(W_up[base + 1*N]) * x1;
                float u2 = float(W_up[base + 2*N]) * x2;
                float u3 = float(W_up[base + 3*N]) * x3;
                float u4 = float(W_up[base + 4*N]) * x4;
                float u5 = float(W_up[base + 5*N]) * x5;
                float u6 = float(W_up[base + 6*N]) * x6;
                float u7 = float(W_up[base + 7*N]) * x7;
                acc_u += (u0+u1+u2+u3+u4+u5+u6+u7);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (col < N) {
        for (uint k = K_main; k < K; k++) {
            float xk = float(x[k]);
            acc_g += xk * float(W_gate[k*N + col]);
            acc_u += xk * float(W_up[k*N + col]);
        }
        // SwiGLU: silu(g) * u = (g / (1 + exp(-g))) * u
        float sig = 1.0f / (1.0f + exp(-acc_g));
        y[col] = half(acc_g * sig * acc_u);
    }
}
