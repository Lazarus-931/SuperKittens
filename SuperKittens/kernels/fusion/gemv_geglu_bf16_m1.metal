// gemv_geglu_bf16_m1.metal — fused gate+up+GeLU+mul at M=1.
//
// Ported from temp/gemma4_kernel_lab/kernels.metal (lab_gemv_geglu_bf16_m1).
// Replaces the down-stream half of the gemma4 GeGLU MLP at decode time:
//
//   y[1, N_int] = gelu_tanh(x @ W_gate) * (x @ W_up)
//
// Where W_gate, W_up are (K, N_int) row-major (K = d_model). The full MLP is
// completed by running gemv_bf16_m1(down, y) → mlp_out.
//
// Grid: (ceil(N_int / 8), 1, 1) threadgroups; 256 threads per TG (8 simdgroups
// of 32 lanes). Each simdgroup owns one output column; lanes share work
// across K with simd_sum.

#include <metal_stdlib>
using namespace metal;

inline float _gelu_tanh(float x) {
    const float K0 = 0.7978845608028654f;  // sqrt(2/pi)
    const float K1 = 0.044715f;
    float u = K0 * (x + K1 * x * x * x);
    float t = precise::tanh(u);
    return 0.5f * x * (1.0f + t);
}

[[host_name("gemv_geglu_bf16_m1")]]
[[kernel, max_total_threads_per_threadgroup(256)]]
void gemv_geglu_bf16_m1(
    device const bfloat* x      [[buffer(0)]],   // (1, K)
    device const bfloat* W_gate [[buffer(1)]],   // (K, N_int)
    device const bfloat* W_up   [[buffer(2)]],   // (K, N_int)
    device       bfloat* y      [[buffer(3)]],   // (1, N_int) = gelu(gate)*up
    constant uint& N_int       [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    uint  gid  [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    constexpr uint SIMDS_PER_TG = 8;
    constexpr uint BN = SIMDS_PER_TG;
    constexpr uint BK = 512;
    const uint col0 = gid * BN;
    const uint col  = col0 + simd;
    threadgroup bfloat xs[BK];

    float acc_g = 0.0f, acc_u = 0.0f;
    const uint K_main = (K / BK) * BK;
    for (uint k0 = 0; k0 < K_main; k0 += BK) {
        for (uint i = tid; i < BK / 4; i += SIMDS_PER_TG * 32) {
            reinterpret_cast<threadgroup bfloat4*>(xs)[i] =
                reinterpret_cast<const device bfloat4*>(x + k0)[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (col < N_int) {
            for (uint kk = lane; kk < BK; kk += 32) {
                float xv = (float)xs[kk];
                acc_g += xv * (float)W_gate[(k0 + kk) * N_int + col];
                acc_u += xv * (float)W_up  [(k0 + kk) * N_int + col];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (col < N_int) {
        if (lane == 0) {
            float ag = 0.0f, au = 0.0f;
            for (uint k = K_main; k < K; ++k) {
                float xv = (float)x[k];
                ag += xv * (float)W_gate[k * N_int + col];
                au += xv * (float)W_up  [k * N_int + col];
            }
            acc_g += ag; acc_u += au;
        }
        float sg = simd_sum(acc_g);
        float su = simd_sum(acc_u);
        if (lane == 0) {
            y[col] = bfloat(_gelu_tanh(sg) * su);
        }
    }
}
