//
//  conv1d_silu.metal
//  SuperKittens — depthwise causal Conv1D + SiLU for Mamba2
//
//  SIMD-group per position. half4 vectorized. No barriers.

#include <metal_stdlib>
using namespace metal;

enum : uint { THREADS = 128, ROWS = 4, K = 4 };

[[host_name("conv1d_silu")]]
[[kernel, max_total_threads_per_threadgroup(THREADS)]]
void conv1d_silu(
    device const half* x,
    device const half* weight,
    device const half* bias,
    device half* y,
    constant uint& B, constant uint& L, constant uint& C,
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint3 gid  [[threadgroup_position_in_grid]])
{
    const uint b   = gid.x;
    const uint pos = gid.y * ROWS + simd;
    if (pos >= L) return;

    const size_t base = (size_t)b * L * C;
    const uint n4 = C / 4;

    // Vectorized: process 4 channels at a time with half4 for x/bias/y
    for (uint ci = lane; ci < n4; ci += 32) {
        float4 acc = float4(reinterpret_cast<const device half4*>(bias)[ci]);
        for (uint k = 0; k < K; k++) {
            int sp = (int)pos - (K - 1) + (int)k;
            float4 xv = (sp >= 0)
                ? float4(reinterpret_cast<const device half4*>(x + base + (uint)sp * C)[ci])
                : float4(0);
            // Weight is [C,K] — strided, load 4 elements individually
            acc.x += xv.x * float(weight[(ci*4 + 0) * K + k]);
            acc.y += xv.y * float(weight[(ci*4 + 1) * K + k]);
            acc.z += xv.z * float(weight[(ci*4 + 2) * K + k]);
            acc.w += xv.w * float(weight[(ci*4 + 3) * K + k]);
        }
        float4 sv = acc / (1.0f + metal::fast::exp(-acc));
        reinterpret_cast<device half4*>(y + base + pos * C)[ci] = half4(sv);
    }

    // Trailing channels (C not divisible by 4)
    for (uint ci = n4 * 4 + lane; ci < C; ci += 32) {
        float acc = float(bias[ci]);
        for (uint k = 0; k < K; k++) {
            int sp = (int)pos - (K - 1) + (int)k;
            half xv = (sp >= 0) ? *(x + base + (uint)sp * C + ci) : half(0);
            acc += float(xv) * float(weight[ci * K + k]);
        }
        acc = acc / (1.0f + metal::fast::exp(-acc));
        *(y + base + pos * C + ci) = half(acc);
    }
}

// Capture the last K-1 pre-conv tokens of a prefill into conv_state so the
// O(1) decode step can continue the causal window. Left-zero-padded when
// L < K-1 (matches HF Mamba2Cache.update_conv_state cache_init pad). Row j of
// conv_state holds prefill position (L - (K-1) + j); the newest is row K-2.
[[host_name("conv_state_capture")]]
[[kernel]]
void conv_state_capture(
    device const half* x,           // (B, L, C) pre-conv xBC
    device       half* conv_state,  // (B, K-1, C)
    constant uint& B, constant uint& L, constant uint& C, constant uint& K,
    uint3 tid [[thread_position_in_grid]])
{
    const uint c = tid.x;
    const uint j = tid.y;           // 0..K-2
    const uint b = tid.z;
    if (c >= C || j >= K - 1 || b >= B) return;
    const int sp = (int)L - (int)(K - 1) + (int)j;
    half v = (sp >= 0) ? x[((size_t)b * L + (uint)sp) * C + c] : half(0);
    conv_state[((size_t)b * (K - 1) + j) * C + c] = v;
}

// Single-step causal Conv1D + SiLU for O(1) decode. Convolves the new token's
// pre-conv xBC against the carried (K-1)-token window, then rolls the window
// (drop oldest, append the new pre-conv token). Mirrors HF causal_conv1d_update.
//   y[c] = silu( bias[c] + sum_{j<K-1} w[c,j]*conv_state[j,c] + w[c,K-1]*x_new[c] )
[[host_name("conv1d_silu_step")]]
[[kernel]]
void conv1d_silu_step(
    device const half* x_new,       // (B, C) new token pre-conv xBC
    device const half* weight,      // (C, K)
    device const half* bias,        // (C,)
    device       half* y,           // (B, C) conv+silu output
    device       half* conv_state,  // (B, K-1, C) in-out
    constant uint& B, constant uint& C, constant uint& K,
    uint3 tid [[thread_position_in_grid]])
{
    const uint c = tid.x;
    const uint b = tid.y;
    if (c >= C || b >= B) return;

    const size_t cs_base = (size_t)b * (K - 1) * C;
    const float xn = float(x_new[(size_t)b * C + c]);

    float acc = float(bias[c]);
    for (uint j = 0; j < K - 1; ++j)
        acc += float(conv_state[cs_base + (size_t)j * C + c]) * float(weight[c * K + j]);
    acc += xn * float(weight[c * K + (K - 1)]);

    y[(size_t)b * C + c] = half(acc / (1.0f + metal::fast::exp(-acc)));

    // Roll the window: shift left, append the new pre-conv token.
    for (uint j = 0; j + 1 < K - 1; ++j)
        conv_state[cs_base + (size_t)j * C + c] =
            conv_state[cs_base + (size_t)(j + 1) * C + c];
    conv_state[cs_base + (size_t)(K - 2) * C + c] = half(xn);
}
