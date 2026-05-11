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
