//
//  conv1d.metal — Depthwise causal Conv1D, half4 vectorized
//  Used by Mamba2 after in_proj. SIMD-group per position, no barriers.
//  Weight: [C, K] where K is kernel width (usually 4).
//

#include <metal_stdlib>
using namespace metal;

enum : uint { THREADS = 128, ROWS = 4, K_MAX = 4 };

[[host_name("conv1d")]]
[[kernel, max_total_threads_per_threadgroup(THREADS)]]
void conv1d(
    device const half* x,       // (B, L, C)
    device const half* weight,  // (C, K)
    device const half* bias,    // (C,)
    device half* y,             // (B, L, C)
    constant uint& B,
    constant uint& L,
    constant uint& C,
    constant uint& K,           // kernel width (e.g. 4)
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint3 gid  [[threadgroup_position_in_grid]])
{
    const uint b   = gid.x;
    const uint pos = gid.y * ROWS + simd;
    if (pos >= L) return;
    const size_t base = (size_t)b * L * C;
    const uint n4 = C / 4;

    // Vectorized path
    for (uint ci = lane; ci < n4; ci += 32) {
        float4 acc = float4(reinterpret_cast<const device half4*>(bias)[ci]);
        for (uint k = 0; k < K; k++) {
            int sp = (int)pos - (int)(K - 1) + (int)k;
            float4 xv = (sp >= 0)
                ? float4(reinterpret_cast<const device half4*>(x + base + (uint)sp * C)[ci])
                : float4(0);
            acc.x += xv.x * float(weight[(ci * 4 + 0) * K + k]);
            acc.y += xv.y * float(weight[(ci * 4 + 1) * K + k]);
            acc.z += xv.z * float(weight[(ci * 4 + 2) * K + k]);
            acc.w += xv.w * float(weight[(ci * 4 + 3) * K + k]);
        }
        reinterpret_cast<device half4*>(y + base + pos * C)[ci] = half4(acc);
    }

    // Trailing channels
    for (uint ci = n4 * 4 + lane; ci < C; ci += 32) {
        float acc = float(bias[ci]);
        for (uint k = 0; k < K; k++) {
            int sp = (int)pos - (int)(K - 1) + (int)k;
            half xv = (sp >= 0) ? x[base + (uint)sp * C + ci] : half(0);
            acc += float(xv) * float(weight[ci * K + k]);
        }
        y[base + pos * C + ci] = half(acc);
    }
}
