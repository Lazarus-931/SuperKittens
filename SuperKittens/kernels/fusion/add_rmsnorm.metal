//  add_rmsnorm.metal v2 — fused (y = x + delta) + RMSNorm(y) → y_norm

#include <metal_stdlib>
using namespace metal;

enum : uint { THREADS = 128, SIMDS = 4 };

[[host_name("add_rmsnorm")]]
[[kernel, max_total_threads_per_threadgroup(THREADS)]]
void add_rmsnorm(
    device const half* x      [[buffer(0)]],
    device const half* delta  [[buffer(1)]],
    device const half* gamma  [[buffer(2)]],
    device half*       y      [[buffer(3)]],
    device half*       y_norm [[buffer(4)]],
    constant uint&     rows   [[buffer(5)]],
    constant uint&     n      [[buffer(6)]],
    constant float&    eps    [[buffer(7)]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint2 gid  [[threadgroup_position_in_grid]])
{
    const uint row = gid.y;
    if (row >= rows) return;

    threadgroup float partial[SIMDS];

    const size_t off = (size_t)row * n;
    const device half4* x4 = reinterpret_cast<const device half4*>(x + off);
    const device half4* d4 = reinterpret_cast<const device half4*>(delta + off);
    const device half4* g4 = reinterpret_cast<const device half4*>(gamma);
    device half4* y4   = reinterpret_cast<device half4*>(y + off);
    device half4* yn4  = reinterpret_cast<device half4*>(y_norm + off);

    const uint n4 = n / 4;
    float sumSq = 0.0f;

    // Pass 1: sum-of-squares of (x+δ) across all 128 threads.
    for (uint k = tid; k < n4; k += THREADS) {
        float4 a = float4(x4[k]);
        float4 b = float4(d4[k]);
        float4 s = a + b;
        sumSq += s.x*s.x + s.y*s.y + s.z*s.z + s.w*s.w;
    }
    for (uint k = n4 * 4 + tid; k < n; k += THREADS) {
        float s = float(x[off + k]) + float(delta[off + k]);
        sumSq += s * s;
    }
    sumSq = simd_sum(sumSq);
    if (lane == 0) partial[simd] = sumSq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final cross-simd reduction (4 values).
    float total;
    if (simd == 0) {
        float v = lane < SIMDS ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = partial[0];
    float inv_rms = metal::precise::rsqrt(total / float(n) + eps);

    // Pass 2: re-add x+δ (L2-hot), write y and y_norm.
    for (uint k = tid; k < n4; k += THREADS) {
        float4 a = float4(x4[k]);
        float4 b = float4(d4[k]);
        float4 s = a + b;
        float4 gv = float4(g4[k]);
        y4[k]  = half4(s);
        yn4[k] = half4(s * inv_rms * gv);
    }
    for (uint k = n4 * 4 + tid; k < n; k += THREADS) {
        float s = float(x[off + k]) + float(delta[off + k]);
        y[off + k] = half(s);
        y_norm[off + k] = half(s * inv_rms * float(gamma[k]));
    }
}
