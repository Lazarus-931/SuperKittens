//
//  rms_norm.metal — RMSNorm, half4 vectorized, SIMD-group per row
//

#include <metal_stdlib>
using namespace metal;

enum : uint { THREADS = 128, ROWS = 4 };

[[host_name("rmsnorm")]]
[[kernel, max_total_threads_per_threadgroup(THREADS)]]
void rmsnorm(
    device const half* x      [[buffer(0)]],
    device const half* gamma  [[buffer(1)]],
    device half* y            [[buffer(2)]],
    constant uint& rows       [[buffer(3)]],
    constant uint& d          [[buffer(4)]],
    constant float& eps       [[buffer(5)]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint2 gid  [[threadgroup_position_in_grid]])
{
    const uint row = gid.y * ROWS + simd;
    if (row >= rows) return;

    const size_t off = (size_t)row * d;
    const device half4* x4 = reinterpret_cast<const device half4*>(x + off);
    const device half4* g4 = reinterpret_cast<const device half4*>(gamma);
    device half4* y4 = reinterpret_cast<device half4*>(y + off);

    float sumSq = 0.0f;
    const uint n4 = d / 4;

    for (uint k = lane; k < n4; k += 32) {
        float4 v = float4(x4[k]);
        sumSq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    for (uint k = n4 * 4 + lane; k < d; k += 32) {
        float v = float(x[off + k]);
        sumSq += v * v;
    }
    sumSq = simd_sum(sumSq);
    float inv_rms = metal::precise::rsqrt(sumSq / float(d) + eps);

    for (uint k = lane; k < n4; k += 32) {
        float4 v = float4(x4[k]);
        float4 wv = float4(g4[k]);
        y4[k] = half4(v * inv_rms * wv);
    }
    for (uint k = n4 * 4 + lane; k < d; k += 32) {
        y[off + k] = half(float(x[off + k]) * inv_rms * float(gamma[k]));
    }
}
