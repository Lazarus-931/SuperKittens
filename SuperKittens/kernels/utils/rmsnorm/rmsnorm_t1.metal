

#include <metal_stdlib>
using namespace metal;

[[host_name("rmsnorm_t1")]]
[[kernel, max_total_threads_per_threadgroup(256)]]
void rmsnorm_t1(
    device const half*  x      [[buffer(0)]],
    device const half*  gamma  [[buffer(1)]],
    device       half*  y      [[buffer(2)]],
    constant uint&      rows   [[buffer(3)]],
    constant uint&      d      [[buffer(4)]],
    constant float&     eps    [[buffer(5)]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint2 gid  [[threadgroup_position_in_grid]])
{
    const uint row = gid.y;
    if (row >= rows) return;

    const size_t off = (size_t)row * d;
    const device half4* x4 = reinterpret_cast<const device half4*>(x + off);
    const device half4* g4 = reinterpret_cast<const device half4*>(gamma);
    device       half4* y4 = reinterpret_cast<device       half4*>(y + off);

    threadgroup float partial[8];

    const uint n4 = d / 4;
    float sumSq = 0.0f;
    for (uint k = tid; k < n4; k += 256u) {
        float4 v = float4(x4[k]);
        sumSq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    // Tail (d not divisible by 4): contributing lanes from any simdgroup.
    for (uint k = n4 * 4u + tid; k < d; k += 256u) {
        float v = float(x[off + k]);
        sumSq += v * v;
    }
    sumSq = simd_sum(sumSq);
    if (lane == 0) partial[simd] = sumSq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        float v = (lane < 8u) ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = metal::precise::rsqrt(partial[0] / float(d) + eps);

    for (uint k = tid; k < n4; k += 256u) {
        float4 v  = float4(x4[k]);
        float4 gv = float4(g4[k]);
        y4[k] = half4(v * inv_rms * gv);
    }
    for (uint k = n4 * 4u + tid; k < d; k += 256u) {
        y[off + k] = half(float(x[off + k]) * inv_rms * float(gamma[k]));
    }
}
