// rmsnorm_bf16.metal — BF16 RMSNorm for Gemma 4.
//   y = x * rsqrt(sum(x^2)/d + eps) * gamma
// Internal compute float32, storage bfloat.

#include <metal_stdlib>
using namespace metal;

namespace meow::gemma4::rmsn {

enum : uint { THREADS = 128, ROWS_PER_GRP = 4 };

[[host_name("rmsnorm_bf16")]]
[[kernel, max_total_threads_per_threadgroup(THREADS)]]
void rmsnorm_bf16(
    device const bfloat* x      [[buffer(0)]],
    device const bfloat* gamma  [[buffer(1)]],
    device bfloat* y            [[buffer(2)]],
    constant uint& rows         [[buffer(3)]],
    constant uint& d            [[buffer(4)]],
    constant float& eps         [[buffer(5)]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint2 gid  [[threadgroup_position_in_grid]])
{
    const uint row = gid.y * ROWS_PER_GRP + simd;
    if (row >= rows) return;

    const size_t off = (size_t)row * d;
    const device bfloat4* x4 = reinterpret_cast<const device bfloat4*>(x + off);
    const device bfloat4* g4 = reinterpret_cast<const device bfloat4*>(gamma);
    device bfloat4* y4 = reinterpret_cast<device bfloat4*>(y + off);

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

    const float inv_std = metal::precise::rsqrt(sumSq / float(d) + eps);

    for (uint k = lane; k < n4; k += 32) {
        float4 v  = float4(x4[k]);
        float4 gv = float4(g4[k]);
        y4[k] = bfloat4(v * inv_std * gv);
    }
    for (uint k = n4 * 4 + lane; k < d; k += 32) {
        float v = float(x[off + k]);
        y[off + k] = bfloat(v * inv_std * float(gamma[k]));
    }
}

// Single-row fast path: 256 threads / 8 SGs collaborating on one row.
// Mirrors rmsnorm_t1.metal (fp16). Same ABI as `rmsnorm_bf16`; substituted
// at PSO level when rows==1. Dispatch: (1, rows, 1) × (256, 1, 1).
[[host_name("rmsnorm_bf16_t1")]]
[[kernel, max_total_threads_per_threadgroup(256)]]
void rmsnorm_bf16_t1(
    device const bfloat* x      [[buffer(0)]],
    device const bfloat* gamma  [[buffer(1)]],
    device bfloat* y            [[buffer(2)]],
    constant uint& rows         [[buffer(3)]],
    constant uint& d            [[buffer(4)]],
    constant float& eps         [[buffer(5)]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint2 gid  [[threadgroup_position_in_grid]])
{
    const uint row = gid.y;
    if (row >= rows) return;

    const size_t off = (size_t)row * d;
    const device bfloat4* x4 = reinterpret_cast<const device bfloat4*>(x + off);
    const device bfloat4* g4 = reinterpret_cast<const device bfloat4*>(gamma);
    device bfloat4* y4 = reinterpret_cast<device bfloat4*>(y + off);

    threadgroup float partial[8];

    const uint n4 = d / 4;
    float sumSq = 0.0f;
    for (uint k = tid; k < n4; k += 256u) {
        float4 v = float4(x4[k]);
        sumSq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
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
    const float inv_std = metal::precise::rsqrt(partial[0] / float(d) + eps);

    for (uint k = tid; k < n4; k += 256u) {
        float4 v  = float4(x4[k]);
        float4 gv = float4(g4[k]);
        y4[k] = bfloat4(v * inv_std * gv);
    }
    for (uint k = n4 * 4u + tid; k < d; k += 256u) {
        float v = float(x[off + k]);
        y[off + k] = bfloat(v * inv_std * float(gamma[k]));
    }
}

} // namespace
