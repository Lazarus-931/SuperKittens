//
//  layernorm.metal
//  SuperKittens
//
//  SIMD-group norm kernels. Each SIMD group processes one row independently.
//  Vectorized half4 loads — no barriers, no threadgroup memory.

#include <metal_stdlib>
using namespace metal;

enum : uint { THREADS = 128, ROWS_PER_GRP = 4 };

// ── LayerNorm: y = (x - mean) * rsqrt(var + eps) * gamma + beta ──

[[host_name("layernorm")]]
[[kernel, max_total_threads_per_threadgroup(THREADS)]]
void layernorm(
    device const half* x      [[buffer(0)]],
    device const half* gamma  [[buffer(1)]],
    device const half* beta   [[buffer(2)]],
    device half* y            [[buffer(3)]],
    constant uint& rows       [[buffer(4)]],
    constant uint& d          [[buffer(5)]],
    constant float& eps       [[buffer(6)]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint2 gid  [[threadgroup_position_in_grid]])
{
    const uint row = gid.y * ROWS_PER_GRP + simd;
    if (row >= rows) return;

    const size_t off = (size_t)row * d;
    const device half4* x4 = reinterpret_cast<const device half4*>(x + off);
    const device half4* g4 = reinterpret_cast<const device half4*>(gamma);
    const device half4* b4 = reinterpret_cast<const device half4*>(beta);
    device half4* y4 = reinterpret_cast<device half4*>(y + off);

    float2 acc = 0.0f;
    const uint n4 = d / 4;

    for (uint k = lane; k < n4; k += 32) {
        float4 v = float4(x4[k]);
        acc.x += v.x + v.y + v.z + v.w;
        acc.y += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    for (uint k = n4 * 4 + lane; k < d; k += 32) {
        float v = float(x[off + k]);
        acc.x += v;
        acc.y += v * v;
    }
    acc.x = simd_sum(acc.x);
    acc.y = simd_sum(acc.y);

    const float inv_d  = 1.0f / float(d);
    const float mean   = acc.x * inv_d;
    const float var    = fmax(acc.y * inv_d - mean * mean, 0.0f);
    const float inv_std = metal::precise::rsqrt(var + eps);

    for (uint k = lane; k < n4; k += 32) {
        float4 v  = float4(x4[k]);
        float4 gv = float4(g4[k]);
        float4 bv = float4(b4[k]);
        y4[k] = half4(((v - mean) * inv_std) * gv + bv);
    }
    for (uint k = n4 * 4 + lane; k < d; k += 32) {
        float v = float(x[off + k]);
        y[off + k] = half(((v - mean) * inv_std) * float(gamma[k]) + float(beta[k]));
    }
}

// ── LayerNorm + residual: y = layernorm(x + res, gamma, beta) ──

[[host_name("layernorm_residual")]]
[[kernel, max_total_threads_per_threadgroup(THREADS)]]
void layernorm_residual(
    device const half* x      [[buffer(0)]],
    device const half* res    [[buffer(1)]],
    device const half* gamma  [[buffer(2)]],
    device const half* beta   [[buffer(3)]],
    device half* y            [[buffer(4)]],
    constant uint& rows       [[buffer(5)]],
    constant uint& d          [[buffer(6)]],
    constant float& eps       [[buffer(7)]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint2 gid  [[threadgroup_position_in_grid]])
{
    const uint row = gid.y * ROWS_PER_GRP + simd;
    if (row >= rows) return;

    const size_t off = (size_t)row * d;
    const device half4* x4  = reinterpret_cast<const device half4*>(x + off);
    const device half4* r4  = reinterpret_cast<const device half4*>(res + off);
    const device half4* g4  = reinterpret_cast<const device half4*>(gamma);
    const device half4* b4  = reinterpret_cast<const device half4*>(beta);
    device half4* y4 = reinterpret_cast<device half4*>(y + off);

    float2 acc = 0.0f;
    const uint n4 = d / 4;

    for (uint k = lane; k < n4; k += 32) {
        float4 v = float4(x4[k]) + float4(r4[k]);
        acc.x += v.x + v.y + v.z + v.w;
        acc.y += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    for (uint k = n4 * 4 + lane; k < d; k += 32) {
        float v = float(x[off + k]) + float(res[off + k]);
        acc.x += v;
        acc.y += v * v;
    }
    acc.x = simd_sum(acc.x);
    acc.y = simd_sum(acc.y);

    const float inv_d  = 1.0f / float(d);
    const float mean   = acc.x * inv_d;
    const float var    = fmax(acc.y * inv_d - mean * mean, 0.0f);
    const float inv_std = metal::precise::rsqrt(var + eps);

    for (uint k = lane; k < n4; k += 32) {
        float4 v  = float4(x4[k]) + float4(r4[k]);
        float4 gv = float4(g4[k]);
        float4 bv = float4(b4[k]);
        y4[k] = half4(((v - mean) * inv_std) * gv + bv);
    }
    for (uint k = n4 * 4 + lane; k < d; k += 32) {
        float v = float(x[off + k]) + float(res[off + k]);
        y[off + k] = half(((v - mean) * inv_std) * float(gamma[k]) + float(beta[k]));
    }
}

// ── RMSNorm: y = x * rsqrt(sum(x^2)/d + eps) * gamma ──

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
    const uint row = gid.y * ROWS_PER_GRP + simd;
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

    const float inv_std = metal::precise::rsqrt(sumSq / float(d) + eps);

    for (uint k = lane; k < n4; k += 32) {
        float4 v  = float4(x4[k]);
        float4 gv = float4(g4[k]);
        y4[k] = half4(v * inv_std * gv);
    }
    for (uint k = n4 * 4 + lane; k < d; k += 32) {
        float v = float(x[off + k]);
        y[off + k] = half(v * inv_std * float(gamma[k]));
    }
}

// ── RMSNorm + residual: y = rmsnorm(x + res, gamma) ──

[[host_name("rmsnorm_residual")]]
[[kernel, max_total_threads_per_threadgroup(THREADS)]]
void rmsnorm_residual(
    device const half* x      [[buffer(0)]],
    device const half* res    [[buffer(1)]],
    device const half* gamma  [[buffer(2)]],
    device half* y            [[buffer(3)]],
    constant uint& rows       [[buffer(4)]],
    constant uint& d          [[buffer(5)]],
    constant float& eps       [[buffer(6)]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint2 gid  [[threadgroup_position_in_grid]])
{
    const uint row = gid.y * ROWS_PER_GRP + simd;
    if (row >= rows) return;

    const size_t off = (size_t)row * d;
    const device half4* x4  = reinterpret_cast<const device half4*>(x + off);
    const device half4* r4  = reinterpret_cast<const device half4*>(res + off);
    const device half4* g4  = reinterpret_cast<const device half4*>(gamma);
    device half4* y4 = reinterpret_cast<device half4*>(y + off);

    float sumSq = 0.0f;
    const uint n4 = d / 4;

    for (uint k = lane; k < n4; k += 32) {
        float4 v = float4(x4[k]) + float4(r4[k]);
        sumSq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    for (uint k = n4 * 4 + lane; k < d; k += 32) {
        float v = float(x[off + k]) + float(res[off + k]);
        sumSq += v * v;
    }
    sumSq = simd_sum(sumSq);

    const float inv_std = metal::precise::rsqrt(sumSq / float(d) + eps);

    for (uint k = lane; k < n4; k += 32) {
        float4 v  = float4(x4[k]) + float4(r4[k]);
        float4 gv = float4(g4[k]);
        y4[k] = half4(v * inv_std * gv);
    }
    for (uint k = n4 * 4 + lane; k < d; k += 32) {
        float v = float(x[off + k]) + float(res[off + k]);
        y[off + k] = half(v * inv_std * float(gamma[k]));
    }
}
