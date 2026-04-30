//
//  layernorm.metal
//  SuperKittens
//
//  SIMD-group layernorm. Each SIMD group processes one row independently.
//  No threadgroup barriers — sum/sumSq reduced via simd_sum within each SIMD group.

#include <metal_stdlib>
using namespace metal;

enum : uint { THREADS = 128, ROWS_PER_GRP = 4 };

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
    uint  simd_id  [[simdgroup_index_in_threadgroup]],
    uint  lane_id  [[thread_index_in_simdgroup]],
    uint2 gid      [[threadgroup_position_in_grid]])
{
    const uint row = gid.y * ROWS_PER_GRP + simd_id;
    if (row >= rows) return;

    const size_t off = (size_t)row * d;
    device const half* x_row = x + off;
    device const half* g = gamma;
    device const half* b = beta;
    device half* y_row = y + off;

    // ── Pass 1: compute sum and sumSq via simd reduction ──
    float sum   = 0.0f;
    float sumSq = 0.0f;

    for (uint k = lane_id; k < d; k += 32) {
        float val = float(x_row[k]);
        sum   += val;
        sumSq += val * val;
    }

    sum   = simd_sum(sum);
    sumSq = simd_sum(sumSq);

    const float inv_d  = 1.0f / float(d);
    const float mean   = sum * inv_d;
    const float var    = fmax(sumSq * inv_d - mean * mean, 0.0f);
    const float inv_std = metal::precise::rsqrt(var + eps);

    // ── Pass 2: normalize + affine ──
    for (uint k = lane_id; k < d; k += 32) {
        float val = float(x_row[k]);
        val = (val - mean) * inv_std;
        val = val * float(g[k]) + float(b[k]);
        y_row[k] = half(val);
    }
}
