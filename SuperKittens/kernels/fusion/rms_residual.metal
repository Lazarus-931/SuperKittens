//
//  rms_residual.metal — fused RMSNorm(x + residual)
//
//  Reads x and residual, adds them, computes RMS(x+res) * weight / rms,
//  writes output.  Half4 vectorized, SIMD-group per row, zero barriers.
//

#include <metal_stdlib>
using namespace metal;

enum : uint { THREADS = 256, ROWS = 8 };

[[host_name("rms_residual")]]
[[kernel, max_total_threads_per_threadgroup(THREADS)]]
void rms_residual(
    device const half* x,
    device const half* residual,
    device const half* weight,
    device half* y,
    constant uint& rows,
    constant uint& cols,
    constant float& eps,
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    const uint row = gid.y * ROWS + simd;
    if (row >= rows) return;

    const size_t off = (size_t)row * cols;
    const device half4* x4 = reinterpret_cast<const device half4*>(x + off);
    const device half4* r4 = reinterpret_cast<const device half4*>(residual + off);
    const device half4* w4 = reinterpret_cast<const device half4*>(weight);
    device half4* y4 = reinterpret_cast<device half4*>(y + off);

    // Pass 1: compute sum of squares
    float sq = 0;
    const uint n4 = cols / 4;
    for (uint k = lane; k < n4; k += 32) {
        float4 v = float4(x4[k]) + float4(r4[k]);
        sq += dot(v, v);
    }
    for (uint k = n4 * 4 + lane; k < cols; k += 32) {
        float v = float(x[off + k]) + float(residual[off + k]);
        sq += v * v;
    }
    sq = simd_sum(sq);
    float inv_rms = metal::precise::rsqrt(sq / float(cols) + eps);

    // Pass 2: write normalized output
    for (uint k = lane; k < n4; k += 32) {
        float4 v = (float4(x4[k]) + float4(r4[k])) * inv_rms;
        y4[k] = half4(v * float4(w4[k]));
    }
    for (uint k = n4 * 4 + lane; k < cols; k += 32) {
        float v = (float(x[off + k]) + float(residual[off + k])) * inv_rms;
        y[off + k] = half(v * float(weight[k]));
    }
}
