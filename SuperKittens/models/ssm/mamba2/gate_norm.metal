//
//  gate_norm.metal
//  SuperKittens — gate(ssm_out * silu(z)) + RMSNorm for Mamba2
//
//  y = rmsnorm(ssm_out * silu(z)) * weight
//  SIMD-group per row, no barriers.

#include <metal_stdlib>
using namespace metal;

enum : uint { THREADS = 128, ROWS = 4 };

[[host_name("gate_norm")]]
[[kernel, max_total_threads_per_threadgroup(THREADS)]]
void gate_norm(
    device const half* ssm_out,  // (B, L, E)
    device const half* z,        // (B, L, E) — gate
    device const half* weight,   // (E,) — RMSNorm weight
    device half* y,              // (B, L, E) — output
    constant uint& L,
    constant uint& E,
    constant float& eps,
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint2 gid  [[threadgroup_position_in_grid]])
{
    const uint row = gid.y * ROWS + simd;
    if (row >= L) return;

    const size_t off = (size_t)row * E;
    const device half4* s4 = reinterpret_cast<const device half4*>(ssm_out + off);
    const device half4* z4 = reinterpret_cast<const device half4*>(z + off);
    const device half4* w4 = reinterpret_cast<const device half4*>(weight);
    device half4* y4 = reinterpret_cast<device half4*>(y + off);

    float sumSq = 0.0f;
    const uint n4 = E / 4;

    // Pass 1: gate + accumulate sumSq
    for (uint k = lane; k < n4; k += 32) {
        float4 sv = float4(s4[k]);
        float4 zv = float4(z4[k]);
        float4 gv = sv * (zv / (1.0f + metal::fast::exp(-zv)));  // silu(z) * ssm_out
        sumSq += gv.x*gv.x + gv.y*gv.y + gv.z*gv.z + gv.w*gv.w;
    }
    for (uint k = n4 * 4 + lane; k < E; k += 32) {
        float sv = float(ssm_out[off + k]);
        float zv = float(z[off + k]);
        float gv = sv * (zv / (1.0f + metal::fast::exp(-zv)));
        sumSq += gv * gv;
    }
    sumSq = simd_sum(sumSq);

    float inv_rms = metal::precise::rsqrt(sumSq / float(E) + eps);

    // Pass 2: normalize + weight
    for (uint k = lane; k < n4; k += 32) {
        float4 sv = float4(s4[k]);
        float4 zv = float4(z4[k]);
        float4 gv = sv * (zv / (1.0f + metal::fast::exp(-zv)));
        float4 wv = float4(w4[k]);
        y4[k] = half4(gv * inv_rms * wv);
    }
    for (uint k = n4 * 4 + lane; k < E; k += 32) {
        float sv = float(ssm_out[off + k]);
        float zv = float(z[off + k]);
        float gv = sv * (zv / (1.0f + metal::fast::exp(-zv)));
        y[off + k] = half(gv * inv_rms * float(weight[k]));
    }
}
