//
//  post_ssm.metal
//  SuperKittens — Mamba-3 post-SSM: gate(z) * ssm_out + RMSNorm
//
//  SIMD-group per position. gated = rms_norm(silu(z) * ssm_out)

#include <metal_stdlib>
using namespace metal;

enum : uint { THREADS = 128, ROWS = 4 };

[[host_name("mamba3_post_ssm")]]
[[kernel, max_total_threads_per_threadgroup(THREADS)]]
void mamba3_post_ssm(
    device const half* z,         // (BH, L, DV) — gate
    device const half* ssm_out,   // (BH, L, DV) — SSM output
    device const half* norm_w,    // (DV,) — RMSNorm weight
    device half* gated,           // (BH, L, DV) — gated + normalized output
    constant uint& L, constant uint& DV, constant float& eps,
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    const uint bh = gid.x, pos = gid.y * ROWS + simd;
    if (pos >= L) return;
    const size_t off = ((size_t)bh * L + pos) * DV;
    const device half4* z4 = reinterpret_cast<const device half4*>(z + off);
    const device half4* s4 = reinterpret_cast<const device half4*>(ssm_out + off);
    const device half4* w4 = reinterpret_cast<const device half4*>(norm_w);
    device half4* g4 = reinterpret_cast<device half4*>(gated + off);
    const uint n4 = DV / 4;

    float sumSq = 0.0f;

    // Pass 1: gate + accumulate sumSq for RMSNorm
    for (uint i = lane; i < n4; i += 32) {
        float4 zv = float4(z4[i]);
        float4 sv = float4(s4[i]);
        zv = zv / (1.0f + metal::fast::exp(-zv));  // silu(z)
        float4 gv = zv * sv;
        sumSq += gv.x*gv.x + gv.y*gv.y + gv.z*gv.z + gv.w*gv.w;
    }
    for (uint i = n4 * 4 + lane; i < DV; i += 32) {
        float zv = float(z[off + i]);
        zv = zv / (1.0f + metal::fast::exp(-zv));
        float gv = zv * float(ssm_out[off + i]);
        sumSq += gv * gv;
    }
    sumSq = simd_sum(sumSq);
    float inv_rms = metal::precise::rsqrt(sumSq / float(DV) + eps);

    // Pass 2: gate + normalize + apply weight
    for (uint i = lane; i < n4; i += 32) {
        float4 zv = float4(z4[i]);
        float4 sv = float4(s4[i]);
        zv = zv / (1.0f + metal::fast::exp(-zv));
        float4 gv = zv * sv;
        float4 wv = float4(w4[i]);
        g4[i] = half4(gv * inv_rms * wv);
    }
    for (uint i = n4 * 4 + lane; i < DV; i += 32) {
        float zv = float(z[off + i]);
        zv = zv / (1.0f + metal::fast::exp(-zv));
        float gv = zv * float(ssm_out[off + i]);
        gated[off + i] = half(gv * inv_rms * float(norm_w[i]));
    }
}
