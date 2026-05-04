//
//  rms_rope.metal — fused RMSNorm + RoPE on Q and K
//
//  Two-pass per tensor: compute RMS then apply norm+rotation.
//  128 threads, 4 SIMD groups, 4 rows per threadgroup.
//

#include <metal_stdlib>
using namespace metal;

enum : uint { THREADS = 256, ROWS = 8 };

[[host_name("rms_rope")]]
[[kernel, max_total_threads_per_threadgroup(THREADS)]]
void rms_rope(
    device const half* Q_in, device const half* K_in,
    device const half* q_weight, device const half* k_weight,
    device const half* cos, device const half* sin,
    device half* Q_out, device half* K_out,
    constant uint& heads, constant uint& seq, constant uint& head_dim, constant float& eps,
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    const uint head = gid.x;
    const uint row  = gid.y * ROWS + simd;
    if (head >= heads || row >= seq) return;

    const uint hd = head_dim / 2, hd4 = hd / 4, n4 = head_dim / 4;
    const size_t off = ((size_t)head * seq + row) * head_dim;
    const size_t cs_off = (size_t)row * hd;

    const device half4* q4_in = reinterpret_cast<const device half4*>(Q_in + off);
    const device half4* k4_in = reinterpret_cast<const device half4*>(K_in + off);
    device half4* q4_out = reinterpret_cast<device half4*>(Q_out + off);
    device half4* k4_out = reinterpret_cast<device half4*>(K_out + off);

    // RMSNorm: compute sq for Q and K
    float q_sq = 0, k_sq = 0;
    for (uint i = lane; i < n4; i += 32) {
        float4 v = float4(q4_in[i]); q_sq += dot(v, v);
        v = float4(k4_in[i]);        k_sq += dot(v, v);
    }
    q_sq = simd_sum(q_sq); k_sq = simd_sum(k_sq);
    float q_inv = metal::precise::rsqrt(q_sq / float(head_dim) + eps);
    float k_inv = metal::precise::rsqrt(k_sq / float(head_dim) + eps);

    const device half4* qw4 = reinterpret_cast<const device half4*>(q_weight);
    const device half4* kw4 = reinterpret_cast<const device half4*>(k_weight);
    const device half4* c4 = reinterpret_cast<const device half4*>(cos + cs_off);
    const device half4* s4 = reinterpret_cast<const device half4*>(sin + cs_off);

    // Apply norm + RoPE in one sweep
    for (uint i = lane; i < hd4; i += 32) {
        float4 c = float4(c4[i]), s = float4(s4[i]);

        float4 q_lo = float4(q4_in[i]) * q_inv * float4(qw4[i]);
        float4 q_hi = float4(q4_in[i + hd4]) * q_inv * float4(qw4[i + hd4]);
        q4_out[i]       = half4(q_lo * c - q_hi * s);
        q4_out[i + hd4] = half4(q_lo * s + q_hi * c);

        float4 k_lo = float4(k4_in[i]) * k_inv * float4(kw4[i]);
        float4 k_hi = float4(k4_in[i + hd4]) * k_inv * float4(kw4[i + hd4]);
        k4_out[i]       = half4(k_lo * c - k_hi * s);
        k4_out[i + hd4] = half4(k_lo * s + k_hi * c);
    }
}
