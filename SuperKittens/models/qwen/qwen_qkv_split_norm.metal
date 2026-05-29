//  qwen_qkv_split_norm.metal — fused QKV-split + per-head Q/K RMSNorm.

#include <metal_stdlib>
using namespace metal;

// WHY: collapses split_packed(q|kv) + split_packed(k|v) + per-head Q-RMSNorm +
// per-head K-RMSNorm (4 small fenced dispatches) into one. Each threadgroup is
// a single simdgroup that owns one (token, head) row of the packed QKV tensor:
// Q/K heads are RMSNorm'd (γ from w_q_norm/w_k_norm) into the split q/k buffers,
// V heads are copied verbatim into v. Reduction mirrors rms_norm.metal exactly
// (simd_sum over head_dim, precise::rsqrt) so the math is bit-identical.
[[host_name("qwen_qkv_split_norm")]]
[[kernel, max_total_threads_per_threadgroup(32)]]
void qwen_qkv_split_norm(
    device const half*  qkv        [[buffer(0)]],  // (T, (Hq+2*Hkv)*D) packed
    device       half*  q_out      [[buffer(1)]],  // (T, Hq*D)
    device       half*  k_out      [[buffer(2)]],  // (T, Hkv*D)
    device       half*  v_out      [[buffer(3)]],  // (T, Hkv*D)
    device const half*  gamma_q    [[buffer(4)]],  // (D,) this layer
    device const half*  gamma_k    [[buffer(5)]],  // (D,) this layer
    constant uint&      T          [[buffer(6)]],
    constant uint&      n_heads    [[buffer(7)]],  // Hq
    constant uint&      n_kv_heads [[buffer(8)]],  // Hkv
    constant uint&      head_dim   [[buffer(9)]],  // D
    constant float&     eps        [[buffer(10)]],
    uint  lane [[thread_index_in_simdgroup]],
    uint2 gid  [[threadgroup_position_in_grid]])
{
    const uint t    = gid.y;
    const uint slot = gid.x;             // 0..(Hq+2*Hkv)-1
    if (t >= T) return;

    const uint D   = head_dim;
    const uint Hq  = n_heads;
    const uint Hkv = n_kv_heads;
    const uint n_slots = Hq + 2u * Hkv;
    if (slot >= n_slots) return;

    const uint qkv_stride = n_slots * D;
    const size_t src_off  = (size_t)t * qkv_stride + (size_t)slot * D;
    const device half4* s4 = reinterpret_cast<const device half4*>(qkv + src_off);

    // V heads: pure split (copy), no norm.
    if (slot >= Hq + Hkv) {
        const uint kvh = slot - (Hq + Hkv);
        const size_t dst_off = ((size_t)t * Hkv + kvh) * D;
        device half4* d4 = reinterpret_cast<device half4*>(v_out + dst_off);
        const uint n4 = D / 4u;
        for (uint k = lane; k < n4; k += 32u) d4[k] = s4[k];
        for (uint k = n4 * 4u + lane; k < D; k += 32u)
            v_out[dst_off + k] = qkv[src_off + k];
        return;
    }

    // Q or K head: RMSNorm with the matching per-head γ, then split-write.
    device       half* dst;
    const device half* gamma;
    if (slot < Hq) {
        dst = q_out + (size_t)(t * Hq + slot) * D;
        gamma = gamma_q;
    } else {
        const uint kvh = slot - Hq;
        dst = k_out + (size_t)(t * Hkv + kvh) * D;
        gamma = gamma_k;
    }
    const device half4* g4 = reinterpret_cast<const device half4*>(gamma);
    device       half4* d4 = reinterpret_cast<device       half4*>(dst);

    const uint n4 = D / 4u;
    float sumSq = 0.0f;
    for (uint k = lane; k < n4; k += 32u) {
        float4 v = float4(s4[k]);
        sumSq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    for (uint k = n4 * 4u + lane; k < D; k += 32u) {
        float v = float(qkv[src_off + k]);
        sumSq += v * v;
    }
    sumSq = simd_sum(sumSq);
    float inv_rms = metal::precise::rsqrt(sumSq / float(D) + eps);

    for (uint k = lane; k < n4; k += 32u) {
        float4 v  = float4(s4[k]);
        float4 gv = float4(g4[k]);
        d4[k] = half4(v * inv_rms * gv);
    }
    for (uint k = n4 * 4u + lane; k < D; k += 32u) {
        dst[k] = half(float(qkv[src_off + k]) * inv_rms * float(gamma[k]));
    }
}
