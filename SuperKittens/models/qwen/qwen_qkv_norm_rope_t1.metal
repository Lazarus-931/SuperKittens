// qwen_qkv_norm_rope_t1.metal — fused QKV split + per-head Q/K-RMSNorm + full
// NeoX-style RoPE(Q,K) at T=1 (decode). One dispatch replaces six tiny T=1
// dispatches in dispatch_layer: split_packed×2 + rmsnorm(Q) + rmsnorm(K) +
// qwen_rope_qk(Q) + qwen_rope_qk(K). At decode these are launch/barrier bound,
// not compute bound, so collapsing the inter-dispatch gap is the win.
//
// Grid: (n_heads + 2*n_kv_heads, 1, 1) threadgroups — one TG per head slot.
// TG  : (head_dim, 1, 1) threads. Slot layout matches qkv_packed at T=1:
//   [ Q(0..n_heads) | K(n_heads..+n_kv_heads) | V(..+n_kv_heads) ].
// Q/K heads: RMSNorm(gamma) then full split-half RoPE → q_out / k_out.
// V heads: copied straight to v_out (no norm, no RoPE).
//
// RoPE matches qwen_rope_qk (split-half / NeoX): for i < head_dim/2, pair
// (i, i+head_dim/2) rotated by cos/sin at row write_pos, table stride head_dim/2.

#include <metal_stdlib>
using namespace metal;

[[host_name("qwen_qkv_norm_rope_t1")]]
[[kernel, max_total_threads_per_threadgroup(256)]]
void qwen_qkv_norm_rope_t1(
    device const half* qkv      [[buffer(0)]],
    device const half* gamma_q  [[buffer(1)]],
    device const half* gamma_k  [[buffer(2)]],
    device const half* cos      [[buffer(3)]],
    device const half* sin      [[buffer(4)]],
    device       half* q_out    [[buffer(5)]],
    device       half* k_out    [[buffer(6)]],
    device       half* v_out    [[buffer(7)]],
    constant uint& n_heads      [[buffer(8)]],
    constant uint& n_kv_heads   [[buffer(9)]],
    constant uint& head_dim     [[buffer(10)]],
    constant uint& write_pos    [[buffer(11)]],
    constant float& eps         [[buffer(12)]],
    uint  slot [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint q_end  = n_heads;
    const uint k_end  = n_heads + n_kv_heads;
    const uint kv_end = n_heads + 2u * n_kv_heads;
    if (slot >= kv_end) return;

    // qkv_packed at T=1 is contiguous [Q|K|V]; slot indexes head_dim-sized bands.
    const uint slot_off = slot * head_dim;

    // ── RMSNorm reduction over head_dim ──
    float x = (tid < head_dim) ? (float)qkv[slot_off + tid] : 0.0f;
    float ssq = x * x;
    threadgroup float scratch[32];
    float v = simd_sum(ssq);
    uint sg = tid / 32, ln = tid % 32;
    if (ln == 0) scratch[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const uint n_sg = (head_dim + 31) / 32;
        float t = (ln < n_sg) ? scratch[ln] : 0.0f;
        t = simd_sum(t);
        if (ln == 0) scratch[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = metal::precise::rsqrt(scratch[0] / (float)head_dim + eps);
    if (tid >= head_dim) return;

    const bool is_v = (slot >= k_end);
    const bool is_q = (slot < q_end);

    // V: no gamma, no RoPE — copy normed? No: V is NOT normed/roped in qwen.
    // The original split routes V straight from qkv_packed to v_tmp untouched,
    // so write the raw value (not x*inv_rms).
    if (is_v) {
        const uint h = slot - k_end;
        v_out[h * head_dim + tid] = (half)qkv[slot_off + tid];
        return;
    }

    const float g = is_q ? (float)gamma_q[tid] : (float)gamma_k[tid];
    const float y = x * inv_rms * g;

    // ── Full split-half RoPE for Q/K ──
    const uint hd_half = head_dim / 2;

    threadgroup float ys[256];
    ys[tid] = y;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint h = is_q ? slot : (slot - q_end);
    device half* out = is_q ? q_out : k_out;
    const size_t base = (size_t)h * head_dim;
    const size_t cs_base = (size_t)write_pos * hd_half;

    if (tid < hd_half) {
        const float c  = (float)cos[cs_base + tid];
        const float sv = (float)sin[cs_base + tid];
        const float lo = ys[tid];
        const float hi = ys[tid + hd_half];
        out[base + tid] = (half)(lo * c - hi * sv);
    } else {
        const uint i = tid - hd_half;
        const float c  = (float)cos[cs_base + i];
        const float sv = (float)sin[cs_base + i];
        const float lo = ys[i];
        const float hi = ys[tid];
        out[base + tid] = (half)(lo * sv + hi * c);
    }
}
