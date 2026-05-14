// gemma4_qkv_norm_rope_partial_t1.metal — fused qkv split + per-head RMSNorm
// + partial RoPE (Q,K) at T=1 (decode).
//
// Ported from temp/gemma4_kernel_lab/kernels.metal (lab_qkv_norm_rope_partial_t1).
// Replaces 3 dispatches at decode time on global (full-attention) layers:
//   gemma4_qkv_norm + gemma4_rope_qk_partial (Q) + gemma4_rope_qk_partial (K).
// V is normed without gamma and written straight to v_out (no RoPE on V).
//
// Grid: (n_heads + 2*n_kv_heads, 1, 1) threadgroups — one TG per slot.
// TG  : (head_dim, 1, 1) threads. Each TG handles one head.
//
// rot_dims = head_dim/4 for gemma4 global layers; cos/sin tables consumed
// at write_pos with stride head_dim/2 (matches gemma4_rope_qk_partial).

#include <metal_stdlib>
using namespace metal;

[[host_name("gemma4_qkv_norm_rope_partial_t1")]]
[[kernel, max_total_threads_per_threadgroup(512)]]
void gemma4_qkv_norm_rope_partial_t1(
    device const bfloat* qkv      [[buffer(0)]],
    device const bfloat* gamma_q  [[buffer(1)]],
    device const bfloat* gamma_k  [[buffer(2)]],
    device const bfloat* cos      [[buffer(3)]],
    device const bfloat* sin      [[buffer(4)]],
    device       bfloat* q_out    [[buffer(5)]],
    device       bfloat* k_out    [[buffer(6)]],
    device       bfloat* v_out    [[buffer(7)]],
    constant uint& n_heads      [[buffer(8)]],
    constant uint& n_kv_heads   [[buffer(9)]],
    constant uint& head_dim     [[buffer(10)]],
    constant uint& rot_dims     [[buffer(11)]],
    constant uint& write_pos    [[buffer(12)]],
    constant float& eps         [[buffer(13)]],
    uint  slot [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint q_end  = n_heads;
    const uint k_end  = n_heads + n_kv_heads;
    const uint kv_end = n_heads + 2u * n_kv_heads;
    if (slot >= kv_end) return;

    // Packed QKV at T=1: stride = (n_heads + 2*n_kv_heads) * head_dim.
    const uint slot_off = slot * head_dim;

    // ── RMSNorm reduction ──
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
    const float inv_rms = rsqrt(scratch[0] / (float)head_dim + eps);
    if (tid >= head_dim) return;

    // ── Apply gamma (Q/K) or none (V) ──
    bool is_v = (slot >= k_end);
    bool is_q = (slot < q_end);
    float g = is_v ? 1.0f : (is_q ? (float)gamma_q[tid] : (float)gamma_k[tid]);
    float y = x * inv_rms * g;

    // V: skip rope, write straight to v_out (head_in_kind=slot-k_end).
    if (is_v) {
        const uint h = slot - k_end;
        v_out[h * head_dim + tid] = bfloat(y);
        return;
    }

    // ── Apply partial RoPE for Q/K. ──
    const uint rot_half = rot_dims / 2;
    const uint hd_half  = head_dim / 2;

    threadgroup float ys[1024];  // accommodates head_dim up to 512
    ys[tid] = y;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each Q/K head writes (head, 0, tid) row at offset h * head_dim + tid.
    const uint h = is_q ? slot : (slot - q_end);
    device bfloat* out = is_q ? q_out : k_out;
    const size_t base = (size_t)h * head_dim;

    // Rotated region: pair (i, i+hd_half) for i < rot_half.
    if (tid < rot_half) {
        const uint i_lo = tid, i_hi = tid + hd_half;
        const float c  = (float)cos[write_pos * hd_half + i_lo];
        const float sv = (float)sin[write_pos * hd_half + i_lo];
        const float lo = ys[i_lo];
        const float hi = ys[i_hi];
        out[base + i_lo] = bfloat(lo * c - hi * sv);
    } else if (tid >= hd_half && tid < hd_half + rot_half) {
        const uint i_lo = tid - hd_half, i_hi = tid;
        const float c  = (float)cos[write_pos * hd_half + i_lo];
        const float sv = (float)sin[write_pos * hd_half + i_lo];
        const float lo = ys[i_lo];
        const float hi = ys[i_hi];
        out[base + i_hi] = bfloat(lo * sv + hi * c);
    } else {
        // Untouched (high freqs).
        out[base + tid] = bfloat(ys[tid]);
    }
}
