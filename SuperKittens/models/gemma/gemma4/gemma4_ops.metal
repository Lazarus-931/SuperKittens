#include <metal_stdlib>
using namespace metal;

namespace meow::gemma4::ops {

inline float tg_sum_sq(threadgroup float* scratch, float x, uint tid, uint head_dim) {
    const uint sg = tid / 32;
    const uint ln = tid % 32;
    float v = simd_sum(x);
    if (ln == 0) scratch[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const uint n_sg = (head_dim + 31) / 32;
        float t = (ln < n_sg) ? scratch[ln] : 0.0f;
        t = simd_sum(t);
        if (ln == 0) scratch[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}

[[host_name("gemma4_rmsnorm_noscale")]]
[[kernel]]
void gemma4_rmsnorm_noscale(
    device const half* x          [[buffer(0)]],
    device       half* y          [[buffer(1)]],
    constant uint& T              [[buffer(2)]],
    constant uint& n_kv_heads     [[buffer(3)]],
    constant uint& head_dim       [[buffer(4)]],
    constant float& eps           [[buffer(5)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint head = gid.x;
    const uint t    = gid.y;
    if (t >= T || head >= n_kv_heads) return;

    const size_t row_off = ((size_t)t * n_kv_heads + head) * head_dim;

    float xv = (tid < head_dim) ? (float)x[row_off + tid] : 0.0f;

    threadgroup float scratch[32];
    float ssq = tg_sum_sq(scratch, xv * xv, tid, head_dim);
    float inv_rms = rsqrt(ssq / (float)head_dim + eps);

    if (tid >= head_dim) return;
    y[row_off + tid] = half(xv * inv_rms);
}

[[host_name("gemma4_rope_qk_partial")]]
[[kernel, max_total_threads_per_threadgroup(512)]]
void gemma4_rope_qk_partial(
    device half* Q          [[buffer(0)]],
    device half* K          [[buffer(1)]],
    device const half* cos  [[buffer(2)]],
    device const half* sin  [[buffer(3)]],
    constant uint& seq      [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& n_heads  [[buffer(6)]],
    constant uint& rot_dims [[buffer(7)]],
    constant uint& write_pos [[buffer(8)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tpg [[threads_per_threadgroup]])
{
    const uint head = gid.x;
    if (head >= n_heads) return;
    const uint row_blk = gid.y;
    const uint rows_per_tg = tpg.y;
    const uint row = row_blk * rows_per_tg + tid.y;
    if (row >= seq) return;

    // Gemma 4 proportional RoPE (HF modeling_rope_utils.py:187 _compute_proportional_rope_parameters)
    // Pairs are (x[i], x[i + head_dim/2]) for i in [0..head_dim/2); only first
    // rot_dims/2 pairs are rotated. cos/sin table stride is head_dim/2.
    const uint rot_half = rot_dims / 2;     // number of pairs to rotate (e.g., 64)
    const uint hd_half  = head_dim / 2;     // pair stride within the head (e.g., 256)
    const uint hd4 = rot_half / 4;
    const uint d4  = tid.x;
    if (d4 >= hd4) return;
    const uint i = d4 * 4;

    const size_t head_off = (size_t)head * seq * head_dim;
    const size_t qk_off = head_off + (size_t)row * head_dim;
    // cos/sin tables are baked with `head_dim/2` entries per position.
    const size_t cs_off = (size_t)(write_pos + row) * hd_half + i;

    float4 q_lo = float4(*reinterpret_cast<device half4*>(Q + qk_off + i));
    float4 q_hi = float4(*reinterpret_cast<device half4*>(Q + qk_off + i + hd_half));
    float4 k_lo = float4(*reinterpret_cast<device half4*>(K + qk_off + i));
    float4 k_hi = float4(*reinterpret_cast<device half4*>(K + qk_off + i + hd_half));

    float4 c  = float4(*reinterpret_cast<const device half4*>(cos + cs_off));
    float4 sv = float4(*reinterpret_cast<const device half4*>(sin + cs_off));

    *reinterpret_cast<device half4*>(Q + qk_off + i)            = half4(q_lo * c - q_hi * sv);
    *reinterpret_cast<device half4*>(Q + qk_off + i + hd_half)  = half4(q_lo * sv + q_hi * c);
    *reinterpret_cast<device half4*>(K + qk_off + i)            = half4(k_lo * c - k_hi * sv);
    *reinterpret_cast<device half4*>(K + qk_off + i + hd_half)  = half4(k_lo * sv + k_hi * c);
}

// per_layer_inputs context-mix: combines context projection with token identity.
//   per_layer_inputs[t, L, p] = ( RMSnorm(proj[t, L, p] * scale_proj, gamma) + token_id[t, L, p] ) * (1/sqrt(2))
// proj is (T, n_layers*ple_dim) and reinterpreted as (T, n_layers, ple_dim).
// Grid (n_layers, T) × TG (ple_dim).
//
// HF reference: modeling_gemma4.py:1757-1790 project_per_layer_inputs.
[[host_name("gemma4_ple_context_mix")]]
[[kernel]]
void gemma4_ple_context_mix(
    device const half*  proj           [[buffer(0)]],  // (T, n_layers, ple_dim)
    device const half*  gamma          [[buffer(1)]],  // (ple_dim,)
    device       half*  per_layer_in   [[buffer(2)]],  // (T, n_layers, ple_dim) in/out
    constant uint& T                   [[buffer(3)]],
    constant uint& n_layers            [[buffer(4)]],
    constant uint& ple_dim             [[buffer(5)]],
    constant float& scale_proj         [[buffer(6)]],  // 1/sqrt(d_model)
    constant float& scale_combine      [[buffer(7)]],  // 1/sqrt(2)
    constant float& eps                [[buffer(8)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint L = gid.x;
    const uint t = gid.y;
    if (L >= n_layers || t >= T) return;

    const size_t row_off = ((size_t)t * n_layers + L) * ple_dim;

    // 1) scale proj by scale_proj, compute mean of squares
    float xv = (tid < ple_dim) ? (float)proj[row_off + tid] * scale_proj : 0.0f;

    threadgroup float scratch[32];
    float sq = xv * xv;
    const uint sg = tid / 32, ln = tid % 32;
    float v = simd_sum(sq);
    if (ln == 0) scratch[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const uint n_sg = (ple_dim + 31) / 32;
        float t2 = (ln < n_sg) ? scratch[ln] : 0.0f;
        t2 = simd_sum(t2);
        if (ln == 0) scratch[0] = t2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float invrms = rsqrt(scratch[0] / (float)ple_dim + eps);

    if (tid >= ple_dim) return;
    float normed = xv * invrms * (float)gamma[tid];
    float tok    = (float)per_layer_in[row_off + tid];
    float out    = (normed + tok) * scale_combine;
    per_layer_in[row_off + tid] = (half)out;
}

[[host_name("gemma4_logit_softcap")]]
[[kernel]]
void gemma4_logit_softcap(
    device half* logits     [[buffer(0)]],
    constant uint& n        [[buffer(1)]],
    constant float& cap     [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    const uint i = gid * 4;
    if (i + 4 <= n) {
        device half4* p = reinterpret_cast<device half4*>(logits + i);
        float4 v = float4(*p);
        float inv = 1.0f / cap;
        float4 t = float4(precise::tanh(v.x * inv),
                          precise::tanh(v.y * inv),
                          precise::tanh(v.z * inv),
                          precise::tanh(v.w * inv));
        *p = half4(t * cap);
    } else {
        for (uint j = i; j < n; ++j) {
            float v = (float)logits[j];
            logits[j] = half(precise::tanh(v / cap) * cap);
        }
    }
}

// gemma4_logit_descale: divide logits by `inv_scale` (e.g., 1/sqrt(d_model)).
// Used because SK pre-scales w_embed by sqrt(d_model) at load, so the tied
// lm_head GEMM produces logits scaled UP. Apply this BEFORE softcap so the
// pre-softcap dump matches HF.
[[host_name("gemma4_logit_descale")]]
[[kernel]]
void gemma4_logit_descale(
    device half* logits      [[buffer(0)]],
    constant uint& n         [[buffer(1)]],
    constant float& inv_scale [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    const uint i = gid * 4;
    if (i + 4 <= n) {
        device half4* p = reinterpret_cast<device half4*>(logits + i);
        *p = half4(float4(*p) * inv_scale);
    } else {
        for (uint j = i; j < n; ++j) {
            logits[j] = half((float)logits[j] * inv_scale);
        }
    }
}

} // namespace meow::gemma4::ops
