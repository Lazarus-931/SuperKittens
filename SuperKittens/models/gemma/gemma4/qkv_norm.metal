//
//  qkv_norm.metal — Gemma 4 QKV split + per-head RMSNorm.
//  Q/K with γ; V without γ (Gemma 4 specific, replaces logit softcap).
//  Grid (n_heads + 2*n_kv_heads, T) × TG (head_dim).
//

#include <metal_stdlib>
using namespace metal;

namespace meow::gemma4::qkvn {

// Sum-of-squares reduction across threadgroup (head_dim threads).
// Returns the scalar sum on every thread (broadcast via threadgroup mem).
inline float tg_sum_sq(threadgroup float* scratch, float x, uint tid, uint head_dim) {
    const uint sg = tid / 32;
    const uint ln = tid % 32;
    float v = simd_sum(x);                  // per-simdgroup partial
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

[[host_name("gemma4_qkv_norm")]]
[[kernel]]
void gemma4_qkv_norm(
    device const bfloat* qkv          [[buffer(0)]],
    device const bfloat* gamma_q      [[buffer(1)]],
    device const bfloat* gamma_k      [[buffer(2)]],
    device       bfloat* q_out        [[buffer(3)]],
    device       bfloat* k_out        [[buffer(4)]],
    device       bfloat* v_out        [[buffer(5)]],
    constant uint& T                [[buffer(6)]],
    constant uint& n_heads          [[buffer(7)]],
    constant uint& n_kv_heads       [[buffer(8)]],
    constant uint& head_dim         [[buffer(9)]],
    constant float& eps             [[buffer(10)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint slot = gid.x;          // 0..(n_heads+2*n_kv_heads)
    const uint t    = gid.y;          // 0..T
    if (t >= T) return;

    const uint q_end  = n_heads;
    const uint k_end  = n_heads + n_kv_heads;
    const uint kv_end = n_heads + 2u * n_kv_heads;
    if (slot >= kv_end) return;

    // QKV packed stride per token = (n_heads + 2*n_kv_heads) * head_dim.
    const uint qkv_stride = kv_end * head_dim;
    const size_t qkv_row_off = (size_t)t * qkv_stride;

    // Per-slot starting offset within the packed row.
    const uint slot_off = slot * head_dim;
    const size_t row_off = qkv_row_off + slot_off;

    // Each thread owns one dim. Read.
    float x = (tid < head_dim) ? (float)qkv[row_off + tid] : 0.0f;

    // Sum of squares across the row (head_dim threads).
    threadgroup float scratch[32];   // up to 1024 threads = 32 simdgroups
    float ssq = tg_sum_sq(scratch, x * x, tid, head_dim);
    float inv_rms = rsqrt(ssq / (float)head_dim + eps);

    if (tid >= head_dim) return;

    // Apply γ depending on slot kind. V uses inv_rms only (no γ).
    float y;
    if (slot < q_end) {
        y = x * inv_rms * (float)gamma_q[tid];
    } else if (slot < k_end) {
        y = x * inv_rms * (float)gamma_k[tid];
    } else {
        y = x * inv_rms;
    }

    // Write to the appropriate split output. The within-buffer row is
    // (t, head_in_kind, tid) row-major.
    if (slot < q_end) {
        const uint head = slot;
        size_t out_off = ((size_t)head * T + t) * head_dim + tid;
        q_out[out_off] = bfloat(y);
    } else if (slot < k_end) {
        const uint head = slot - q_end;
        size_t out_off = ((size_t)head * T + t) * head_dim + tid;
        k_out[out_off] = bfloat(y);
    } else {
        const uint head = slot - k_end;
        size_t out_off = ((size_t)head * T + t) * head_dim + tid;
        v_out[out_off] = bfloat(y);
    }
}

} // namespace meow::gemma4::qkvn
