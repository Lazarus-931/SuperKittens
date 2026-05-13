//
//  mamba3_decode_full.metal — end-to-end decode kernel for Mamba-3.
//
//  Model: Mamba-3 (single-token decode, terminal layer)
//  Fuses: mamba3_step  +  post_ssm  +  lm_head (gemv)  +  argmax sample
//
//  Used as the LAST layer of a decode loop. Takes the per-token inputs to
//  the SSM, carries (h_state, a_cs) state, projects through the LM head,
//  and returns the argmax token id. Eliminates three round-trips through
//  device memory between SSM, post-norm, projection, and sampling.
//
//  Caveats:
//    * Only valid as the final layer — earlier layers still need the
//      hidden activation, not a token id.
//    * Vocab size V is assumed reasonable (≤ ~128K). The argmax is a
//      threadgroup reduction; for larger vocabs we'd want a two-pass.
//    * No softmax sampling; argmax only. (Top-k / top-p / temperature
//      will be a separate `mamba3_decode_topk.metal` if you want them.)
//
//  Inputs (single token):
//    q_t (DQ),  k_t (DQ),  v_t (DV),  a_t (scalar),  b_t (scalar),
//    angle_t (DQ/2),  z_t (DV),
//    norm_w (DV)         — RMSNorm post-SSM weight
//    W_lm   (V, DV)      — LM head weight (row-major; one row per vocab token)
//  State (per BH, in/out):
//    h_state (DQ, DV) fp32
//    a_cs    (scalar) fp32
//  Output:
//    next_id (BH,) int32 — argmax token id
//

#include <metal_stdlib>
using namespace metal;

constant constexpr float PI = 3.14159265358979323846f;

[[host_name("mamba3_decode_full")]]
[[kernel]]
void mamba3_decode_full(
    device const half*  q_t        [[buffer(0)]],   // (BH, DQ)
    device const half*  k_t        [[buffer(1)]],
    device const half*  v_t        [[buffer(2)]],
    device const half*  a_t        [[buffer(3)]],   // (BH,)
    device const half*  b_t        [[buffer(4)]],
    device const half*  angle_t    [[buffer(5)]],   // (BH, DQ/2)
    device const half*  z_t        [[buffer(6)]],
    device const half*  norm_w     [[buffer(7)]],   // (DV,)
    device const half*  W_lm       [[buffer(8)]],   // (V, DV)
    device       float* h_state    [[buffer(9)]],
    device       float* a_cs_state [[buffer(10)]],
    device       int*   next_id    [[buffer(11)]],  // (BH,)
    constant uint& BH              [[buffer(12)]],
    constant uint& DQ              [[buffer(13)]],
    constant uint& DV              [[buffer(14)]],
    constant uint& V               [[buffer(15)]],  // vocab size
    constant float& eps            [[buffer(16)]],
    uint  bh   [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]],
    uint  tcnt [[threads_per_threadgroup]])
{
    const uint Dq2 = DQ / 2;

    // Re-uses identical math from mamba3_step_post.metal (steps 1-6).
    // Annotated minimally to avoid repetition; see that file for prose.

    // ── 1. Scalar update of (a_cs, b_scale, decay) ─────────────────────
    threadgroup float a_cs_new, b_scale, decay;
    if (tid == 0) {
        const float a_t_f = (float)a_t[bh];
        const float b_t_f = (float)b_t[bh];
        a_cs_new = a_cs_state[bh] + a_t_f;
        b_scale  = 1.0f + b_t_f * exp(-a_cs_new);
        decay    = exp(a_t_f) * b_scale;
        a_cs_state[bh] = a_cs_new;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 2. Rotary on q, k ──────────────────────────────────────────────
    threadgroup half q_rot[256], k_rot[256];
    if (tid < DQ) {
        const uint pair = tid % Dq2;
        const bool is_high = tid >= Dq2;
        const float ang = a_cs_new * (float)angle_t[bh * Dq2 + pair] * PI;
        const float c = cos(ang), s = sin(ang);
        const float q_lo = (float)q_t[bh * DQ + pair];
        const float q_hi = (float)q_t[bh * DQ + pair + Dq2];
        const float k_lo = (float)k_t[bh * DQ + pair];
        const float k_hi = (float)k_t[bh * DQ + pair + Dq2];
        q_rot[tid] = (half)(is_high ? q_lo * s + q_hi * c : q_lo * c - q_hi * s);
        k_rot[tid] = (half)(is_high ? k_lo * s + k_hi * c : k_lo * c - k_hi * s);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 3. State update ────────────────────────────────────────────────
    const uint nstate = DQ * DV;
    for (uint i = tid; i < nstate; i += tcnt) {
        const uint dq = i / DV;
        const uint dv = i % DV;
        const float prev = h_state[bh * nstate + i];
        const float kv = (float)k_rot[dq] * (float)v_t[bh * DV + dv];
        h_state[bh * nstate + i] = decay * prev + b_scale * kv;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ── 4. ssm_out = q_rot^T @ h_state ─────────────────────────────────
    threadgroup float ssm_out_tg[256];
    if (tid < DV) {
        float acc = 0.0f;
        for (uint dq = 0; dq < DQ; ++dq) {
            acc += (float)q_rot[dq] * h_state[bh * nstate + dq * DV + tid];
        }
        ssm_out_tg[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 5. SiLU gate ───────────────────────────────────────────────────
    threadgroup float gated_tg[256];
    if (tid < DV) {
        const float z = (float)z_t[bh * DV + tid];
        gated_tg[tid] = (z / (1.0f + exp(-z))) * ssm_out_tg[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 6. RMSNorm ─────────────────────────────────────────────────────
    threadgroup float reduce_buf[8];
    const uint sg   = tid / 32;
    const uint lane = tid % 32;

    float sq = (tid < DV) ? gated_tg[tid] * gated_tg[tid] : 0.0f;
    sq = simd_sum(sq);
    if (lane == 0) reduce_buf[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const uint n_sg = (tcnt + 31) / 32;
        float total = (lane < n_sg) ? reduce_buf[lane] : 0.0f;
        total = simd_sum(total);
        if (lane == 0) reduce_buf[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup half normed[256];
    const float rms = rsqrt(reduce_buf[0] / (float)DV + eps);
    if (tid < DV) {
        normed[tid] = (half)(gated_tg[tid] * rms * (float)norm_w[tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 7. LM head GEMV: logits[v] = W_lm[v, :] @ normed[:DV] ─────────
    //   Per-vocab dot product. Each thread computes one logit; loop over V/tcnt.
    //   Track running argmax in registers.
    float best_val   = -INFINITY;
    int   best_idx   = 0;

    for (uint v_idx = tid; v_idx < V; v_idx += tcnt) {
        float dot = 0.0f;
        for (uint d = 0; d < DV; ++d) {
            dot += (float)W_lm[v_idx * DV + d] * (float)normed[d];
        }
        if (dot > best_val) {
            best_val = dot;
            best_idx = (int)v_idx;
        }
    }

    // ── 8. Argmax reduction across the threadgroup ─────────────────────
    //   We keep a (val, idx) pair per thread. Reduce by max-on-val.
    threadgroup float vals[256];
    threadgroup int   idxs[256];
    vals[tid] = best_val;
    idxs[tid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction.
    for (uint stride = tcnt / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_v = vals[tid + stride];
            if (other_v > vals[tid]) {
                vals[tid] = other_v;
                idxs[tid] = idxs[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        next_id[bh] = idxs[0];
    }
}
