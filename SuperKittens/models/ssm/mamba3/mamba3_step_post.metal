//
//  mamba3_step_post.metal — fused decode kernel for Mamba-3.
//
//  Model: Mamba-3 (single-token decode)
//  Fuses: mamba3_step  +  post_ssm  (silu-gate + RMSNorm)
//
//  One TG per (BH). One token in, one token out. Carries persistent
//  (h_state, a_cs) between calls. Eliminates the round trip through
//  global memory between the SSM step and the gate/norm.
//
//  Inputs  (per BH, single token):
//    q_t (DQ),  k_t (DQ),  v_t (DV),
//    a_t (scalar),  b_t (scalar),  angle_t (DQ/2),
//    z_t (DV)        — gating signal (SiLU branch)
//    norm_w (DV)     — RMSNorm weight
//  State (per BH, in/out):
//    h_state (DQ, DV) fp32,
//    a_cs    (scalar) fp32
//  Output (per BH):
//    y_t (DV) fp16   — final token output (gated + normed)
//
//  Math:
//    a_cs       += a_t
//    b_scale    = 1 + b_t * exp(-a_cs)
//    theta      = a_cs * angle_t * PI                 (DQ/2)
//    q_rot,k_rot = rotate(q_t, k_t, theta)            (DQ)
//    decay      = exp(a_t) * b_scale                  (scalar — single-step, NOT cumsum)
//    h_state   = decay * h_state + b_scale * outer(k_rot, v_t)   (DQ, DV)
//    ssm_out   = q_rot^T @ h_state                    (DV)
//    gated     = silu(z_t) * ssm_out                  (DV)
//    rms       = rsqrt(mean(gated^2) + eps)
//    y_t       = gated * rms * norm_w                 (DV)
//

#include <metal_stdlib>
using namespace metal;

constant constexpr float PI = 3.14159265358979323846f;

// Tunable: this fused kernel assumes DQ <= 128, DV <= 128. Threadgroup memory
// for h_state at fp32 = DQ*DV*4 = 64 KB worst case — exceeds the 32 KB budget
// on M2. So h_state is held in DEVICE memory (4 KB at DQ=DV=32, 16 KB at 64,
// 64 KB at 128 — large but persistent across calls anyway). This is the same
// trade the unfused step kernel makes; we don't pay re-load cost here because
// h_state would round-trip through device memory between step and post_ssm
// anyway.

[[host_name("mamba3_step_post")]]
[[kernel]]
void mamba3_step_post(
    device const half*  q_t        [[buffer(0)]],   // (BH, DQ)
    device const half*  k_t        [[buffer(1)]],   // (BH, DQ)
    device const half*  v_t        [[buffer(2)]],   // (BH, DV)
    device const half*  a_t        [[buffer(3)]],   // (BH,)
    device const half*  b_t        [[buffer(4)]],   // (BH,)
    device const half*  angle_t    [[buffer(5)]],   // (BH, DQ/2)
    device const half*  z_t        [[buffer(6)]],   // (BH, DV)
    device const half*  norm_w     [[buffer(7)]],   // (DV,)
    device       float* h_state    [[buffer(8)]],   // (BH, DQ, DV)  in/out
    device       float* a_cs_state [[buffer(9)]],   // (BH,)         in/out
    device       half*  y_t        [[buffer(10)]],  // (BH, DV)
    constant uint& BH              [[buffer(11)]],
    constant uint& DQ              [[buffer(12)]],
    constant uint& DV              [[buffer(13)]],
    constant float& eps            [[buffer(14)]],
    uint  bh   [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]],
    uint  tcnt [[threads_per_threadgroup]])
{
    const uint Dq2 = DQ / 2;

    // ── 1. Update a_cs (scalar) ────────────────────────────────────────
    threadgroup float a_t_f, a_cs_new, b_t_f, b_scale, decay;
    if (tid == 0) {
        a_t_f    = (float)a_t[bh];
        b_t_f    = (float)b_t[bh];
        a_cs_new = a_cs_state[bh] + a_t_f;
        b_scale  = 1.0f + b_t_f * exp(-a_cs_new);
        decay    = exp(a_t_f) * b_scale;
        a_cs_state[bh] = a_cs_new;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 2. Rotate q_t, k_t with theta = a_cs_new * angle_t * PI ───────
    // Each thread handles one DQ index (assume DQ <= tcnt).
    threadgroup half q_rot[256], k_rot[256];   // upper bound DQ <= 256
    if (tid < DQ) {
        const uint pair    = tid % Dq2;
        const bool is_high = tid >= Dq2;
        const float ang    = a_cs_new * (float)angle_t[bh * Dq2 + pair] * PI;
        const float c = cos(ang), s = sin(ang);

        const float q_lo = (float)q_t[bh * DQ + pair];
        const float q_hi = (float)q_t[bh * DQ + pair + Dq2];
        const float k_lo = (float)k_t[bh * DQ + pair];
        const float k_hi = (float)k_t[bh * DQ + pair + Dq2];

        if (is_high) {
            q_rot[tid] = (half)(q_lo * s + q_hi * c);
            k_rot[tid] = (half)(k_lo * s + k_hi * c);
        } else {
            q_rot[tid] = (half)(q_lo * c - q_hi * s);
            k_rot[tid] = (half)(k_lo * c - k_hi * s);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 3. h_state = decay * h_state + b_scale * outer(k_rot, v_t) ────
    //    Cooperative across DQ * DV elements. Each thread owns a slice.
    const uint nstate = DQ * DV;
    for (uint i = tid; i < nstate; i += tcnt) {
        const uint dq = i / DV;
        const uint dv = i % DV;
        const float prev = h_state[bh * nstate + i];
        const float kv   = (float)k_rot[dq] * (float)v_t[bh * DV + dv];
        h_state[bh * nstate + i] = decay * prev + b_scale * kv;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ── 4. ssm_out[dv] = sum_dq( q_rot[dq] * h_state[dq, dv] )  → device-mem reduce
    //    Each thread owns one (dv) slot.
    threadgroup float ssm_out_tg[256];   // upper bound DV <= 256
    if (tid < DV) {
        float acc = 0.0f;
        for (uint dq = 0; dq < DQ; ++dq) {
            acc += (float)q_rot[dq] * h_state[bh * nstate + dq * DV + tid];
        }
        ssm_out_tg[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 5. SiLU gate: gated[dv] = silu(z_t[dv]) * ssm_out[dv] ─────────
    //    silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    threadgroup float gated_tg[256];
    if (tid < DV) {
        const float z = (float)z_t[bh * DV + tid];
        const float silu = z / (1.0f + exp(-z));
        gated_tg[tid] = silu * ssm_out_tg[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 6. RMSNorm: y = gated * rsqrt(mean(gated^2) + eps) * norm_w ──
    //    Sum-of-squares reduction across DV using simdgroup + threadgroup.
    threadgroup float partial[8];   // up to 8 simdgroups per TG
    const uint sg = tid / 32;
    const uint lane = tid % 32;

    float sq = (tid < DV) ? gated_tg[tid] * gated_tg[tid] : 0.0f;
    sq = simd_sum(sq);
    if (lane == 0) partial[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        const uint n_sg = (tcnt + 31) / 32;
        float total = (lane < n_sg) ? partial[lane] : 0.0f;
        total = simd_sum(total);
        if (lane == 0) partial[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float rms = rsqrt(partial[0] / (float)DV + eps);

    if (tid < DV) {
        y_t[bh * DV + tid] = (half)(gated_tg[tid] * rms * (float)norm_w[tid]);
    }
}
