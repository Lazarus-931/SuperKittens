//
//  mamba2_ssd_ref.metal
//  Reference (non-chunked) Mamba 2 SSD prefill kernel.
//
//  Matches HF transformers Mamba2Mixer.torch_forward
//  (transformers/models/mamba2/modeling_mamba2.py, ~lines 398-460 for the
//  no-cuda fallback path), using the explicit per-token recurrence rather than
//  the chunked associative scan. This is the simplest signature-correct
//  reference; chunking for perf is a follow-up.
//
//  Per-token (decode path identical, just L=1):
//     dt = softplus(dt_raw[t,h] + dt_bias[h]).clamp(dt_min, dt_max)
//     A  = -exp(A_log[h])              (one scalar per head)
//     dA = exp(dt * A)                                                    (H,)
//     dBx[p,n] = dt * B[t,g,n] * x[t,h,p]              g = h / (H/G)
//     state[h,p,n] = dA * state[h,p,n] + dBx[p,n]
//     y[t,h,p] = sum_n C[t,g,n] * state[h,p,n] + D[h] * x[t,h,p]
//
//  Layout (all fp16 unless noted):
//     x        : (B, L, H, P)
//     dt_raw   : (B, L, H)
//     A_log    : (H,)
//     dt_bias  : (H,)
//     B_in     : (B, L, G, N)
//     C_in     : (B, L, G, N)
//     D        : (H,)
//     y_out    : (B, L, H, P)
//     ssm_state: (B, H, P, N)  in-out, fp32
//
//  Grid: (B*H, P, 1), threads/tg: N (state dim, 128).
//  Each thread owns one (h, p, n) entry of the ssm_state.

#include <metal_stdlib>
using namespace metal;

constant uint N_MAX = 256;

[[host_name("mamba2_ssd_ref")]]
[[kernel]]
void mamba2_ssd_ref(
    device const half*  x         [[buffer(0)]],   // (B,L,H,P)
    device const half*  dt_raw    [[buffer(1)]],   // (B,L,H)
    device const half*  A_log     [[buffer(2)]],   // (H,)
    device const half*  B_in      [[buffer(3)]],   // (B,L,G,N)
    device const half*  C_in      [[buffer(4)]],   // (B,L,G,N)
    device const half*  D_in      [[buffer(5)]],   // (H,)
    device const half*  dt_bias   [[buffer(6)]],   // (H,)
    device       half*  y_out     [[buffer(7)]],   // (B,L,H,P)
    device       float* ssm_state [[buffer(8)]],   // (B,H,P,N) fp32 in-out
    constant uint& Batch          [[buffer(9)]],
    constant uint& L              [[buffer(10)]],
    constant uint& H              [[buffer(11)]],
    constant uint& P              [[buffer(12)]],
    constant uint& G              [[buffer(13)]],
    constant uint& Nstate         [[buffer(14)]],
    constant float& dt_min        [[buffer(15)]],
    constant float& dt_max        [[buffer(16)]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint3 gid  [[threadgroup_position_in_grid]])
{
    const uint bh = gid.x;          // 0..B*H
    const uint p  = gid.y;          // 0..P
    const uint n  = lid;            // 0..N
    if (n >= Nstate) return;

    const uint b   = bh / H;
    const uint h   = bh % H;
    const uint hpg = H / G;         // heads per group
    const uint g   = h / hpg;

    // Per-thread persistent state element s[h,p,n].
    const size_t st_idx =
        ((size_t)b * H + h) * (size_t)P * (size_t)Nstate
        + (size_t)p * (size_t)Nstate + (size_t)n;
    float s = ssm_state[st_idx];

    // Cached head-broadcast scalars (computed per token):
    threadgroup float dt_t;
    threadgroup float dA_t;
    threadgroup float x_t;     // x[b,t,h,p]
    threadgroup float B_tn[256];  // we hold B for this group across N
    threadgroup float C_tn[256];

    const float A = -exp((float)A_log[h]);
    const float Db = (float)D_in[h];

    for (uint t = 0; t < L; ++t) {
        // One thread loads scalars.
        if (n == 0) {
            float dtr = (float)dt_raw[((size_t)b * L + t) * H + h]
                       + (float)dt_bias[h];
            // softplus
            float dt;
            if (dtr > 20.0f) dt = dtr;
            else dt = log(1.0f + exp(dtr));
            dt = clamp(dt, dt_min, dt_max);
            dt_t = dt;
            dA_t = exp(dt * A);
            x_t  = (float)x[(((size_t)b * L + t) * H + h) * P + p];
        }
        // All threads load B,C for their n.
        const size_t bcn_base = (((size_t)b * L + t) * G + g) * Nstate;
        B_tn[n] = (float)B_in[bcn_base + n];
        C_tn[n] = (float)C_in[bcn_base + n];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // state update: s = dA*s + dt*B*x
        s = dA_t * s + dt_t * B_tn[n] * x_t;

        // Per-thread partial y contribution = C[n] * s[n]; then tg reduce.
        threadgroup float partial[256];
        partial[n] = C_tn[n] * s;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Tree reduction over N (assumes Nstate is a power of 2, e.g. 128).
        for (uint stride = Nstate / 2; stride > 0; stride >>= 1) {
            if (n < stride) partial[n] += partial[n + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (n == 0) {
            float y_v = partial[0] + Db * x_t;
            y_out[(((size_t)b * L + t) * H + h) * P + p] = (half)y_v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Persist updated state.
    ssm_state[st_idx] = s;
}
