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
    // Per-token stride (elements) of the x/B/C buffers. They alias one
    // interleaved [x(E)|B(G*N)|C(G*N)] tensor with stride C_in, not the
    // packed H*P / G*N strides their separate-tensor shapes would imply.
    constant uint&  XBC_stride     [[buffer(17)]],
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

    threadgroup float partial[256];  // tg-scope: declaring inside the t-loop is UB

    const float A = -exp((float)A_log[h]);
    const float Db = (float)D_in[h];

    for (uint t = 0; t < L; ++t) {
        // dt/dA/x_t are head-/position-scalars (depend on h,p only), identical
        // for every thread in this threadgroup — recompute per-thread rather
        // than broadcast via threadgroup memory (the broadcast needed a leading
        // barrier the loop lacked, corrupting t>=1).
        float dtr = (float)dt_raw[((size_t)b * L + t) * H + h]
                   + (float)dt_bias[h];
        float dt;
        if (dtr > 20.0f) dt = dtr;
        else dt = log(1.0f + exp(dtr));
        // HF time_step_limit clamp. dt_max is +inf by default (no upper bound);
        // fast-math drops infs, so gate on a magnitude sentinel.
        dt = max(dt, dt_min);
        if (dt_max < 1e30f) dt = min(dt, dt_max);
        const float dA  = exp(dt * A);
        const size_t tok = ((size_t)b * L + t) * (size_t)XBC_stride;
        const float x_t = (float)x[tok + (size_t)h * P + p];

        const size_t bcn_base = tok + (size_t)g * Nstate;
        const float Bv = (float)B_in[bcn_base + n];
        const float Cv = (float)C_in[bcn_base + n];

        // state update: s = dA*s + dt*B*x
        s = dA * s + dt * Bv * x_t;

        // Per-thread partial y contribution = C[n] * s[n]; then tg reduce.
        partial[n] = Cv * s;
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
