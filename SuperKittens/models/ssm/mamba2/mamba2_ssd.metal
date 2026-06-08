//
//  mamba2_ssd.metal
//  Mamba 2 SSD prefill (production path).
//
//  Numerically equivalent to mamba2_ssd_ref.metal and HF transformers
//  Mamba2Mixer.torch_forward (the per-token scan and the chunked associative
//  scan are algebraically the same selective state-space recurrence). Identical
//  buffer/dispatch signature to mamba2_ssd_ref so the launcher binds it the
//  same way.
//
//  Per token, per head h (group g = h / (H/G)):
//     dt = softplus(dt_raw[t,h] + dt_bias[h]).clamp(dt_min, dt_max)
//     dA = exp(dt * (-exp(A_log[h])))
//     s[p,n] = dA * s[p,n] + dt * B[t,g,n] * x[t,h,p]
//     y[t,h,p] = sum_n C[t,g,n] * s[p,n] + D[h] * x[t,h,p]
//
//  WHY one simdgroup with register-resident state instead of the ref's
//  N-thread tg-reduction: the ref pays log2(N) threadgroup barriers per token
//  for the C·s reduction plus a barrier to broadcast scalars. Here a single
//  simdgroup (32 lanes) holds the whole (p,·) state row in registers
//  (N/32 elements/lane), so the per-token reduction is one simd_sum with no
//  threadgroup barriers and no shared-memory traffic.
//
//  Layout (all fp16 unless noted):
//     x        : (B, L, H, P)
//     dt_raw   : (B, L, H)
//     A_log    : (H,)
//     B_in     : (B, L, G, N)
//     C_in     : (B, L, G, N)
//     D        : (H,)
//     dt_bias  : (H,)
//     y_out    : (B, L, H, P)
//     ssm_state: (B, H, P, N) fp32 in-out (chunk-0 initial state in, final out)
//
//  Grid: (B*H, P, 1), threads/tg: SIMD_W (32). Requires Nstate % SIMD_W == 0
//  and Nstate <= SIMD_W * NPL_MAX.

#include <metal_stdlib>
using namespace metal;

constant uint SIMD_W   = 32;     // one simdgroup per (h,p)
constant uint NPL_MAX  = 8;      // max state elements per lane (Nstate <= 256)

[[host_name("mamba2_ssd")]]
[[kernel]]
void mamba2_ssd(
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
    const uint lane = lid;          // 0..SIMD_W
    if (lane >= SIMD_W) return;

    const uint b   = bh / H;
    const uint h   = bh % H;
    const uint hpg = H / G;
    const uint g   = h / hpg;

    // This lane owns state columns {lane, lane+SIMD_W, ...} of row (h,p).
    const uint npl = Nstate / SIMD_W;          // state elements per lane

    const size_t st_row =
        ((size_t)b * H + h) * (size_t)P * (size_t)Nstate
        + (size_t)p * (size_t)Nstate;

    float s[NPL_MAX];
    for (uint j = 0; j < npl; ++j)
        s[j] = ssm_state[st_row + (size_t)(lane + j * SIMD_W)];

    const float A  = -exp((float)A_log[h]);
    const float Db = (float)D_in[h];

    for (uint t = 0; t < L; ++t) {
        const float dtr = (float)dt_raw[((size_t)b * L + t) * H + h]
                        + (float)dt_bias[h];
        const float dt  = clamp((dtr > 20.0f) ? dtr : log(1.0f + exp(dtr)),
                                dt_min, dt_max);
        const float dA  = exp(dt * A);
        const float xv  = (float)x[(((size_t)b * L + t) * H + h) * P + p];
        const float dtx = dt * xv;

        const size_t bc_base = (((size_t)b * L + t) * G + g) * Nstate;

        float yacc = 0.0f;
        for (uint j = 0; j < npl; ++j) {
            const uint n = lane + j * SIMD_W;
            const float Bn = (float)B_in[bc_base + n];
            const float Cn = (float)C_in[bc_base + n];
            const float sj = dA * s[j] + dtx * Bn;
            s[j]  = sj;
            yacc += Cn * sj;
        }
        yacc = simd_sum(yacc);
        if (lane == 0)
            y_out[(((size_t)b * L + t) * H + h) * P + p] = (half)(yacc + Db * xv);
    }

    for (uint j = 0; j < npl; ++j)
        ssm_state[st_row + (size_t)(lane + j * SIMD_W)] = s[j];
}
