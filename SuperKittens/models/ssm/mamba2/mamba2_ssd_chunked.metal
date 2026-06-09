//
//  mamba2_ssd_chunked.metal
//  Mamba 2 SSD prefill — chunked associative scan (flag-gated alternative to
//  mamba2_ssd; selected by SK_MAMBA2_SSD_CHUNKED=1, prefill T>1 only).
//
//  mamba2_ssd scans L tokens serially per (h,p) row, so prefill parallelism is
//  fixed at B*H*P threadgroups regardless of L. The recurrence
//     s[t] = dA[t]*s[t-1] + dt[t]*B[t]*x[t]      (dA scalar per (t,h))
//  decomposes over chunks of Q tokens:
//     s[t] = cumdecay[t]*S_in[c] + s_local[t],   cumdecay[t] = prod dA over
//                                                 [chunk start .. t] inclusive
//  so chunks scan in parallel from zero state (pass 1), a tiny serial pass
//  propagates inter-chunk states (pass 2), and a parallel pass adds the
//  C·(cumdecay*S_in) correction to y (pass 3). Parallelism grows ×NC with L.
//
//  cumdecay / chunk states are fp32: cumdecay is a product of up to Q values in
//  (0,1] and underflows fp16 (6e-5 floor) within a few dozen tokens for
//  fast-decaying heads; fp32 underflow (1e-38) only occurs where the true
//  correction is negligible anyway.
//
//  Extra buffers (host scratch, fp32):
//     chunk_states : (B, NC, H, P, N)  pass1 writes S_local[c] (final local
//                    state of chunk c); pass2 rewrites slot c to S_in[c] (state
//                    entering chunk c); pass3 reads S_in[c].
//     cumdecay     : (B, L, H)         per-token inclusive prefix decay within
//                    its chunk (written once by the p==0 threadgroup).
//
//  Same per-token math as mamba2_ssd (dt softplus/clamp, dA, D-skip), same
//  x/B/C interleaved-stride layout, one 32-lane simdgroup per threadgroup with
//  N/32 state columns per lane.

#include <metal_stdlib>
using namespace metal;

constant uint SIMD_W  = 32;
constant uint NPL_MAX = 8;       // Nstate <= 256

// Pass 1: per-chunk local scan from zero state.
// Grid: (B*H, P, NC), threads/tg: SIMD_W.
[[host_name("mamba2_ssd_chunk_scan")]]
[[kernel]]
void mamba2_ssd_chunk_scan(
    device const half*  x            [[buffer(0)]],   // (B,L,H,P) stride XBC_stride
    device const half*  dt_raw       [[buffer(1)]],   // (B,L,H)
    device const half*  A_log        [[buffer(2)]],   // (H,)
    device const half*  B_in         [[buffer(3)]],   // (B,L,G,N) stride XBC_stride
    device const half*  C_in         [[buffer(4)]],   // (B,L,G,N) stride XBC_stride
    device const half*  D_in         [[buffer(5)]],   // (H,)
    device const half*  dt_bias      [[buffer(6)]],   // (H,)
    device       half*  y_out        [[buffer(7)]],   // (B,L,H,P) — local part
    device       float* chunk_states [[buffer(8)]],   // (B,NC,H,P,N) — S_local out
    device       float* cumdecay     [[buffer(9)]],   // (B,L,H)
    constant uint&  Batch            [[buffer(10)]],
    constant uint&  L                [[buffer(11)]],
    constant uint&  H                [[buffer(12)]],
    constant uint&  P                [[buffer(13)]],
    constant uint&  G                [[buffer(14)]],
    constant uint&  Nstate           [[buffer(15)]],
    constant float& dt_min           [[buffer(16)]],
    constant float& dt_max           [[buffer(17)]],
    constant uint&  XBC_stride       [[buffer(18)]],
    constant uint&  Q                [[buffer(19)]],  // chunk size (tokens)
    constant uint&  NC               [[buffer(20)]],  // ceil(L/Q)
    uint  lane [[thread_index_in_threadgroup]],
    uint3 gid  [[threadgroup_position_in_grid]])
{
    const uint bh = gid.x;
    const uint p  = gid.y;
    const uint c  = gid.z;
    if (lane >= SIMD_W) return;

    const uint b   = bh / H;
    const uint h   = bh % H;
    const uint g   = h / (H / G);
    const uint npl = Nstate / SIMD_W;

    const uint t0   = c * Q;
    const uint tend = min(t0 + Q, L);

    float s[NPL_MAX];
    for (uint j = 0; j < npl; ++j) s[j] = 0.0f;

    const float A  = -exp((float)A_log[h]);
    const float Db = (float)D_in[h];
    float cum = 1.0f;

    for (uint t = t0; t < tend; ++t) {
        const float dtr = (float)dt_raw[((size_t)b * L + t) * H + h]
                        + (float)dt_bias[h];
        float dt = (dtr > 20.0f) ? dtr : log(1.0f + exp(dtr));
        dt = max(dt, dt_min);
        if (dt_max < 1e30f) dt = min(dt, dt_max);
        const float dA  = exp(dt * A);
        cum *= dA;
        // cumdecay is per (t,h); the p==0 threadgroup writes it for all.
        if (p == 0 && lane == 0)
            cumdecay[((size_t)b * L + t) * H + h] = cum;

        const size_t tok = ((size_t)b * L + t) * (size_t)XBC_stride;
        const float xv  = (float)x[tok + (size_t)h * P + p];
        const float dtx = dt * xv;
        const size_t bc_base = tok + (size_t)g * Nstate;

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

    const size_t cs_row =
        ((((size_t)b * NC + c) * H + h) * P + p) * (size_t)Nstate;
    for (uint j = 0; j < npl; ++j)
        chunk_states[cs_row + (size_t)(lane + j * SIMD_W)] = s[j];
}

// Pass 2: serial inter-chunk state propagation (NC steps).
//   S_in[0] = ssm_state (incoming, e.g. zero after reset);
//   S_in[c] = prodA[c-1]*S_in[c-1] + S_local[c-1]
// Rewrites chunk_states slot c from S_local[c] to S_in[c] in place, and writes
// the final state (= S_in[NC]) back to ssm_state for decode continuation.
// Grid: (B*H, P, 1), threads/tg: SIMD_W.
[[host_name("mamba2_ssd_chunk_prop")]]
[[kernel]]
void mamba2_ssd_chunk_prop(
    device       float* chunk_states [[buffer(0)]],   // (B,NC,H,P,N) in: S_local, out: S_in
    device const float* cumdecay     [[buffer(1)]],   // (B,L,H)
    device       float* ssm_state    [[buffer(2)]],   // (B,H,P,N) in-out
    constant uint& Batch             [[buffer(3)]],
    constant uint& L                 [[buffer(4)]],
    constant uint& H                 [[buffer(5)]],
    constant uint& P                 [[buffer(6)]],
    constant uint& Nstate            [[buffer(7)]],
    constant uint& Q                 [[buffer(8)]],
    constant uint& NC                [[buffer(9)]],
    uint  lane [[thread_index_in_threadgroup]],
    uint3 gid  [[threadgroup_position_in_grid]])
{
    const uint bh = gid.x;
    const uint p  = gid.y;
    if (lane >= SIMD_W) return;

    const uint b   = bh / H;
    const uint h   = bh % H;
    const uint npl = Nstate / SIMD_W;

    const size_t st_row =
        ((size_t)b * H + h) * (size_t)P * (size_t)Nstate
        + (size_t)p * (size_t)Nstate;

    float s_in[NPL_MAX];
    for (uint j = 0; j < npl; ++j)
        s_in[j] = ssm_state[st_row + (size_t)(lane + j * SIMD_W)];

    for (uint c = 0; c < NC; ++c) {
        const uint last_t = min((c + 1) * Q, L) - 1;
        // prodA[c] = inclusive prefix decay at the chunk's last token.
        const float prod = cumdecay[((size_t)b * L + last_t) * H + h];
        const size_t cs_row =
            ((((size_t)b * NC + c) * H + h) * P + p) * (size_t)Nstate;
        for (uint j = 0; j < npl; ++j) {
            const size_t idx = cs_row + (size_t)(lane + j * SIMD_W);
            const float s_local = chunk_states[idx];
            chunk_states[idx] = s_in[j];
            s_in[j] = prod * s_in[j] + s_local;
        }
    }

    for (uint j = 0; j < npl; ++j)
        ssm_state[st_row + (size_t)(lane + j * SIMD_W)] = s_in[j];
}

// PB-blocked pass 1: one simdgroup scans PB p-rows of the same (h, chunk), so
// each B/C token read (and the dt softplus/exp) is shared across PB rows.
// The serial kernel (and the PB=1 pass above) re-reads B/C once per p-row —
// ~P× redundant traffic with effectively no L2 reuse on M4 — which is what
// makes the SSD stage bandwidth-bound. Requires P % PB == 0 and
// PB * (Nstate/32) <= PB * NPL_PB register columns.
// Grid: (B*H, P/PB, NC), threads/tg: SIMD_W.
constant uint PB     = 4;
constant uint NPL_PB = 4;        // Nstate <= 128 in the PB variants

[[host_name("mamba2_ssd_chunk_scan_pb4")]]
[[kernel]]
void mamba2_ssd_chunk_scan_pb4(
    device const half*  x            [[buffer(0)]],
    device const half*  dt_raw       [[buffer(1)]],
    device const half*  A_log        [[buffer(2)]],
    device const half*  B_in         [[buffer(3)]],
    device const half*  C_in         [[buffer(4)]],
    device const half*  D_in         [[buffer(5)]],
    device const half*  dt_bias      [[buffer(6)]],
    device       half*  y_out        [[buffer(7)]],
    device       float* chunk_states [[buffer(8)]],
    device       float* cumdecay     [[buffer(9)]],
    constant uint&  Batch            [[buffer(10)]],
    constant uint&  L                [[buffer(11)]],
    constant uint&  H                [[buffer(12)]],
    constant uint&  P                [[buffer(13)]],
    constant uint&  G                [[buffer(14)]],
    constant uint&  Nstate           [[buffer(15)]],
    constant float& dt_min           [[buffer(16)]],
    constant float& dt_max           [[buffer(17)]],
    constant uint&  XBC_stride       [[buffer(18)]],
    constant uint&  Q                [[buffer(19)]],
    constant uint&  NC               [[buffer(20)]],
    uint  lane [[thread_index_in_threadgroup]],
    uint3 gid  [[threadgroup_position_in_grid]])
{
    const uint bh = gid.x;
    const uint p0 = gid.y * PB;
    const uint c  = gid.z;
    if (lane >= SIMD_W) return;

    const uint b   = bh / H;
    const uint h   = bh % H;
    const uint g   = h / (H / G);
    const uint npl = Nstate / SIMD_W;

    const uint t0   = c * Q;
    const uint tend = min(t0 + Q, L);

    float s[PB][NPL_PB];
    for (uint q = 0; q < PB; ++q)
        for (uint j = 0; j < npl; ++j) s[q][j] = 0.0f;

    const float A  = -exp((float)A_log[h]);
    const float Db = (float)D_in[h];
    float cum = 1.0f;

    for (uint t = t0; t < tend; ++t) {
        const float dtr = (float)dt_raw[((size_t)b * L + t) * H + h]
                        + (float)dt_bias[h];
        float dt = (dtr > 20.0f) ? dtr : log(1.0f + exp(dtr));
        dt = max(dt, dt_min);
        if (dt_max < 1e30f) dt = min(dt, dt_max);
        const float dA  = exp(dt * A);
        cum *= dA;
        if (gid.y == 0 && lane == 0)
            cumdecay[((size_t)b * L + t) * H + h] = cum;

        const size_t tok = ((size_t)b * L + t) * (size_t)XBC_stride;
        float dtx[PB];
        float xv[PB];
        for (uint q = 0; q < PB; ++q) {
            xv[q]  = (float)x[tok + (size_t)h * P + p0 + q];
            dtx[q] = dt * xv[q];
        }
        const size_t bc_base = tok + (size_t)g * Nstate;

        float yacc[PB] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint j = 0; j < npl; ++j) {
            const uint n = lane + j * SIMD_W;
            const float Bn = (float)B_in[bc_base + n];
            const float Cn = (float)C_in[bc_base + n];
            for (uint q = 0; q < PB; ++q) {
                const float sj = dA * s[q][j] + dtx[q] * Bn;
                s[q][j]  = sj;
                yacc[q] += Cn * sj;
            }
        }
        for (uint q = 0; q < PB; ++q) {
            const float yq = simd_sum(yacc[q]);
            if (lane == 0)
                y_out[(((size_t)b * L + t) * H + h) * P + p0 + q] =
                    (half)(yq + Db * xv[q]);
        }
    }

    for (uint q = 0; q < PB; ++q) {
        const size_t cs_row =
            ((((size_t)b * NC + c) * H + h) * P + p0 + q) * (size_t)Nstate;
        for (uint j = 0; j < npl; ++j)
            chunk_states[cs_row + (size_t)(lane + j * SIMD_W)] = s[q][j];
    }
}

// PB-blocked pass 3 (shared C reads across PB p-rows).
// Grid: (B*H, P/PB, NC), threads/tg: SIMD_W.
[[host_name("mamba2_ssd_chunk_fix_pb4")]]
[[kernel]]
void mamba2_ssd_chunk_fix_pb4(
    device const half*  C_in         [[buffer(0)]],
    device const float* cumdecay     [[buffer(1)]],
    device const float* chunk_states [[buffer(2)]],
    device       half*  y_out        [[buffer(3)]],
    constant uint& Batch             [[buffer(4)]],
    constant uint& L                 [[buffer(5)]],
    constant uint& H                 [[buffer(6)]],
    constant uint& P                 [[buffer(7)]],
    constant uint& G                 [[buffer(8)]],
    constant uint& Nstate            [[buffer(9)]],
    constant uint& XBC_stride        [[buffer(10)]],
    constant uint& Q                 [[buffer(11)]],
    constant uint& NC                [[buffer(12)]],
    uint  lane [[thread_index_in_threadgroup]],
    uint3 gid  [[threadgroup_position_in_grid]])
{
    const uint bh = gid.x;
    const uint p0 = gid.y * PB;
    const uint c  = gid.z;
    if (lane >= SIMD_W) return;

    const uint b   = bh / H;
    const uint h   = bh % H;
    const uint g   = h / (H / G);
    const uint npl = Nstate / SIMD_W;

    float s_in[PB][NPL_PB];
    float amax = 0.0f;
    for (uint q = 0; q < PB; ++q) {
        const size_t cs_row =
            ((((size_t)b * NC + c) * H + h) * P + p0 + q) * (size_t)Nstate;
        for (uint j = 0; j < npl; ++j) {
            s_in[q][j] = chunk_states[cs_row + (size_t)(lane + j * SIMD_W)];
            amax = max(amax, fabs(s_in[q][j]));
        }
    }
    if (simd_max(amax) == 0.0f) return;

    const uint t0   = c * Q;
    const uint tend = min(t0 + Q, L);

    for (uint t = t0; t < tend; ++t) {
        const size_t bc_base =
            ((size_t)b * L + t) * (size_t)XBC_stride + (size_t)g * Nstate;
        float yacc[PB] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint j = 0; j < npl; ++j) {
            const float Cn = (float)C_in[bc_base + (size_t)(lane + j * SIMD_W)];
            for (uint q = 0; q < PB; ++q)
                yacc[q] += Cn * s_in[q][j];
        }
        for (uint q = 0; q < PB; ++q) {
            const float yq = simd_sum(yacc[q]);
            if (lane == 0) {
                const float cum = cumdecay[((size_t)b * L + t) * H + h];
                const size_t yi = (((size_t)b * L + t) * H + h) * P + p0 + q;
                y_out[yi] = (half)((float)y_out[yi] + cum * yq);
            }
        }
    }
}

// Pass 3: y[t] += cumdecay[t] * (C[t] · S_in[chunk(t)]).
// Grid: (B*H, P, NC), threads/tg: SIMD_W.
[[host_name("mamba2_ssd_chunk_fix")]]
[[kernel]]
void mamba2_ssd_chunk_fix(
    device const half*  C_in         [[buffer(0)]],   // (B,L,G,N) stride XBC_stride
    device const float* cumdecay     [[buffer(1)]],   // (B,L,H)
    device const float* chunk_states [[buffer(2)]],   // (B,NC,H,P,N) = S_in
    device       half*  y_out        [[buffer(3)]],   // (B,L,H,P) in-out
    constant uint& Batch             [[buffer(4)]],
    constant uint& L                 [[buffer(5)]],
    constant uint& H                 [[buffer(6)]],
    constant uint& P                 [[buffer(7)]],
    constant uint& G                 [[buffer(8)]],
    constant uint& Nstate            [[buffer(9)]],
    constant uint& XBC_stride        [[buffer(10)]],
    constant uint& Q                 [[buffer(11)]],
    constant uint& NC                [[buffer(12)]],
    uint  lane [[thread_index_in_threadgroup]],
    uint3 gid  [[threadgroup_position_in_grid]])
{
    const uint bh = gid.x;
    const uint p  = gid.y;
    const uint c  = gid.z;
    if (lane >= SIMD_W) return;

    const uint b   = bh / H;
    const uint h   = bh % H;
    const uint g   = h / (H / G);
    const uint npl = Nstate / SIMD_W;

    const size_t cs_row =
        ((((size_t)b * NC + c) * H + h) * P + p) * (size_t)Nstate;
    float s_in[NPL_MAX];
    float amax = 0.0f;
    for (uint j = 0; j < npl; ++j) {
        s_in[j] = chunk_states[cs_row + (size_t)(lane + j * SIMD_W)];
        amax = max(amax, fabs(s_in[j]));
    }
    // Zero incoming state (always chunk 0 after reset): correction is exactly 0.
    if (simd_max(amax) == 0.0f) return;

    const uint t0   = c * Q;
    const uint tend = min(t0 + Q, L);

    for (uint t = t0; t < tend; ++t) {
        const size_t bc_base =
            ((size_t)b * L + t) * (size_t)XBC_stride + (size_t)g * Nstate;
        float yacc = 0.0f;
        for (uint j = 0; j < npl; ++j)
            yacc += (float)C_in[bc_base + (size_t)(lane + j * SIMD_W)] * s_in[j];
        yacc = simd_sum(yacc);
        if (lane == 0) {
            const float cum = cumdecay[((size_t)b * L + t) * H + h];
            const size_t yi = (((size_t)b * L + t) * H + h) * P + p;
            y_out[yi] = (half)((float)y_out[yi] + cum * yacc);
        }
    }
}
