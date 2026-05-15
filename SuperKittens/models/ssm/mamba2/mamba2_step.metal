//
//  mamba2_step.metal — single-token Mamba-2 decode kernel.
//
//  Math (per BH):
//    h_state = exp(A_log_t) * h_state + outer(B_t, x_t)   // (Ds, Dv)
//    y_t     = h_state^T @ C_t                            // (Dv,)
//
//  One TG of 64 threads per BH. h_state lives in TG memory.
//
#include <metal_stdlib>
using namespace metal;

[[host_name("mamba2_step")]]
[[kernel]]
void mamba2_step(
    device const half*  x_t,        // (BH, Dv)
    device const half*  B_t,        // (BH, Ds)
    device const half*  C_t,        // (BH, Ds)
    device const half*  A_log_t,    // (BH,)
    device float*       h_state,    // (BH, Ds, Dv) — read+write
    device half*        y_t,        // (BH, Dv)
    constant uint& Ds, constant uint& Dv,
    uint  lid [[thread_index_in_threadgroup]],
    uint  bh  [[threadgroup_position_in_grid]])
{
    const uint T = 64;
    threadgroup float h[64*64];
    threadgroup half  Bs[64], Cs[64], xs[64];
    threadgroup float a_decay_tg[1];

    const size_t state_bo = (size_t)bh * Ds * Dv;
    const size_t s_bo     = (size_t)bh * Ds;
    const size_t v_bo     = (size_t)bh * Dv;

    // Load h_state
    for (uint i = lid; i < Ds * Dv; i += T) h[i] = h_state[state_bo + i];
    if (lid < Ds) Bs[lid] = B_t[s_bo + lid];
    if (lid < Ds) Cs[lid] = C_t[s_bo + lid];
    if (lid < Dv) xs[lid] = x_t[v_bo + lid];
    if (lid == 0) a_decay_tg[0] = metal::fast::exp(float(A_log_t[bh]));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float decay = a_decay_tg[0];

    // h = decay*h + outer(B, x).  Iterate column j over Dv.
    for (uint j = lid; j < Dv; j += T) {
        float xj = float(xs[j]);
        for (uint i = 0; i < Ds; i += 4) {
            half4 b4 = reinterpret_cast<threadgroup half4*>(Bs)[i/4];
            uint base = i * Dv + j;
            h[base + 0*Dv] = h[base + 0*Dv] * decay + float(b4.x) * xj;
            h[base + 1*Dv] = h[base + 1*Dv] * decay + float(b4.y) * xj;
            h[base + 2*Dv] = h[base + 2*Dv] * decay + float(b4.z) * xj;
            h[base + 3*Dv] = h[base + 3*Dv] * decay + float(b4.w) * xj;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // y = h^T @ C  → y[j] = sum_i C[i] * h[i,j]
    for (uint j = lid; j < Dv; j += T) {
        float acc = 0;
        for (uint i = 0; i < Ds; i += 4) {
            half4 c4 = reinterpret_cast<threadgroup half4*>(Cs)[i/4];
            acc += float(c4.x) * h[(i+0)*Dv + j];
            acc += float(c4.y) * h[(i+1)*Dv + j];
            acc += float(c4.z) * h[(i+2)*Dv + j];
            acc += float(c4.w) * h[(i+3)*Dv + j];
        }
        y_t[v_bo + j] = half(acc);
    }

    // Write back updated h_state
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lid; i < Ds * Dv; i += T) h_state[state_bo + i] = h[i];
}
