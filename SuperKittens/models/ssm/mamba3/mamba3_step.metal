//
//  mamba3_step.metal — single-token Mamba-3 decode.
//
//  Per BH math:
//    a_cs    += a_t                                   // running cumsum
//    b_scale  = 1 + b_t * exp(-a_cs)
//    theta    = a_cs * angle_t * PI                   // (DQ/2,)
//    q_rot,k_rot = rotate(q_t, k_t, theta)
//    decay    = exp(a_t) * b_scale                    // single-step decay (NOT cumulative)
//    h_state  = decay * h_state + b_scale * outer(k_rot, v_t)
//    y_t      = q_rot^T @ h_state                     // (DV,)
//
#include <metal_stdlib>
using namespace metal;

[[host_name("mamba3_step")]]
[[kernel]]
void mamba3_step(
    device const half*  q_t,      // (BH, DQ)
    device const half*  k_t,      // (BH, DQ)
    device const half*  v_t,      // (BH, DV)
    device const half*  a_t,      // (BH,)
    device const half*  b_t,      // (BH,)
    device const half*  angle_t,  // (BH, DQ/2)
    device float*       h_state,  // (BH, DQ, DV)
    device float*       a_cs,     // (BH,)
    device half*        y_t,      // (BH, DV)
    constant uint& DQ, constant uint& DV,
    uint  lid [[thread_index_in_threadgroup]],
    uint  bh  [[threadgroup_position_in_grid]])
{
    const uint T = 64;
    const float PI = 3.141592653589793f;
    const uint HD = DQ / 2;

    threadgroup float h[64*64];
    threadgroup half  Q[64], K[64], V[64];
    threadgroup float scratch[4];   // [0]=a_cs_new, [1]=b_scale, [2]=decay, [3]=unused
    threadgroup half  Qr[64], Kr[64];

    const size_t state_bo = (size_t)bh * DQ * DV;
    const size_t q_bo     = (size_t)bh * DQ;
    const size_t v_bo     = (size_t)bh * DV;
    const size_t ang_bo   = (size_t)bh * HD;

    // Load h_state, q, k, v
    for (uint i = lid; i < DQ * DV; i += T) h[i] = h_state[state_bo + i];
    if (lid < DQ) { Q[lid] = q_t[q_bo + lid]; K[lid] = k_t[q_bo + lid]; }
    if (lid < DV)   V[lid] = v_t[v_bo + lid];

    if (lid == 0) {
        float a = float(a_t[bh]);
        float b = float(b_t[bh]);
        float cs_new = a_cs[bh] + a;
        float bs = 1.0f + b * metal::fast::exp(-cs_new);
        scratch[0] = cs_new;
        scratch[1] = bs;
        scratch[2] = metal::fast::exp(a) * bs;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float a_cs_new = scratch[0];
    float b_scale  = scratch[1];
    float decay    = scratch[2];

    // Rotary: rotate q and k by theta = a_cs_new * angle * PI
    for (uint p = lid; p < HD; p += T) {
        float th = a_cs_new * float(angle_t[ang_bo + p]) * PI;
        float c = metal::fast::cos(th), s = metal::fast::sin(th);
        float q0 = float(Q[p]), q1 = float(Q[p + HD]);
        float k0 = float(K[p]), k1 = float(K[p + HD]);
        Qr[p]      = half(q0*c - q1*s);
        Qr[p + HD] = half(q0*s + q1*c);
        Kr[p]      = half(k0*c - k1*s);
        Kr[p + HD] = half(k0*s + k1*c);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // h = decay*h + b_scale * outer(k_rot, v_t)
    for (uint j = lid; j < DV; j += T) {
        float vj = float(V[j]);
        for (uint i = 0; i < DQ; i += 4) {
            half4 k4 = reinterpret_cast<threadgroup half4*>(Kr)[i/4];
            uint base = i * DV + j;
            h[base + 0*DV] = h[base + 0*DV] * decay + b_scale * float(k4.x) * vj;
            h[base + 1*DV] = h[base + 1*DV] * decay + b_scale * float(k4.y) * vj;
            h[base + 2*DV] = h[base + 2*DV] * decay + b_scale * float(k4.z) * vj;
            h[base + 3*DV] = h[base + 3*DV] * decay + b_scale * float(k4.w) * vj;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // y = q_rot @ h   →  y[j] = sum_i Qr[i] * h[i,j]
    for (uint j = lid; j < DV; j += T) {
        float acc = 0;
        for (uint i = 0; i < DQ; i += 4) {
            half4 q4 = reinterpret_cast<threadgroup half4*>(Qr)[i/4];
            acc += float(q4.x) * h[(i+0)*DV + j];
            acc += float(q4.y) * h[(i+1)*DV + j];
            acc += float(q4.z) * h[(i+2)*DV + j];
            acc += float(q4.w) * h[(i+3)*DV + j];
        }
        y_t[v_bo + j] = half(acc);
    }

    // Write state + a_cs
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lid; i < DQ * DV; i += T) h_state[state_bo + i] = h[i];
    if (lid == 0) a_cs[bh] = a_cs_new;
}
