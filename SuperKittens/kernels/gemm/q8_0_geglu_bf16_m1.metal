// q8_0_geglu_bf16_m1.metal — Fused gate+up Q8_0 matvec + GeLU·mul for M=1 decode.
//
//   y[N] = gelu_tanh(x[K] @ W_gate^T) * (x[K] @ W_up^T),  W_gate, W_up: Q8_0 [N, K]
//
// bf16-activation / bf16-output variant of q8_0_swiglu_m1 (which is silu, half).
// gelu_tanh matches gemma4_geglu_mul / gemv_geglu_bf16_m1. Replaces the 3-dispatch
// Q8 MLP front (q8_matvec gate, q8_matvec up, gemma4_geglu_mul) with one dispatch:
// x is read once for both rows, no m_up_scratch round-trip.

#include <metal_stdlib>
using namespace metal;

#ifndef QGG_BLOCK
#define QGG_BLOCK 32
#endif

struct __attribute__((packed)) qgg_block { half d; int8_t qs[QGG_BLOCK]; };

constant constexpr short QGG_NW  = 32;
constant constexpr short QGG_NSG = 4;
constant constexpr short QGG_NR0 = 2;
constant constexpr short QGG_NQ  = 8;

[[host_name("q8_0_geglu_bf16_m1")]]
kernel void q8_0_geglu_bf16_m1(
    device const bfloat    *B       [[buffer(0)]],   // x, bf16 [K]
    device const uchar     *Ag_raw  [[buffer(1)]],   // W_gate, Q8_0 [N, K]
    device const uchar     *Au_raw  [[buffer(2)]],   // W_up,   Q8_0 [N, K]
    device       bfloat    *C       [[buffer(3)]],   // y, bf16 [N]
    constant uint           &K      [[buffer(4)]],
    constant uint           &N      [[buffer(5)]],
    uint3   tgpig [[threadgroup_position_in_grid]],
    ushort  tiisg [[thread_index_in_simdgroup]],
    ushort  sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nb   = K / QGG_BLOCK;
    const uint row0 = tgpig.x * QGG_NR0;

    const ushort ix = tiisg / (QGG_NW / QGG_NQ);
    const ushort il = tiisg % (QGG_NW / QGG_NQ);
    const uint   ib0 = (uint)sgitg * QGG_NQ + ix;

    device const bfloat *yb = B + ib0 * QGG_BLOCK + il * QGG_NQ;

    device const qgg_block *axg[QGG_NR0];
    device const qgg_block *axu[QGG_NR0];
    #pragma unroll
    for (short r = 0; r < QGG_NR0; ++r) {
        const uint row = row0 + r;
        axg[r] = (device const qgg_block *)(Ag_raw + (size_t)row * nb * sizeof(qgg_block));
        axu[r] = (device const qgg_block *)(Au_raw + (size_t)row * nb * sizeof(qgg_block));
    }

    float sumg[QGG_NR0] = {0.f};
    float sumu[QGG_NR0] = {0.f};
    float yl[QGG_NQ];

    for (uint ib = ib0; ib < nb; ib += QGG_NSG * QGG_NQ) {
        #pragma unroll
        for (short i = 0; i < QGG_NQ; ++i) yl[i] = (float)yb[i];

        #pragma unroll
        for (short r = 0; r < QGG_NR0; ++r) {
            device const int8_t *qsg = axg[r][ib].qs + il * QGG_NQ;
            device const int8_t *qsu = axu[r][ib].qs + il * QGG_NQ;
            float sg = 0.f, su = 0.f;
            #pragma unroll
            for (short i = 0; i < QGG_NQ; ++i) {
                float xv = yl[i];
                sg += (float)qsg[i] * xv;
                su += (float)qsu[i] * xv;
            }
            sumg[r] += sg * (float)axg[r][ib].d;
            sumu[r] += su * (float)axu[r][ib].d;
        }
        yb += QGG_NSG * QGG_NQ * QGG_BLOCK;
    }

    threadgroup float shg[QGG_NR0 * QGG_NSG];
    threadgroup float shu[QGG_NR0 * QGG_NSG];
    #pragma unroll
    for (short r = 0; r < QGG_NR0; ++r) {
        float vg = simd_sum(sumg[r]);
        float vu = simd_sum(sumu[r]);
        if (tiisg == 0) {
            shg[r * QGG_NSG + sgitg] = vg;
            shu[r * QGG_NSG + sgitg] = vu;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        const float K0 = 0.7978845608028654f;  // sqrt(2/pi)
        const float K1 = 0.044715f;
        #pragma unroll
        for (short r = 0; r < QGG_NR0; ++r) {
            float vg = (tiisg < QGG_NSG) ? shg[r * QGG_NSG + tiisg] : 0.0f;
            float vu = (tiisg < QGG_NSG) ? shu[r * QGG_NSG + tiisg] : 0.0f;
            vg = simd_sum(vg);
            vu = simd_sum(vu);
            if (tiisg == 0) {
                const uint out_row = row0 + r;
                if (out_row < N) {
                    float gv = 0.5f * vg * (1.0f + precise::tanh(K0 * (vg + K1 * vg * vg * vg)));
                    C[out_row] = (bfloat)(gv * vu);
                }
            }
        }
    }
}
