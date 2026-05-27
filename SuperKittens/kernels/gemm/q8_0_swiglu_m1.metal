// q8_0_swiglu_m1.metal — Fused gate+up Q8_0 matvec + SiLU·mul for M=1 decode.
//
//   y[N] = silu(x[K] @ W_gate^T) * (x[K] @ W_up^T),  W_gate, W_up: Q8_0 [N, K]
//
// Replaces 3 dispatches (q8_0_matvec gate, q8_0_matvec up, silu_mul) per MLP.
// Structure mirrors q8_0_matvec: TG writes Q8_NR0=2 output rows; 4 simdgroups
// × 32 lanes share K. Each TG step computes both gate-row and up-row partials,
// then reduces and applies SwiGLU at the tail.

#include <metal_stdlib>
using namespace metal;

#ifndef Q8_BLOCK
#define Q8_BLOCK 32
#endif

struct __attribute__((packed)) q8_block_sw { half d; int8_t qs[Q8_BLOCK]; };

constant constexpr short SW_NW  = 32;
constant constexpr short SW_NSG = 4;
constant constexpr short SW_NR0 = 2;
constant constexpr short SW_NQ  = 8;

[[host_name("q8_0_swiglu_m1")]]
kernel void q8_0_swiglu_m1(
    device const half      *B       [[buffer(0)]],   // x, fp16 [K]
    device const uchar     *Ag_raw  [[buffer(1)]],   // W_gate, Q8_0 [N, K]
    device const uchar     *Au_raw  [[buffer(2)]],   // W_up,   Q8_0 [N, K]
    device       half      *C       [[buffer(3)]],   // y, fp16 [N]
    constant uint           &K      [[buffer(4)]],
    constant uint           &N      [[buffer(5)]],
    uint3   tgpig [[threadgroup_position_in_grid]],
    ushort  tiisg [[thread_index_in_simdgroup]],
    ushort  sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nb   = K / Q8_BLOCK;
    const uint row0 = tgpig.x * SW_NR0;

    const ushort ix = tiisg / (SW_NW / SW_NQ);
    const ushort il = tiisg % (SW_NW / SW_NQ);
    const uint   ib0 = (uint)sgitg * SW_NQ + ix;

    device const half *yb = B + ib0 * Q8_BLOCK + il * SW_NQ;

    device const q8_block_sw *axg[SW_NR0];
    device const q8_block_sw *axu[SW_NR0];
    #pragma unroll
    for (short r = 0; r < SW_NR0; ++r) {
        const uint row = row0 + r;
        axg[r] = (device const q8_block_sw *)(Ag_raw + (size_t)row * nb * sizeof(q8_block_sw));
        axu[r] = (device const q8_block_sw *)(Au_raw + (size_t)row * nb * sizeof(q8_block_sw));
    }

    float sumg[SW_NR0] = {0.f};
    float sumu[SW_NR0] = {0.f};
    half  yl[SW_NQ];

    for (uint ib = ib0; ib < nb; ib += SW_NSG * SW_NQ) {
        #pragma unroll
        for (short i = 0; i < SW_NQ; ++i) yl[i] = yb[i];

        #pragma unroll
        for (short r = 0; r < SW_NR0; ++r) {
            device const int8_t *qsg = axg[r][ib].qs + il * SW_NQ;
            device const int8_t *qsu = axu[r][ib].qs + il * SW_NQ;
            float sg = 0.f, su = 0.f;
            #pragma unroll
            for (short i = 0; i < SW_NQ; ++i) {
                float xv = (float)yl[i];
                sg += (float)qsg[i] * xv;
                su += (float)qsu[i] * xv;
            }
            sumg[r] += sg * (float)axg[r][ib].d;
            sumu[r] += su * (float)axu[r][ib].d;
        }
        yb += SW_NSG * SW_NQ * Q8_BLOCK;
    }

    threadgroup float shg[SW_NR0 * SW_NSG];
    threadgroup float shu[SW_NR0 * SW_NSG];
    #pragma unroll
    for (short r = 0; r < SW_NR0; ++r) {
        float vg = simd_sum(sumg[r]);
        float vu = simd_sum(sumu[r]);
        if (tiisg == 0) {
            shg[r * SW_NSG + sgitg] = vg;
            shu[r * SW_NSG + sgitg] = vu;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        #pragma unroll
        for (short r = 0; r < SW_NR0; ++r) {
            float vg = (tiisg < SW_NSG) ? shg[r * SW_NSG + tiisg] : 0.0f;
            float vu = (tiisg < SW_NSG) ? shu[r * SW_NSG + tiisg] : 0.0f;
            vg = simd_sum(vg);
            vu = simd_sum(vu);
            if (tiisg == 0) {
                const uint out_row = row0 + r;
                if (out_row < N) {
                    float sig = 1.0f / (1.0f + exp(-vg));
                    C[out_row] = (half)(vg * sig * vu);
                }
            }
        }
    }
}
