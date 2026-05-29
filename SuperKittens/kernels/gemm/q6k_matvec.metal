// q6k_matvec.metal — Q6_K weight × fp16 activation matvec (decode-time, M=1).
//
// Unblocks native Q4_K_M: those GGUFs store v_proj / down_proj / token_embd as
// Q6_K, so without a Q6_K matvec the whole tensor falls back to host-dequant.
//
// Block layout (GGML block_q6_K, 210 B / 256 weights), attribution: derived from
// llama.cpp dequantize_row_q6_K and cross-checked against gguf.quants.Q6_K
// (ds4 registers q6_K=210B in quants.c but ships no Q6_K kernel/quantizer):
//     uint8 ql[128]    // low 4 bits
//     uint8 qh[64]     // high 2 bits
//     int8  scales[16] // per-16-element sub-scale
//     half  d          // super-block scale
// element q = ((ql_nibble) | (qh_2bit << 4)) - 32 ;  w = d * scales[is] * q
//
// Per 128-element half h in {0,1} and l in [0,32): the four sub-positions and
// the ql/qh/scale taps that feed them are fixed by the GGML packing —
//   y[l +  0] : ql[l]    & 0x0F , (qh[l] >> 0) & 3 , scales[8h + 0 + l/16]
//   y[l + 32] : ql[l+32] & 0x0F , (qh[l] >> 2) & 3 , scales[8h + 2 + l/16]
//   y[l + 64] : ql[l]    >>   4 , (qh[l] >> 4) & 3 , scales[8h + 4 + l/16]
//   y[l + 96] : ql[l+32] >>   4 , (qh[l] >> 6) & 3 , scales[8h + 6 + l/16]
// with ql += 64, qh += 32 between the two halves. d is folded once per block.
//
// Dispatch (host): grid (ceil(N/NR0), 1, 1), threadgroup (NW * NSG, 1, 1).
//   NW=32 lanes, NSG=4 simdgroups cooperate over the K (super-block) dimension,
//   NR0=2 output rows per TG. The 32 lanes of a simdgroup tile the 32 l-values
//   of a super-block once; every lane accumulates all four sub-positions across
//   both halves, so one simdgroup covers all 256 weights of a block per step.
//
// Bindings (mirror q8_0_matvec):
//   0: B activation, fp16 [K]
//   1: A weights, q6_K row-major [N, K]
//   2: C output, fp16 [N]
//   3: uint K
//   4: uint N

#include <metal_stdlib>
using namespace metal;

#ifndef QK_K
#define QK_K 256
#endif

struct __attribute__((packed)) block_q6_K {
    uint8_t ql[QK_K / 2];      // 128
    uint8_t qh[QK_K / 4];      // 64
    int8_t  scales[QK_K / 16]; // 16
    half    d;
};

constant constexpr short Q6_NW  = 32;
constant constexpr short Q6_NSG = 4;
constant constexpr short Q6_NR0 = 2;

kernel void q6k_matvec(
    device const half       *B      [[buffer(0)]],
    device const uchar      *A_raw  [[buffer(1)]],
    device       half       *C      [[buffer(2)]],
    constant uint            &K     [[buffer(3)]],
    constant uint            &N     [[buffer(4)]],
    uint3   tgpig [[threadgroup_position_in_grid]],
    ushort  tiisg [[thread_index_in_simdgroup]],
    ushort  sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nb   = K / QK_K;            // super-blocks per row
    const uint row0 = tgpig.x * Q6_NR0;

    // Each lane owns one l in [0,32); the 32 lanes tile a 128-element half once.
    const short l  = tiisg;
    const short is = l / 16;               // sub-scale offset within a half

    device const block_q6_K *ax[Q6_NR0];
    #pragma unroll
    for (short r = 0; r < Q6_NR0; ++r) {
        const uint row = row0 + r;
        ax[r] = (device const block_q6_K *)(A_raw + (size_t)row * nb * sizeof(block_q6_K));
    }

    float sumf[Q6_NR0] = {0.f};

    for (uint ib = sgitg; ib < nb; ib += Q6_NSG) {
        device const half *yb = B + ib * QK_K;

        const float y0 = (float)yb[l +   0];
        const float y1 = (float)yb[l +  32];
        const float y2 = (float)yb[l +  64];
        const float y3 = (float)yb[l +  96];
        const float y4 = (float)yb[l + 128];
        const float y5 = (float)yb[l + 160];
        const float y6 = (float)yb[l + 192];
        const float y7 = (float)yb[l + 224];

        #pragma unroll
        for (short r = 0; r < Q6_NR0; ++r) {
            device const uint8_t *ql = ax[r][ib].ql;
            device const uint8_t *qh = ax[r][ib].qh;
            device const int8_t  *sc = ax[r][ib].scales;

            float bsum = 0.f;
            {   // half 0
                const uint8_t lo  = ql[l];
                const uint8_t lo2 = ql[l + 32];
                const uint8_t hi  = qh[l];
                const int q0 = (int)((lo  & 0x0F) | (((hi >> 0) & 3) << 4)) - 32;
                const int q1 = (int)((lo2 & 0x0F) | (((hi >> 2) & 3) << 4)) - 32;
                const int q2 = (int)((lo  >>   4) | (((hi >> 4) & 3) << 4)) - 32;
                const int q3 = (int)((lo2 >>   4) | (((hi >> 6) & 3) << 4)) - 32;
                bsum += y0 * (float)sc[0 + is] * (float)q0;
                bsum += y1 * (float)sc[2 + is] * (float)q1;
                bsum += y2 * (float)sc[4 + is] * (float)q2;
                bsum += y3 * (float)sc[6 + is] * (float)q3;
            }
            {   // half 1
                const uint8_t lo  = ql[l + 64];
                const uint8_t lo2 = ql[l + 96];
                const uint8_t hi  = qh[l + 32];
                const int q0 = (int)((lo  & 0x0F) | (((hi >> 0) & 3) << 4)) - 32;
                const int q1 = (int)((lo2 & 0x0F) | (((hi >> 2) & 3) << 4)) - 32;
                const int q2 = (int)((lo  >>   4) | (((hi >> 4) & 3) << 4)) - 32;
                const int q3 = (int)((lo2 >>   4) | (((hi >> 6) & 3) << 4)) - 32;
                bsum += y4 * (float)sc[8  + is] * (float)q0;
                bsum += y5 * (float)sc[10 + is] * (float)q1;
                bsum += y6 * (float)sc[12 + is] * (float)q2;
                bsum += y7 * (float)sc[14 + is] * (float)q3;
            }
            sumf[r] += bsum * (float)ax[r][ib].d;
        }
    }

    threadgroup float shmem[Q6_NR0 * Q6_NSG];
    #pragma unroll
    for (short r = 0; r < Q6_NR0; ++r) {
        float v = simd_sum(sumf[r]);
        if (tiisg == 0) shmem[r * Q6_NSG + sgitg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        #pragma unroll
        for (short r = 0; r < Q6_NR0; ++r) {
            float v = (tiisg < Q6_NSG) ? shmem[r * Q6_NSG + tiisg] : 0.0f;
            v = simd_sum(v);
            if (tiisg == 0) {
                const uint out_row = row0 + r;
                if (out_row < N) C[out_row] = (half)v;
            }
        }
    }
}
