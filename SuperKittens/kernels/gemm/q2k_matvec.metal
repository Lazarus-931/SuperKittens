// q2k_matvec.metal — Q2_K weight × fp16 activation matvec (decode-time, M=1).
//
// Q2_K_M GGUFs store most projections as Q2_K (~2.625 bpw) — roughly half the
// bytes of Q4_K — so decode (bandwidth-bound) wins from routing them through a
// native Q2_K dot instead of host-dequant. Mirrors q4k_matvec / q6k_matvec
// binding + dispatch (B=0,A=1,C=2,K=3,N=4 ; NR0=2 rows/TG, 128 threads) so the
// qwen launcher picks the PSO by dtype with one uniform call shape.
//
// block_q2_K (84 B / 256 weights), per llama.cpp dequantize_row_q2_K:
//     uint8 scales[16]  // per-16 sub-block: low nibble = 4-bit scale, high = 4-bit min
//     uint8 qs[64]      // 256 2-bit quants
//     half  d           // super-block scale
//     half  dmin        // super-block min
//   w = d * sc * q - dmin * (m/16)   (sc, m from the matching scales[] nibble)
//
// The per-block dequant/dot is the validated ds4 mul_mv_q2_K layout (also used
// by kernels/moe/down_scatter_q2k.metal): a 32-lane simdgroup tiles one
// super-block's 256 weights per step. Lane split: ix=lane/8 selects the
// stride-4 block step, it=lane%8 picks the 8-elem tap within the half.
//
// gemm_q2k_mv below is a SEPARATE self-contained matvec (struct args, single
// SG/row, buffers 0..3) kept for the quant_mv.c++ lab launcher — do not merge.
//
// Dispatch (host): grid (ceil(N/NR0), 1, 1), threadgroup (NW * NSG, 1, 1).
//   NW=32, NSG=4 simdgroups, NR0=2 rows/TG. Two simdgroups cooperate per output
//   row (rows {0,1} ← SGs {0,2},{1,3}); each pair tiles the K dimension.
//
// Bindings (mirror q8_0_matvec):
//   0: B activation, fp16 [K]
//   1: A weights, q2_K row-major [N, K]
//   2: C output, fp16 [N]
//   3: uint K
//   4: uint N

#include <metal_stdlib>
using namespace metal;

#ifndef QK_K
#define QK_K 256
#endif

struct block_q2_K {
    uchar scales[QK_K / 16];  // 16
    uchar qs[QK_K / 4];       // 64
    half  d;
    half  dmin;
};

constant constexpr short Q2_NW  = 32;
constant constexpr short Q2_NSG = 4;
constant constexpr short Q2_NR0 = 2;

kernel void q2k_matvec(
    device const half       *B      [[buffer(0)]],
    device const uchar      *A_raw  [[buffer(1)]],
    device       half       *C      [[buffer(2)]],
    constant uint            &K     [[buffer(3)]],
    constant uint            &N     [[buffer(4)]],
    uint3   tgpig [[threadgroup_position_in_grid]],
    ushort  tiisg [[thread_index_in_simdgroup]],
    ushort  sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nb   = K / QK_K;             // super-blocks per row
    const uint row0 = tgpig.x * Q2_NR0;

    // SGs {0,1} → row offset 0, SGs {2,3} → row offset 1. The two SGs sharing a
    // row tile the K dimension at stride 2 (sg_in_row ∈ {0,1}).
    const short row_off   = sgitg / 2;      // 0..1
    const short sg_in_row = sgitg % 2;      // 0..1
    const uint  row = row0 + row_off;
    if (row >= N) return;

    // Classic q2_K tap layout within the 32-lane simdgroup.
    const short ix = tiisg / 8;             // 0..3 — stride-4 block step
    const short it = tiisg % 8;             // 0..7
    const short iq = it / 4;                // 0..1 — low/high 128
    const short ir = it % 4;                // 0..3
    const short is = (8 * ir) / 16;         // 0..1 — sub-scale offset within half

    device const block_q2_K *xrow =
        (device const block_q2_K *)(A_raw + (size_t)row * nb * sizeof(block_q2_K));

    // Each SG-in-row covers super-blocks [ix + 4*sg_in_row], stride 8.
    const uint ib_start  = (uint)ix + 4u * (uint)sg_in_row;
    const uint ib_stride = 4u * 2u;

    device const half *y4 = B + ib_start * QK_K + 128 * iq + 8 * ir;

    float yl[32];
    float sumf = 0.f;

    for (uint ib = ib_start; ib < nb; ib += ib_stride) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i +  0] = (float)y4[i +  0]; sumy[0] += yl[i +  0];
            yl[i +  8] = (float)y4[i + 32]; sumy[1] += yl[i +  8];
            yl[i + 16] = (float)y4[i + 64]; sumy[2] += yl[i + 16];
            yl[i + 24] = (float)y4[i + 96]; sumy[3] += yl[i + 24];
        }

        device const uint8_t  *sc = (device const uint8_t  *)xrow[ib].scales + 8 * iq + is;
        device const uint16_t *qs = (device const uint16_t *)xrow[ib].qs     + 16 * iq + 4 * ir;
        device const half     *dh = &xrow[ib].d;

        float4 acc1 = {0.f, 0.f, 0.f, 0.f};
        float4 acc2 = {0.f, 0.f, 0.f, 0.f};
        for (int i = 0; i < 8; i += 2) {
            acc1[0] += yl[i +  0] * (qs[i/2] & 0x0003);
            acc2[0] += yl[i +  1] * (qs[i/2] & 0x0300);
            acc1[1] += yl[i +  8] * (qs[i/2] & 0x000c);
            acc2[1] += yl[i +  9] * (qs[i/2] & 0x0c00);
            acc1[2] += yl[i + 16] * (qs[i/2] & 0x0030);
            acc2[2] += yl[i + 17] * (qs[i/2] & 0x3000);
            acc1[3] += yl[i + 24] * (qs[i/2] & 0x00c0);
            acc2[3] += yl[i + 25] * (qs[i/2] & 0xc000);
        }
        const float dall = (float)dh[0];
        const float dmin = (float)dh[1] * (1.f / 16.f);
        sumf += dall * ((acc1[0] + (1.f/256.f) * acc2[0]) * (sc[0] & 0xF) * (1.f/ 1.f) +
                        (acc1[1] + (1.f/256.f) * acc2[1]) * (sc[2] & 0xF) * (1.f/ 4.f) +
                        (acc1[2] + (1.f/256.f) * acc2[2]) * (sc[4] & 0xF) * (1.f/16.f) +
                        (acc1[3] + (1.f/256.f) * acc2[3]) * (sc[6] & 0xF) * (1.f/64.f))
              - dmin * (sumy[0] * (sc[0] & 0xF0) +
                        sumy[1] * (sc[2] & 0xF0) +
                        sumy[2] * (sc[4] & 0xF0) +
                        sumy[3] * (sc[6] & 0xF0));

        y4 += ib_stride * QK_K;
    }

    // Reduce within each simdgroup, then across the SG pair sharing the row.
    const float sg_sum = simd_sum(sumf);

    threadgroup float shmem[Q2_NR0 * 2];    // [row_off][sg_in_row]
    if (tiisg == 0) shmem[row_off * 2 + sg_in_row] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg < Q2_NR0 && tiisg == 0) {
        const uint out_row = row0 + sgitg;
        if (out_row < N) {
            C[out_row] = (half)(shmem[sgitg * 2 + 0] + shmem[sgitg * 2 + 1]);
        }
    }
}

// ----------------------------------------------------------------------------
// gemm_q2k_mv — self-contained Q2_K matvec for the quant_mv.c++ lab launcher.
// Distinct binding convention (struct args at buffer 0, single SG per row).
// ----------------------------------------------------------------------------

struct q2k_mv_args {
    int D;   // input dim (multiple of 256)
    int N;   // output dim
};

kernel void gemm_q2k_mv(
        constant q2k_mv_args & args     [[buffer(0)]],
        device const half    * x        [[buffer(1)]], // [D]
        device const char    * W        [[buffer(2)]], // [N * (D/256) * sizeof(block_q2_K)]
        device       half    * y        [[buffer(3)]], // [N]
        uint   tgpig                    [[threadgroup_position_in_grid]],
        ushort tiisg                    [[thread_index_in_simdgroup]])
{
    const int row = (int)tgpig;
    if (row >= args.N) return;

    const int nb = args.D / QK_K;
    const ulong row_bytes = (ulong)nb * sizeof(block_q2_K);
    device const block_q2_K * xb = (device const block_q2_K *)(W + row * row_bytes);

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const short is = (8 * ir) / 16;

    device const half * y4 = x + ix * QK_K + 128 * iq + 8 * ir;

    float sumf = 0.f;
    float yl[32];

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i +  0] = (float)y4[i +  0]; sumy[0] += yl[i +  0];
            yl[i +  8] = (float)y4[i + 32]; sumy[1] += yl[i +  8];
            yl[i + 16] = (float)y4[i + 64]; sumy[2] += yl[i + 16];
            yl[i + 24] = (float)y4[i + 96]; sumy[3] += yl[i + 24];
        }

        device const uint8_t  * sc = (device const uint8_t  *)xb[ib].scales + 8 * iq + is;
        device const uint16_t * qs = (device const uint16_t *)xb[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &xb[ib].d;

        float4 acc1 = {0.f, 0.f, 0.f, 0.f};
        float4 acc2 = {0.f, 0.f, 0.f, 0.f};
        for (int i = 0; i < 8; i += 2) {
            acc1[0] += yl[i +  0] * (qs[i/2] & 0x0003);
            acc2[0] += yl[i +  1] * (qs[i/2] & 0x0300);
            acc1[1] += yl[i +  8] * (qs[i/2] & 0x000c);
            acc2[1] += yl[i +  9] * (qs[i/2] & 0x0c00);
            acc1[2] += yl[i + 16] * (qs[i/2] & 0x0030);
            acc2[2] += yl[i + 17] * (qs[i/2] & 0x3000);
            acc1[3] += yl[i + 24] * (qs[i/2] & 0x00c0);
            acc2[3] += yl[i + 25] * (qs[i/2] & 0xc000);
        }
        float dall = (float)dh[0];
        float dmin = (float)dh[1] * (1.f/16.f);
        sumf += dall * ((acc1[0] + (1.f/256.f) * acc2[0]) * (sc[0] & 0xF) * (1.f/ 1.f) +
                        (acc1[1] + (1.f/256.f) * acc2[1]) * (sc[2] & 0xF) * (1.f/ 4.f) +
                        (acc1[2] + (1.f/256.f) * acc2[2]) * (sc[4] & 0xF) * (1.f/16.f) +
                        (acc1[3] + (1.f/256.f) * acc2[3]) * (sc[6] & 0xF) * (1.f/64.f))
              - dmin * (sumy[0] * (sc[0] & 0xF0) +
                        sumy[1] * (sc[2] & 0xF0) +
                        sumy[2] * (sc[4] & 0xF0) +
                        sumy[3] * (sc[6] & 0xF0));

        y4 += 4 * QK_K;
    }

    float total = simd_sum(sumf);
    if (tiisg == 0) {
        y[row] = (half)total;
    }
}
