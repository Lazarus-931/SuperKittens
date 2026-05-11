// Self-contained Q2_K matvec for SuperKittens.

#include <metal_stdlib>
using namespace metal;

#define QK_K 256

struct block_q2_K {
    uchar scales[QK_K/16]; // 16 bytes: low nibble = scale, high nibble = min
    uchar qs[QK_K/4];      // 64 bytes
    half  d;               // super-scale
    half  dmin;            // super-min
};

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

    // Lane layout (matches ds4 reference)
    const short ix = tiisg / 8;   // 0..3 : which block-step within stride of 4
    const short it = tiisg % 8;   // 0..7
    const short iq = it / 4;      // 0 or 1 : selects 128-elem half of QK_K
    const short ir = it % 4;      // 0..3
    const short is = (8 * ir) / 16; // 0 or 1

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
