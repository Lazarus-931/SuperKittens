// fp16 activation x Q4_K weight matvec.

#include <metal_stdlib>
using namespace metal;

#define QK_K 256

struct block_q4_K {
    half  d;
    half  dmin;
    uchar scales[12];
    uchar qs[QK_K/2];
};

struct q4k_mv_args {
    uint D;   // input dim (multiple of 256)
    uint N;   // output dim
};

// Each TG: tgpig.x = output row r, 32 threads cooperate.
// Layout of work per super-block (256 weights):
//   ix = tiisg / 8  (4 groups, each covers one of 4 super-blocks per stride)
//   it = tiisg % 8
//   iq = it / 4    (0..1)  selects which half of the 256 (low/high 128)
//   ir = it % 4    (0..3)  selects which quartet of 8 floats within the 64-block
// We iterate ib over [ix, nb, step=4]. Inside, load 16 activation entries
// at positions y[ib*256 + 64*iq + 8*ir + {0..7, 32..39, 128..135, 160..167}].
[[host_name("q4k_matvec")]]
kernel void gemm_q4k_mv(
        constant q4k_mv_args & args [[buffer(0)]],
        device const half    * x    [[buffer(1)]],   // [D] fp16
        device const char    * W    [[buffer(2)]],   // [N, D] q4_K
        device       half    * y    [[buffer(3)]],   // [N] fp16
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort  tiisg [[thread_index_in_simdgroup]]) {

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const uint r = tgpig.x;
    if (r >= args.N) return;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    const uint  nb = args.D / QK_K;          // blocks per row
    const ulong row_bytes = (ulong)nb * sizeof(block_q4_K);

    device const block_q4_K * xrow =
        (device const block_q4_K *)(W + (ulong)r * row_bytes);

    // We need fp32 activations for the accumulation math; load fp16 and cast.
    device const half * y4_h = x + ix * QK_K + 64 * iq + 8 * ir;

    float yl[16];
    float yh[16];
    float sumf = 0.f;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (uint ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = (float)y4_h[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = (float)y4_h[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = (float)y4_h[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = (float)y4_h[i + 160]; sumy[3] += yh[i + 8];
        }

        device const uint16_t * sc = (device const uint16_t *)xrow[ib].scales + iq;
        device const uint16_t * q1 = (device const uint16_t *)xrow[ib].qs + 16 * iq + 4 * ir;
        device const uint16_t * q2 = q1 + 32;
        device const half     * dh = &xrow[ib].d;

        sc16[0] = sc[0] & kmask1;
        sc16[1] = sc[2] & kmask1;
        sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
        sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

        float4 acc1 = {0.f, 0.f, 0.f, 0.f};
        float4 acc2 = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 4; ++i) {
            acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
            acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
            acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
            acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
            acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
            acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
            acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
            acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
        }

        const float d    = (float)dh[0];
        const float dmin = (float)dh[1];

        sumf += d * ((acc1[0] + (1.f/256.f) * acc1[1]) * sc8[0] +
                     (acc1[2] + (1.f/256.f) * acc1[3]) * sc8[1] * (1.f/16.f) +
                     (acc2[0] + (1.f/256.f) * acc2[1]) * sc8[4] +
                     (acc2[2] + (1.f/256.f) * acc2[3]) * sc8[5] * (1.f/16.f))
              - dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                        sumy[2] * sc8[6] + sumy[3] * sc8[7]);

        y4_h += 4 * QK_K;
    }

    const float total = simd_sum(sumf);
    if (tiisg == 0) {
        y[r] = (half)total;
    }
}
