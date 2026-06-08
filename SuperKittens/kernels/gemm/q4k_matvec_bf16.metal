// q4k_matvec_bf16.metal — bf16-activation / bf16-output variant of q4k_matvec.
//
// Identical inner math to kernels/gemm/q4k_matvec.metal (half acts), but reads
// bf16 activations and writes bf16 outputs to match the gemma4 pipeline (bf16
// end-to-end). Lets the gemma4_unified Q4_K body route through the GGUF's native
// Q4_K blocks (no dequant→Q8 re-quant), halving the resident body bytes vs Q8.
//
// Bindings (mirror q8_0_matvec_bf16):
//   0: B activation, bf16 [K]
//   1: A weights, q4_K row-major [N, K]
//   2: C output, bf16 [N]
//   3: uint K
//   4: uint N

#include <metal_stdlib>
using namespace metal;

#ifndef QK_K
#define QK_K 256
#endif

struct block_q4_K_b {
    half  d;
    half  dmin;
    uchar scales[12];
    uchar qs[QK_K / 2];
};

constant constexpr short Q4B_NW  = 32;
constant constexpr short Q4B_NSG = 4;
constant constexpr short Q4B_NR0 = 2;

[[host_name("q4k_matvec_bf16")]]
kernel void q4k_matvec_bf16(
    device const bfloat     *B      [[buffer(0)]],
    device const uchar      *A_raw  [[buffer(1)]],
    device       bfloat     *C      [[buffer(2)]],
    constant uint            &K     [[buffer(3)]],
    constant uint            &N     [[buffer(4)]],
    uint3   tgpig [[threadgroup_position_in_grid]],
    ushort  tiisg [[thread_index_in_simdgroup]],
    ushort  sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const uint nb   = K / QK_K;
    const uint row0 = tgpig.x * Q4B_NR0;

    const short row_off   = sgitg / 2;
    const short sg_in_row = sgitg % 2;
    const uint  row = row0 + row_off;
    if (row >= N) return;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    device const block_q4_K_b *xrow =
        (device const block_q4_K_b *)(A_raw + (size_t)row * nb * sizeof(block_q4_K_b));

    const uint ib_start  = (uint)ix + 4u * (uint)sg_in_row;
    const uint ib_stride = 4u * 2u;

    device const bfloat *y4_h = B + ib_start * QK_K + 64 * iq + 8 * ir;

    float yl[16];
    float yh[16];
    float sumf = 0.f;

    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (uint ib = ib_start; ib < nb; ib += ib_stride) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = (float)y4_h[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = (float)y4_h[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = (float)y4_h[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = (float)y4_h[i + 160]; sumy[3] += yh[i + 8];
        }

        device const uint16_t *sc = (device const uint16_t *)xrow[ib].scales + iq;
        device const uint16_t *q1 = (device const uint16_t *)xrow[ib].qs + 16 * iq + 4 * ir;
        device const uint16_t *q2 = q1 + 32;
        device const half     *dh = &xrow[ib].d;

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

        y4_h += ib_stride * QK_K;
    }

    const float sg_sum = simd_sum(sumf);

    threadgroup float shmem[Q4B_NR0 * 2];
    if (tiisg == 0) shmem[row_off * 2 + sg_in_row] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg < Q4B_NR0 && tiisg == 0) {
        const uint out_row = row0 + sgitg;
        if (out_row < N) {
            C[out_row] = (bfloat)(shmem[sgitg * 2 + 0] + shmem[sgitg * 2 + 1]);
        }
    }
}
