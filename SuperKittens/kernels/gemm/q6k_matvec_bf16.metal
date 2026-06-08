// q6k_matvec_bf16.metal — bf16-activation / bf16-output variant of q6k_matvec.
//
// Identical inner math to kernels/gemm/q6k_matvec.metal (half acts), but reads
// bf16 activations and writes bf16 outputs to match the gemma4 pipeline. The
// Q4_K_M GGUF stores some attn_v / ffn_down projections as Q6_K, so the
// gemma4_unified Q4_K body needs a bf16 Q6_K matvec for those rows (the rest
// route through q4k_matvec_bf16).
//
// Bindings (mirror q8_0_matvec_bf16):
//   0: B activation, bf16 [K]
//   1: A weights, q6_K row-major [N, K]
//   2: C output, bf16 [N]
//   3: uint K
//   4: uint N

#include <metal_stdlib>
using namespace metal;

#ifndef QK_K
#define QK_K 256
#endif

struct __attribute__((packed)) block_q6_K_b {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t  scales[QK_K / 16];
    half    d;
};

constant constexpr short Q6B_NW  = 32;
constant constexpr short Q6B_NSG = 4;
constant constexpr short Q6B_NR0 = 2;

[[host_name("q6k_matvec_bf16")]]
kernel void q6k_matvec_bf16(
    device const bfloat     *B      [[buffer(0)]],
    device const uchar      *A_raw  [[buffer(1)]],
    device       bfloat     *C      [[buffer(2)]],
    constant uint            &K     [[buffer(3)]],
    constant uint            &N     [[buffer(4)]],
    uint3   tgpig [[threadgroup_position_in_grid]],
    ushort  tiisg [[thread_index_in_simdgroup]],
    ushort  sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nb   = K / QK_K;
    const uint row0 = tgpig.x * Q6B_NR0;

    const short l  = tiisg;
    const short is = l / 16;

    device const block_q6_K_b *ax[Q6B_NR0];
    #pragma unroll
    for (short r = 0; r < Q6B_NR0; ++r) {
        const uint row = row0 + r;
        ax[r] = (device const block_q6_K_b *)(A_raw + (size_t)row * nb * sizeof(block_q6_K_b));
    }

    float sumf[Q6B_NR0] = {0.f};

    for (uint ib = sgitg; ib < nb; ib += Q6B_NSG) {
        device const bfloat *yb = B + ib * QK_K;

        const float y0 = (float)yb[l +   0];
        const float y1 = (float)yb[l +  32];
        const float y2 = (float)yb[l +  64];
        const float y3 = (float)yb[l +  96];
        const float y4 = (float)yb[l + 128];
        const float y5 = (float)yb[l + 160];
        const float y6 = (float)yb[l + 192];
        const float y7 = (float)yb[l + 224];

        #pragma unroll
        for (short r = 0; r < Q6B_NR0; ++r) {
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

    threadgroup float shmem[Q6B_NR0 * Q6B_NSG];
    #pragma unroll
    for (short r = 0; r < Q6B_NR0; ++r) {
        float v = simd_sum(sumf[r]);
        if (tiisg == 0) shmem[r * Q6B_NSG + sgitg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        #pragma unroll
        for (short r = 0; r < Q6B_NR0; ++r) {
            float v = (tiisg < Q6B_NSG) ? shmem[r * Q6B_NSG + tiisg] : 0.0f;
            v = simd_sum(v);
            if (tiisg == 0) {
                const uint out_row = row0 + r;
                if (out_row < N) C[out_row] = (bfloat)v;
            }
        }
    }
}
