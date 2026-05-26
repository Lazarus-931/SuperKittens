// q8_0_matvec_addres.metal — Q8_0 matvec with fused residual add:
//     C[row] = Res[row] + (W @ x)[row]
// Used for the qwen MLP down-projection where the next step is the
// y_out = y_attn + mlp_out residual; folding the add into the matvec
// drops one elementwise dispatch + one L2-drain fence per layer at decode.

#include <metal_stdlib>
using namespace metal;

#ifndef Q8_BLOCK
#define Q8_BLOCK 32
#endif

struct __attribute__((packed)) q8_block_ar { half d; int8_t qs[Q8_BLOCK]; };

constant constexpr short AR_NW  = 32;
constant constexpr short AR_NSG = 4;
constant constexpr short AR_NR0 = 2;
constant constexpr short AR_NQ  = 8;

[[host_name("q8_0_matvec_addres")]]
kernel void q8_0_matvec_addres(
    device const half      *B      [[buffer(0)]],   // x, fp16 [K]
    device const uchar     *A_raw  [[buffer(1)]],   // W, Q8_0 [N, K]
    device const half      *R      [[buffer(2)]],   // residual, fp16 [N]
    device       half      *C      [[buffer(3)]],   // out, fp16 [N]
    constant uint           &K     [[buffer(4)]],
    constant uint           &N     [[buffer(5)]],
    uint3   tgpig [[threadgroup_position_in_grid]],
    ushort  tiisg [[thread_index_in_simdgroup]],
    ushort  sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nb = K / Q8_BLOCK;
    const uint row0 = tgpig.x * AR_NR0;

    const ushort ix = tiisg / (AR_NW / AR_NQ);
    const ushort il = tiisg % (AR_NW / AR_NQ);
    const uint   ib0 = (uint)sgitg * AR_NQ + ix;

    device const half *yb = B + ib0 * Q8_BLOCK + il * AR_NQ;

    device const q8_block_ar *ax[AR_NR0];
    #pragma unroll
    for (short r = 0; r < AR_NR0; ++r) {
        const uint row = row0 + r;
        ax[r] = (device const q8_block_ar *)(A_raw + (size_t)row * nb * sizeof(q8_block_ar));
    }

    float sumf[AR_NR0] = {0.f};
    half  yl[AR_NQ];

    for (uint ib = ib0; ib < nb; ib += AR_NSG * AR_NQ) {
        #pragma unroll
        for (short i = 0; i < AR_NQ; ++i) yl[i] = yb[i];
        #pragma unroll
        for (short r = 0; r < AR_NR0; ++r) {
            device const int8_t *qs = ax[r][ib].qs + il * AR_NQ;
            float sumq = 0.f;
            #pragma unroll
            for (short i = 0; i < AR_NQ; ++i) sumq += (float)qs[i] * (float)yl[i];
            sumf[r] += sumq * (float)ax[r][ib].d;
        }
        yb += AR_NSG * AR_NQ * Q8_BLOCK;
    }

    threadgroup float shmem[AR_NR0 * AR_NSG];
    #pragma unroll
    for (short r = 0; r < AR_NR0; ++r) {
        float v = simd_sum(sumf[r]);
        if (tiisg == 0) shmem[r * AR_NSG + sgitg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        #pragma unroll
        for (short r = 0; r < AR_NR0; ++r) {
            float v = (tiisg < AR_NSG) ? shmem[r * AR_NSG + tiisg] : 0.0f;
            v = simd_sum(v);
            if (tiisg == 0) {
                const uint out_row = row0 + r;
                if (out_row < N) C[out_row] = (half)((float)R[out_row] + v);
            }
        }
    }
}
