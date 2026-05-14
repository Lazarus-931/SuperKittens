// q8_0_matvec_bf16.metal — bf16-activation / bf16-output variant of q8_0_matvec.
//
// Identical algorithm to kernels/gemm/q8_0_matvec.metal (half acts), but
// reads bf16 activations and writes bf16 outputs to match the gemma4 pipeline
// (which is bf16 end-to-end).
//
// Each Q8_0 block:
//     half  d           (per-32-element scale)
//     int8  qs[32]
//   → 34 bytes / 32 weights.
//
// Bindings (encode_q8_0_matvec_bf16):
//   0: B   activation,  bf16 [K]
//   1: A   weights, q8_0 row-major [N, K]
//   2: C   output,   bf16 [N]
//   3: uint K
//   4: uint N

#include <metal_stdlib>
using namespace metal;

#ifndef Q8B_BLOCK
#define Q8B_BLOCK 32
#endif

struct __attribute__((packed)) q8b_block { half d; int8_t qs[Q8B_BLOCK]; };

constant constexpr short Q8B_NW  = 32;
constant constexpr short Q8B_NSG = 4;
constant constexpr short Q8B_NR0 = 2;
constant constexpr short Q8B_NQ  = 8;

[[host_name("q8_0_matvec_bf16")]]
kernel void q8_0_matvec_bf16(
    device const bfloat    *B      [[buffer(0)]],
    device const uchar     *A_raw  [[buffer(1)]],
    device       bfloat    *C      [[buffer(2)]],
    constant uint           &K     [[buffer(3)]],
    constant uint           &N     [[buffer(4)]],
    uint3   tgpig [[threadgroup_position_in_grid]],
    ushort  tiisg [[thread_index_in_simdgroup]],
    ushort  sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nb = K / Q8B_BLOCK;
    const uint row0 = tgpig.x * Q8B_NR0;

    const ushort ix = tiisg / (Q8B_NW / Q8B_NQ);
    const ushort il = tiisg % (Q8B_NW / Q8B_NQ);
    const uint   ib0 = (uint)sgitg * Q8B_NQ + ix;

    device const bfloat *yb = B + ib0 * Q8B_BLOCK + il * Q8B_NQ;

    device const q8b_block *ax[Q8B_NR0];
    #pragma unroll
    for (short r = 0; r < Q8B_NR0; ++r) {
        const uint row = row0 + r;
        ax[r] = (device const q8b_block *)(A_raw + (size_t)row * nb * sizeof(q8b_block));
    }

    float sumf[Q8B_NR0] = {0.f};
    float yl[Q8B_NQ];

    for (uint ib = ib0; ib < nb; ib += Q8B_NSG * Q8B_NQ) {
        #pragma unroll
        for (short i = 0; i < Q8B_NQ; ++i) yl[i] = (float)yb[i];

        #pragma unroll
        for (short r = 0; r < Q8B_NR0; ++r) {
            device const int8_t *qs = ax[r][ib].qs + il * Q8B_NQ;
            float sumq = 0.f;
            #pragma unroll
            for (short i = 0; i < Q8B_NQ; ++i) sumq += (float)qs[i] * yl[i];
            sumf[r] += sumq * (float)ax[r][ib].d;
        }
        yb += Q8B_NSG * Q8B_NQ * Q8B_BLOCK;
    }

    threadgroup float shmem[Q8B_NR0 * Q8B_NSG];
    #pragma unroll
    for (short r = 0; r < Q8B_NR0; ++r) {
        float v = simd_sum(sumf[r]);
        if (tiisg == 0) shmem[r * Q8B_NSG + sgitg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        #pragma unroll
        for (short r = 0; r < Q8B_NR0; ++r) {
            float v = (tiisg < Q8B_NSG) ? shmem[r * Q8B_NSG + tiisg] : 0.0f;
            v = simd_sum(v);
            if (tiisg == 0) {
                const uint out_row = row0 + r;
                if (out_row < N) C[out_row] = (bfloat)v;
            }
        }
    }
}
