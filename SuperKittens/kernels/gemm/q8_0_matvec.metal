// q8_0_matvec.metal — Q8_0 weight × fp16 activation matvec (decode-time, M=1).
//
// Each Q8_0 block:
//     half  d           (per-32-element scale)
//     int8  qs[32]
//   → 34 bytes / 32 weights.
//
// Layout: weights as row-major [N rows, K cols]; row r occupies K/32 contiguous
// blocks at byte offset r * (K/32) * 34.  Activation B is fp16 [K], same as the
// fp16 GEMM path.  Output C is fp16 [N].
//
// Dispatch (host): grid (N/NR0, 1, 1), threadgroup (NW * NSG, 1, 1).
//   - NW  = 32 (simdgroup width)
//   - NSG = 4  simdgroups per TG  (all SGs cooperate over the K dimension)
//   - NR0 = 2  output rows per TG (TG writes 2 rows; NOT NSG*NR0=8)
//
// One thread reads NQ=8 activations and 8 quantized weights per inner step; the
// simdgroup tiles over the K dimension by stepping NSG*NQ blocks at a time, then
// simd_sum reduces.
//
// Bindings (encode_q8_0_matvec):
//   0: B   activation,  fp16 [K]                           (offset = 0)
//   1: A   weights, q8_0 row-major [N, K]                  (offset can vary)
//   2: C   output,   fp16 [N]
//   3: uint K
//   4: uint N

#include <metal_stdlib>
using namespace metal;

#ifndef Q8_BLOCK
#define Q8_BLOCK 32
#endif

struct __attribute__((packed)) q8_block { half d; int8_t qs[Q8_BLOCK]; };

constant constexpr short Q8_NW  = 32;
constant constexpr short Q8_NSG = 4;
constant constexpr short Q8_NR0 = 2;
constant constexpr short Q8_NQ  = 8;

kernel void q8_0_matvec(
    device const half      *B      [[buffer(0)]],
    device const uchar     *A_raw  [[buffer(1)]],
    device       half      *C      [[buffer(2)]],
    constant uint           &K     [[buffer(3)]],
    constant uint           &N     [[buffer(4)]],
    uint3   tgpig [[threadgroup_position_in_grid]],
    ushort  tiisg [[thread_index_in_simdgroup]],
    ushort  sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nb = K / Q8_BLOCK;                 // blocks per row
    const uint row0 = tgpig.x * Q8_NR0;

    // Within each simdgroup, partition the 32 lanes as (NQ=8 groups of NW/NQ=4 lanes).
    const ushort ix = tiisg / (Q8_NW / Q8_NQ);    // 0..7  (which 8-wide block)
    const ushort il = tiisg % (Q8_NW / Q8_NQ);    // 0..3  (which 8 elems inside)
    const uint   ib0 = (uint)sgitg * Q8_NQ + ix;

    // Activation pointer for this lane.
    device const half *yb = B + ib0 * Q8_BLOCK + il * Q8_NQ;

    // Row base pointers.
    device const q8_block *ax[Q8_NR0];
    #pragma unroll
    for (short r = 0; r < Q8_NR0; ++r) {
        const uint row = row0 + r;
        ax[r] = (device const q8_block *)(A_raw + (size_t)row * nb * sizeof(q8_block));
    }

    float sumf[Q8_NR0] = {0.f};
    half  yl[Q8_NQ];

    for (uint ib = ib0; ib < nb; ib += Q8_NSG * Q8_NQ) {
        #pragma unroll
        for (short i = 0; i < Q8_NQ; ++i) yl[i] = yb[i];

        #pragma unroll
        for (short r = 0; r < Q8_NR0; ++r) {
            device const int8_t *qs = ax[r][ib].qs + il * Q8_NQ;
            float sumq = 0.f;
            #pragma unroll
            for (short i = 0; i < Q8_NQ; ++i) sumq += (float)qs[i] * (float)yl[i];
            sumf[r] += sumq * (float)ax[r][ib].d;
        }
        yb += Q8_NSG * Q8_NQ * Q8_BLOCK;
    }

    // Reduce across the simdgroup (lanes), then across simdgroups via threadgroup memory.
    threadgroup float shmem[Q8_NR0 * Q8_NSG];
    #pragma unroll
    for (short r = 0; r < Q8_NR0; ++r) {
        float v = simd_sum(sumf[r]);
        if (tiisg == 0) shmem[r * Q8_NSG + sgitg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        // All 32 lanes participate in simd_sum (only lanes 0..NSG-1 carry data).
        #pragma unroll
        for (short r = 0; r < Q8_NR0; ++r) {
            float v = (tiisg < Q8_NSG) ? shmem[r * Q8_NSG + tiisg] : 0.0f;
            v = simd_sum(v);
            if (tiisg == 0) {
                const uint out_row = row0 + r;
                if (out_row < N) C[out_row] = (half)v;
            }
        }
    }
}
