// q3k_matvec.metal — Q3_K weight × fp16 activation matvec (decode-time, M=1).
//
// The standard Q3_K_M / Q2_K_M K-quant recipes store the biggest projections
// (ffn_down, attn_output, and the LM head) as Q3_K. Without this kernel SK
// host-dequants those to fp16 (16 bpw), so the per-token byte savings of a
// 3-bit quant are lost on exactly the largest matvecs. This keeps Q3_K packed
// (110 B / 256 weights ≈ 3.44 bpw) end-to-end.
//
// Block layout (GGML block_q3_K, 110 B / 256 weights; per llama.cpp
// dequantize_row_q3_K, validated against gguf.quants.Q3_K and SK's bit-exact
// host dequant_q3_k_to_fp16):
//     uint8 hmask[32]   // high (3rd) bit per weight; a SET bit drops the +4 bias
//     uint8 qs[64]      // low 2 bits per weight, 4 weights per byte
//     uint8 scales[12]  // 16 signed 6-bit sub-scales, packed (offset -32)
//     half  d           // super-block scale
//
// ggml element order: n = group*128 + shift*32 + pos, group∈{0,1}, shift∈[0,4),
// pos∈[0,32). scale index = n>>4 (per-16). qs byte = group*32+pos, value at
// bit-shift 2*shift. hmask bit (group*4+shift) of byte pos — SET → no +4 bias
// (hbit=0), CLEAR → subtract 4 (hbit=4). w = d * (scale[n>>4]-32) * (q2 - hbit).
//
// Dispatch (host): grid (ceil(N/NR0), 1, 1), threadgroup (NW * NSG, 1, 1).
//   NW=32 lanes, NSG=4 simdgroups cooperate over the K (super-block) dimension,
//   NR0=2 output rows per TG. Lane l owns pos=l; the 32 lanes tile the 32 pos
//   values of a super-block once, each accumulating all 8 (group,shift) taps —
//   so one simdgroup covers all 256 weights of a block per step.
//
// Bindings (mirror q4k/q6k/q8_0_matvec):
//   0: B activation, fp16 [K]
//   1: A weights, q3_K row-major [N, K]
//   2: C output, fp16 [N]
//   3: uint K
//   4: uint N

#include <metal_stdlib>
using namespace metal;

#ifndef QK_K
#define QK_K 256
#endif

struct __attribute__((packed)) block_q3_K {
    uint8_t hmask[QK_K / 8];   // 32
    uint8_t qs[QK_K / 4];      // 64
    uint8_t scales[12];        // 16 signed 6-bit sub-scales
    half    d;
};

constant constexpr short Q3_NW  = 32;
constant constexpr short Q3_NSG = 4;
constant constexpr short Q3_NR0 = 2;

kernel void q3k_matvec(
    device const half       *B      [[buffer(0)]],
    device const uchar      *A_raw  [[buffer(1)]],
    device       half       *C      [[buffer(2)]],
    constant uint            &K     [[buffer(3)]],
    constant uint            &N     [[buffer(4)]],
    uint3   tgpig [[threadgroup_position_in_grid]],
    ushort  tiisg [[thread_index_in_simdgroup]],
    ushort  sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr uint32_t kmask1 = 0x03030303;
    constexpr uint32_t kmask2 = 0x0f0f0f0f;

    const uint nb   = K / QK_K;            // super-blocks per row
    const uint row0 = tgpig.x * Q3_NR0;

    const short pos = tiisg;               // each lane owns one pos in [0,32)

    device const block_q3_K *ax[Q3_NR0];
    #pragma unroll
    for (short r = 0; r < Q3_NR0; ++r) {
        const uint row = row0 + r;
        ax[r] = (device const block_q3_K *)(A_raw + (size_t)row * nb * sizeof(block_q3_K));
    }

    float sumf[Q3_NR0] = {0.f};

    for (uint ib = sgitg; ib < nb; ib += Q3_NSG) {
        device const half *yb = B + ib * QK_K;

        // The 8 (group,shift) taps map to y[pos + 32*g4s] for g4s in [0,8).
        const float y0 = (float)yb[pos +   0];
        const float y1 = (float)yb[pos +  32];
        const float y2 = (float)yb[pos +  64];
        const float y3 = (float)yb[pos +  96];
        const float y4 = (float)yb[pos + 128];
        const float y5 = (float)yb[pos + 160];
        const float y6 = (float)yb[pos + 192];
        const float y7 = (float)yb[pos + 224];

        #pragma unroll
        for (short r = 0; r < Q3_NR0; ++r) {
            device const uint8_t *hmask = ax[r][ib].hmask;
            device const uint8_t *qs    = ax[r][ib].qs;

            // Unpack the 16 signed 6-bit sub-scales (offset -32), per-block.
            // scales[12] is 2-aligned in every block (offset 96 + 110·ib, both
            // even), so read as uint16 and only materialize the 6 scale values
            // this lane actually taps: scale[n>>4] for n = 32k+pos with
            // k∈[0,8). pos∈[0,32) → n>>4 ∈ { (32k+pos)>>4 } takes the values
            // 2k and 2k+(pos>=16). Two scale indices per group-of-4 → 8 used,
            // but consecutive shifts in a group share one index pair, so unpack
            // all 16 once into registers (cheap vs the per-element reshuffle).
            uint32_t aux[4];
            {
                device const uint16_t *sh = (device const uint16_t *)ax[r][ib].scales;
                uint32_t a0 = (uint32_t)sh[0] | ((uint32_t)sh[1] << 16);
                uint32_t a1 = (uint32_t)sh[2] | ((uint32_t)sh[3] << 16);
                uint32_t a2 = (uint32_t)sh[4] | ((uint32_t)sh[5] << 16);
                aux[2] = ((a0 >> 4) & kmask2) | (((a2 >> 4) & kmask1) << 4);
                aux[3] = ((a1 >> 4) & kmask2) | (((a2 >> 6) & kmask1) << 4);
                aux[0] = (a0 & kmask2) | (((a2 >> 0) & kmask1) << 4);
                aux[1] = (a1 & kmask2) | (((a2 >> 2) & kmask1) << 4);
            }
            thread const int8_t *sc8 = (thread const int8_t *)aux;
            // The two scale indices this lane uses per group (pos<16 vs >=16).
            const short shi = pos >> 4;            // 0 or 1
            const float sg0a = (float)(sc8[0 + shi] - 32);
            const float sg0b = (float)(sc8[2 + shi] - 32);
            const float sg0c = (float)(sc8[4 + shi] - 32);
            const float sg0d = (float)(sc8[6 + shi] - 32);
            const float sg1a = (float)(sc8[8  + shi] - 32);
            const float sg1b = (float)(sc8[10 + shi] - 32);
            const float sg1c = (float)(sc8[12 + shi] - 32);
            const float sg1d = (float)(sc8[14 + shi] - 32);

            const uint8_t hm = hmask[pos];
            const uint8_t q0b = qs[pos];        // group 0 quants (shifts 0..3)
            const uint8_t q1b = qs[pos + 32];   // group 1 quants (shifts 0..3)

            float bsum = 0.f;
            // group 0: n = shift*32 + pos, hmask bit (0*4 + shift) = 1<<shift
            bsum += y0 * sg0a * (float)((int)((q0b >> 0) & 3) - ((hm & 0x01) ? 0 : 4));
            bsum += y1 * sg0b * (float)((int)((q0b >> 2) & 3) - ((hm & 0x02) ? 0 : 4));
            bsum += y2 * sg0c * (float)((int)((q0b >> 4) & 3) - ((hm & 0x04) ? 0 : 4));
            bsum += y3 * sg0d * (float)((int)((q0b >> 6) & 3) - ((hm & 0x08) ? 0 : 4));
            // group 1: n = 128 + shift*32 + pos, hmask bit (1*4 + shift) = 1<<(4+shift)
            bsum += y4 * sg1a * (float)((int)((q1b >> 0) & 3) - ((hm & 0x10) ? 0 : 4));
            bsum += y5 * sg1b * (float)((int)((q1b >> 2) & 3) - ((hm & 0x20) ? 0 : 4));
            bsum += y6 * sg1c * (float)((int)((q1b >> 4) & 3) - ((hm & 0x40) ? 0 : 4));
            bsum += y7 * sg1d * (float)((int)((q1b >> 6) & 3) - ((hm & 0x80) ? 0 : 4));
            sumf[r] += bsum * (float)ax[r][ib].d;
        }
    }

    threadgroup float shmem[Q3_NR0 * Q3_NSG];
    #pragma unroll
    for (short r = 0; r < Q3_NR0; ++r) {
        float v = simd_sum(sumf[r]);
        if (tiisg == 0) shmem[r * Q3_NSG + sgitg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        #pragma unroll
        for (short r = 0; r < Q3_NR0; ++r) {
            float v = (tiisg < Q3_NSG) ? shmem[r * Q3_NSG + tiisg] : 0.0f;
            v = simd_sum(v);
            if (tiisg == 0) {
                const uint out_row = row0 + r;
                if (out_row < N) C[out_row] = (half)v;
            }
        }
    }
}
