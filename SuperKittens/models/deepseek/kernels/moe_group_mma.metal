// moe_group_mma.metal — DeepSeek V2-Lite PREFILL (T>1) MMA-tiled grouped MoE.
//
// The per-slot mul_mv_id matvec dispatches one M=1 GEMV per (token,slot): every
// routed slot re-reads its expert's full weight slab. At prefill (T>1) that is
// ~(T*top_k / experts_touched) redundant weight reads AND a per-row matvec that
// is throughput-suboptimal vs an MMA tile. The earlier SCALAR grouped kernel
// (moe_group.metal) reused the weight read but kept the per-row scalar dot, so
// it stayed occupancy-bound and lost. This file groups the routed slots by
// expert (the existing deepseek_moe_group_build counting sort) and runs a
// simdgroup-MMA tile per expert: each weight K-tile is dequant'd into TG memory
// ONCE and reused across the expert's BM-row activation tile (gathered + cast to
// fp16). Math is C[m,n] = sum_k A[m,k]*W[n,k] (A @ W^T), the same dot products as
// the per-slot path, reordered. Activated only for T>1 (decode T=1 keeps the
// per-slot path byte-identical). DeepSeek-specific (own host_names, own file).
//
// host_names:
//   deepseek_moe_mma_q4k_grp   (gate/up: Q4_K weight, K=d_model 256-aligned)
//   deepseek_moe_mma_q5_0_grp  (down:    Q5_0 weight, K=n_int 32-aligned)

#include "ds4_preamble.h"

#ifndef QK_K
#define QK_K 256
#endif

namespace skmoemma {

// BM=32 rows (expert token-tile), BN=32 cols (output rows), BK=32. NSG=2
// simdgroups: each owns BN/NSG=16 output cols (MC=2 8-wide) and all BM rows
// (MR=4 8-row tiles). Mirrors kernels/gemm/gemm_mma.metal's geometry.
enum : uint { BM = 32, BN = 32, BK = 32, NSG = 2, MR = 4, MC = 2 };

struct block_q4_K_m {
    half  d;
    half  dmin;
    uchar scales[12];
    uchar qs[QK_K / 2];
};

// Gather A[BM][BK] for this expert's m-tile: row r (0..BM) maps to the global
// slot group_slots[mb+r]; the activation row is token = slot/top_k. The src is
// fp32 [*, K] (moe_x_f32 for gate/up indexed by token; moe_mid_f32 for down
// indexed by slot). Padded rows zeroed. 64 threads cooperatively load, cast to
// fp16. `slot_to_arow` picks token (gate/up) vs slot (down) row of src.
template <bool ROW_IS_SLOT>
inline void gather_A(threadgroup half* As,
                     device const float* src,
                     device const int32_t* group_slots,
                     uint mb, uint mcnt, uint k0, uint K, uint top_k,
                     uint lid)
{
    for (uint i = lid; i < BM * BK; i += 64) {
        const uint r = i / BK;
        const uint c = i % BK;
        half v = half(0);
        if (r < mcnt) {
            const int slot = group_slots[mb + r];
            const uint arow = ROW_IS_SLOT ? (uint)slot : ((uint)slot / top_k);
            const uint gc = k0 + c;
            if (gc < K) v = (half)src[(size_t)arow * K + gc];
        }
        As[i] = v;
    }
}

// Q4_K weight tile loader (per-expert): W base already offset to this expert's
// slab. column-parallel sub-scale hoist, same dequant as gemm_mma.metal's
// load_W_q4k. nb01 = weight row stride (bytes). out_rows = N (expert out dim).
inline void load_W_q4k_grp(threadgroup half* Ws,
                           device const uchar* W,
                           uint bc, uint k0, uint N, uint K, ulong nb01,
                           uint lid)
{
    const uint sb  = k0 / QK_K;        // super-block index within the row
    const uint sub = (k0 / 32) % 8;    // sub-block 0..7
    const uint wr  = lid;              // 0..63; only 0..BN-1 work
    if (wr >= BN) return;
    const uint gn = bc + wr;
    if (gn < N) {
        device const block_q4_K_m* blk =
            (device const block_q4_K_m*)(W + (size_t)gn * nb01 + (size_t)sb * sizeof(block_q4_K_m));
        const float d    = (float)blk->d;
        const float dmin = (float)blk->dmin;
        uint8_t sc, m;
        device const uchar* q = blk->scales;
        if (sub < 4) {
            sc = q[sub] & 63;
            m  = q[sub + 4] & 63;
        } else {
            sc = (q[sub + 4] & 0x0F) | ((q[sub - 4] >> 6) << 4);
            m  = (q[sub + 4] >>   4) | ((q[sub    ] >> 6) << 4);
        }
        const float dsc = d * sc, dm = dmin * m;
        device const uchar* qs = blk->qs + 32 * (sub / 2);
        const bool hi = (sub & 1);
        for (uint wk = 0; wk < BK; ++wk) {
            const uint8_t byte = qs[wk];
            const uint8_t qv = hi ? (byte >> 4) : (byte & 0x0F);
            Ws[wk * BN + wr] = (half)(dsc * (float)qv - dm);
        }
    } else {
        for (uint wk = 0; wk < BK; ++wk) Ws[wk * BN + wr] = half(0);
    }
}

// Q5_0 weight tile loader (per-expert, down). One Q5_0 block = 32 weights = one
// BK=32 slice of one output row. Each of BN output rows contributes its block at
// k0/32. Dequant matches deepseek_mul_mv_id_q5_0 / ggml: q = ((qs&0xF)|hbit<<4)-16.
inline void load_W_q5_0_grp(threadgroup half* Ws,
                            device const uchar* W,
                            uint bc, uint k0, uint N, uint K, ulong nb01,
                            uint lid)
{
    const uint kb = k0 / 32;           // block index within the row
    for (uint i = lid; i < BN * BK; i += 64) {
        const uint wr = i / BK;        // output column within tile
        const uint wk = i % BK;        // 0..31 weight within the block
        const uint gn = bc + wr;
        half v = half(0);
        if (gn < N) {
            device const block_q5_0* blk =
                (device const block_q5_0*)(W + (size_t)gn * nb01 + (size_t)kb * sizeof(block_q5_0));
            uint32_t qh = (uint32_t)blk->qh[0] | ((uint32_t)blk->qh[1] << 8)
                        | ((uint32_t)blk->qh[2] << 16) | ((uint32_t)blk->qh[3] << 24);
            const float d = (float)blk->d;
            // weight wk: low nibble of qs[wk&15] for wk<16, high nibble for wk>=16.
            const uint8_t byte = blk->qs[wk & 15];
            const int lo = (wk < 16) ? (byte & 0x0F) : (byte >> 4);
            const int hb = (int)((qh >> wk) & 1) << 4;
            v = (half)((float)((lo | hb) - 16) * d);
        }
        Ws[wk * BN + wr] = v;
    }
}

} // namespace skmoemma

using namespace skmoemma;

// Grouped MMA body. Grid: (ceil(N/BN), n_expert, ceil(maxM/BM)). gid.y = expert,
// gid.z = m-tile within the expert's bucket, gid.x = output-col tile. Each
// threadgroup handles one (expert, m-tile, n-tile). Output dst is per-slot:
// dst + slot*N + out_row (slot = group_slots[mb+r]), matching the per-slot
// kernel's [T, top_k, N] = slot-indexed layout.
#define MOE_MMA_BODY(LOAD_W, ROW_IS_SLOT)                                      \
    const int expert = (int)gid.y;                                             \
    const int beg = group_off[expert];                                         \
    const int end = group_off[expert + 1];                                     \
    const uint mb = (uint)beg + gid.z * BM;                                    \
    if ((int)mb >= end) return;                                                \
    const uint mcnt = min((uint)BM, (uint)end - mb);                           \
    const uint bc = gid.x * BN;                                                \
    device const uchar* W = src0 + (size_t)expert * nb02;                      \
    threadgroup half As[BM * BK];                                              \
    threadgroup half Ws[BK * BN];                                              \
    const uint lid = simd * 32 + lane;                                         \
    const uint c0  = simd * MC * 8;                                            \
    simdgroup_float8x8 acc[MR][MC] = {};                                       \
    for (uint k0 = 0; k0 < K; k0 += BK) {                                      \
        gather_A<ROW_IS_SLOT>(As, src1, group_slots, mb, mcnt, k0, K, top_k, lid); \
        LOAD_W(Ws, W, bc, k0, N, K, nb01, lid);                               \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
        for (uint k = 0; k < BK / 8; ++k) {                                    \
            simdgroup_half8x8 a[MR];                                           \
            for (uint r = 0; r < MR; ++r)                                      \
                simdgroup_load(a[r], As + (r * 8) * BK + k * 8, BK);           \
            for (uint c = 0; c < MC; ++c) {                                    \
                simdgroup_half8x8 b;                                           \
                simdgroup_load(b, Ws + (k * 8) * BN + c0 + c * 8, BN);         \
                for (uint r = 0; r < MR; ++r)                                  \
                    simdgroup_multiply_accumulate(acc[r][c], a[r], b, acc[r][c]); \
            }                                                                  \
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
    }                                                                          \
    threadgroup float Cs[BM * BN];                                             \
    for (uint r = 0; r < MR; ++r)                                              \
        for (uint c = 0; c < MC; ++c)                                          \
            simdgroup_store(acc[r][c], Cs + (r * 8) * BN + c0 + c * 8, BN);    \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    for (uint i = lid; i < BM * BN; i += 64) {                                 \
        const uint r = i / BN, cc = i % BN;                                    \
        const uint gc = bc + cc;                                               \
        if (r < mcnt && gc < N) {                                              \
            const int slot = group_slots[mb + r];                              \
            dst[(size_t)slot * N + gc] = Cs[i];                                \
        }                                                                      \
    }

[[host_name("deepseek_moe_mma_q4k_grp")]]
[[kernel]]
void deepseek_moe_mma_q4k_grp(
    device const uchar*    src0        [[buffer(0)]],   // expert weight slabs
    device const float*    src1        [[buffer(1)]],   // moe_x_f32 [T, K]
    device       float*    dst         [[buffer(2)]],   // [T*top_k, N] (slot-indexed)
    device const int32_t*  group_off   [[buffer(3)]],
    device const int32_t*  group_slots [[buffer(4)]],
    constant uint&         K           [[buffer(5)]],   // in_dim
    constant uint&         N           [[buffer(6)]],   // out_rows
    constant uint&         top_k       [[buffer(7)]],
    constant ulong&        nb01        [[buffer(8)]],   // weight row stride
    constant ulong&        nb02        [[buffer(9)]],   // weight expert(slab) stride
    uint3 gid  [[threadgroup_position_in_grid]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    MOE_MMA_BODY(load_W_q4k_grp, false)   // gate/up: A row = token
}

[[host_name("deepseek_moe_mma_q5_0_grp")]]
[[kernel]]
void deepseek_moe_mma_q5_0_grp(
    device const uchar*    src0        [[buffer(0)]],
    device const float*    src1        [[buffer(1)]],   // moe_mid_f32 [T*top_k, K] (slot-indexed)
    device       float*    dst         [[buffer(2)]],   // [T*top_k, N] (slot-indexed)
    device const int32_t*  group_off   [[buffer(3)]],
    device const int32_t*  group_slots [[buffer(4)]],
    constant uint&         K           [[buffer(5)]],
    constant uint&         N           [[buffer(6)]],
    constant uint&         top_k       [[buffer(7)]],
    constant ulong&        nb01        [[buffer(8)]],
    constant ulong&        nb02        [[buffer(9)]],
    uint3 gid  [[threadgroup_position_in_grid]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    MOE_MMA_BODY(load_W_q5_0_grp, true)   // down: A row = slot (moe_mid is slot-indexed)
}
