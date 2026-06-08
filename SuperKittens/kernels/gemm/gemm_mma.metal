// gemm_mma.metal — batched (seq>1) weight × activation GEMM via simdgroup MMA.
//
// SK decode is a pure M=1 engine: every projection is a GEMV dispatched per row,
// so a seq=K forward re-reads each weight K times (cost ~linear in K). This
// kernel amortizes one weight read across all M rows of the activation by
// staging a [BK][BN] weight tile in threadgroup memory and reusing it across the
// BM-row MMA, so a seq=K forward approaches the cost of seq=1. Used by the qwen
// prefill path (T>1); the M=1 matvec stays the decode fast path.
//
// Math: C[m,n] = sum_k A[m,k] * W[n,k]  (i.e. A @ W^T), matching the
// row-major [N,K] weight layout the q8_0/q4k/q6k matvecs already use. The
// dequant store transposes W into Ws[BK][BN] so the MMA is a plain
// A[BM][BK] @ Ws[BK][BN].
//
// Five weight dtypes share one tiling body via a per-dtype tile-loader:
//   gemm_mma_f16  — fp16   weight [N,K]
//   gemm_mma_bf16 — bf16   weight [N,K]  (widened to fp16 on tile load)
//   gemm_mma_q8_0 — Q8_0   weight [N,K]  (34 B / 32 weights)
//   gemm_mma_q4k  — Q4_K   weight [N,K]  (144 B / 256 weights)
//   gemm_mma_q6k  — Q6_K   weight [N,K]  (210 B / 256 weights)
//
// Bindings (encode_gemm_mma):
//   0: A activation, fp16 [M, K]   (row stride ldA = K)
//   1: W weights, quant/fp16 row-major [N, K]
//   2: C output, fp16 [M, N]       (row stride ldC, defaults to N)
//   3: uint M
//   4: uint N
//   5: uint K
//   6: uint ldC
//
// Dispatch (host): grid (ceil(N/BN), ceil(M/BM), 1), threadgroup (32*NSG,1,1).
//   BM=8 rows, BN=32 cols, BK=32. NSG=2 simdgroups; each owns BN/NSG=16 output
//   columns (MC=2 8-wide tiles) and all BM=8 rows (MR=1).

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#ifndef QK_K
#define QK_K 256
#endif

namespace skmma {

enum : uint { BM = 8, BN = 32, BK = 32, NSG = 2, MR = 1, MC = 2 };

struct __attribute__((packed)) q8_block { half d; int8_t qs[32]; };

struct block_q4_K {
    half  d;
    half  dmin;
    uchar scales[12];
    uchar qs[QK_K / 2];
};

struct __attribute__((packed)) block_q6_K {
    uchar  ql[QK_K / 2];      // 128 — low 4 bits
    uchar  qh[QK_K / 4];      // 64  — high 2 bits
    int8_t scales[QK_K / 16]; // 16  — per-16-element sub-scale
    half   d;                 // super-block scale
};

// Cooperatively load A[br:br+BM][k0:k0+BK] into As[BM][BK] (row-major, padded
// rows zeroed). 64 threads, half-vectorized when K is 8-aligned at this offset.
inline void load_A(threadgroup half* As,
                   device const half* A,
                   uint br, uint k0, uint M, uint K, uint ldA,
                   uint lid)
{
    const uint n = BM * BK;            // 256 elems
    for (uint i = lid; i < n / 4; i += 64) {
        const uint r = (i * 4) / BK;
        const uint c = (i * 4) % BK;
        const uint gr = br + r, gc = k0 + c;
        half4 v = (gr < M && gc + 3 < K)
                  ? *reinterpret_cast<const device half4*>(A + (size_t)gr * ldA + gc)
                  : half4(0);
        *reinterpret_cast<threadgroup half4*>(As + i * 4) = v;
    }
}

} // namespace skmma

using namespace skmma;

// ── fp16 weight tile loader: W[bc:bc+BN][k0:k0+BK] → Ws[BK][BN] (transposed).
inline void load_W_f16(threadgroup half* Ws,
                       device const half* W,
                       uint bc, uint k0, uint N, uint K,
                       uint lid)
{
    const uint n = BN * BK;
    for (uint i = lid; i < n; i += 64) {
        const uint wr = i / BK;        // output column within tile (0..BN)
        const uint wk = i % BK;        // K within tile (0..BK)
        const uint gn = bc + wr, gk = k0 + wk;
        half v = (gn < N && gk < K) ? W[(size_t)gn * K + gk] : half(0);
        Ws[wk * BN + wr] = v;          // transpose: [BK][BN]
    }
}

// ── bf16 weight tile loader: W[bc:bc+BN][k0:k0+BK] → Ws[BK][BN] (transposed).
// Weight is bf16 [N,K]; the MMA runs on fp16 so each element is widened on load.
inline void load_W_bf16(threadgroup half* Ws,
                        device const bfloat* W,
                        uint bc, uint k0, uint N, uint K,
                        uint lid)
{
    const uint n = BN * BK;
    for (uint i = lid; i < n; i += 64) {
        const uint wr = i / BK;        // output column within tile (0..BN)
        const uint wk = i % BK;        // K within tile (0..BK)
        const uint gn = bc + wr, gk = k0 + wk;
        half v = (gn < N && gk < K) ? (half)(float)W[(size_t)gn * K + gk] : half(0);
        Ws[wk * BN + wr] = v;          // transpose: [BK][BN]
    }
}

// ── Q8_0 weight tile loader. block = 32 weights = one BK row of one output row.
// Each of the BN output rows contributes its [k0:k0+BK] block.
inline void load_W_q8_0(threadgroup half* Ws,
                        device const uchar* W,
                        uint bc, uint k0, uint N, uint K,
                        uint lid)
{
    const uint nb = K / 32;            // blocks per output row
    const uint kb = k0 / 32;           // which block this BK slice is
    // Strided: 64 threads cover BN*BK=1024 tile elems (16 each). Coalesced
    // device reads; the per-element blk->d reload is cheap vs the MMA.
    for (uint i = lid; i < BN * BK; i += 64) {
        const uint wr = i / BK;        // output column within tile
        const uint wk = i % BK;
        const uint gn = bc + wr;
        half v = half(0);
        if (gn < N) {
            device const q8_block* blk =
                (device const q8_block*)(W + ((size_t)gn * nb + kb) * sizeof(q8_block));
            v = (half)((float)blk->qs[wk] * (float)blk->d);
        }
        Ws[wk * BN + wr] = v;
    }
}

// ── Q4_K weight tile loader. One super-block = 256 weights spans 8 BK=32 slices.
// Dequant: w = d*sc*q - dmin*m, with sc/m the 6-bit sub-scale/min for the
// 32-element sub-block (8 sub-blocks per super-block, indexed by k0/32 mod 8).
inline void load_W_q4k(threadgroup half* Ws,
                       device const uchar* W,
                       uint bc, uint k0, uint N, uint K,
                       uint lid)
{
    const uint nsb = K / QK_K;         // super-blocks per output row
    const uint sb  = k0 / QK_K;        // super-block index
    const uint sub = (k0 / 32) % 8;    // sub-block (0..7) within the super-block
    // One thread per output column (wr in 0..BN). The 6-bit sub-scale decode
    // (real ALU: cross-byte shifts/masks) is constant across all 32 wk of one
    // BK=32 sub-block, so it is hoisted out of the wk loop and run ONCE — the
    // pre-fix layout split each column across two threads (wk halves), decoding
    // the same scale twice. The idle threads (lid≥BN) are cheap vs that re-unpack.
    const uint wr = lid;                 // 0..63; only 0..BN-1 do work
    if (wr >= BN) return;
    const uint gn = bc + wr;
    if (gn < N) {
        device const block_q4_K* blk =
            (device const block_q4_K*)(W + ((size_t)gn * nsb + sb) * sizeof(block_q4_K));
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
        // qs: sub-block `sub` is low/high nibble of bytes [32*(sub/2) .. +32].
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

// ── Q6_K weight tile loader. One super-block = 256 weights = QK_K, spanning 8
// BK=32 slices. The element→byte map is the GGML interleave (4 sub-positions ×
// 2 halves per 128, see q6k_matvec.metal); each thread decodes its element's
// ql/qh nibble + 2 high bits and per-16 sub-scale. Element-parallel like q8_0
// (the per-element d/scale reload is cheap vs the MMA), so partial tiles and the
// non-contiguous K mapping stay correct without a column-parallel scale hoist.
inline void load_W_q6k(threadgroup half* Ws,
                       device const uchar* W,
                       uint bc, uint k0, uint N, uint K,
                       uint lid)
{
    const uint nsb = K / QK_K;         // super-blocks per output row
    const uint sb  = k0 / QK_K;        // super-block index
    const uint e0  = k0 % QK_K;        // first element of this BK slice in the block
    for (uint i = lid; i < BN * BK; i += 64) {
        const uint wr = i / BK;        // output column within tile
        const uint wk = i % BK;        // K within tile
        const uint gn = bc + wr;
        half v = half(0);
        if (gn < N) {
            device const block_q6_K* blk =
                (device const block_q6_K*)(W + ((size_t)gn * nsb + sb) * sizeof(block_q6_K));
            const uint e  = e0 + wk;            // 0..255 element within super-block
            const uint h  = e >> 7;             // half (0/1)
            const uint eh = e & 127;            // 0..127 within the half
            const uint sp = eh >> 5;            // sub-position 0..3
            const uint l  = eh & 31;            // 0..31 within sub-position
            const uint ql_base = 64u * h;
            const uint qh_base = 32u * h;
            const int  sc_base = (int)(8u * h);
            uint  ql_idx; uint qsh; int sc_idx; bool hinib;
            if (sp == 0)      { ql_idx = ql_base + l;      qsh = 0; sc_idx = sc_base + 0 + (int)(l >> 4); hinib = false; }
            else if (sp == 1) { ql_idx = ql_base + l + 32; qsh = 2; sc_idx = sc_base + 2 + (int)(l >> 4); hinib = false; }
            else if (sp == 2) { ql_idx = ql_base + l;      qsh = 4; sc_idx = sc_base + 4 + (int)(l >> 4); hinib = true;  }
            else              { ql_idx = ql_base + l + 32; qsh = 6; sc_idx = sc_base + 6 + (int)(l >> 4); hinib = true;  }
            const uint8_t lo = blk->ql[ql_idx];
            const uint8_t hb = blk->qh[qh_base + l];
            const int qval = (int)((hinib ? (lo >> 4) : (lo & 0x0F))
                                   | (((hb >> qsh) & 3) << 4)) - 32;
            v = (half)((float)blk->d * (float)blk->scales[sc_idx] * (float)qval);
        }
        Ws[wk * BN + wr] = v;
    }
}

#define GEMM_MMA_BODY(LOAD_W)                                                  \
    const uint br = gid.y * BM;                                                \
    const uint bc = gid.x * BN;                                                \
    threadgroup half As[BM * BK];                                              \
    threadgroup half Ws[BK * BN];                                              \
    const uint lid = simd * 32 + lane;                                         \
    const uint c0  = simd * MC * 8;        /* this SG's first output column */ \
    simdgroup_float8x8 acc[MR][MC] = {};                                       \
    for (uint k0 = 0; k0 < K; k0 += BK) {                                      \
        load_A(As, A, br, k0, M, K, K, lid);                                   \
        LOAD_W(Ws, W, bc, k0, N, K, lid);                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
        for (uint k = 0; k < BK / 8; ++k) {                                    \
            simdgroup_half8x8 a;                                               \
            simdgroup_load(a, As + (0 * 8) * BK + k * 8, BK);                  \
            for (uint c = 0; c < MC; ++c) {                                    \
                simdgroup_half8x8 b;                                           \
                simdgroup_load(b, Ws + (k * 8) * BN + c0 + c * 8, BN);         \
                simdgroup_multiply_accumulate(acc[0][c], a, b, acc[0][c]);     \
            }                                                                  \
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
    }                                                                          \
    threadgroup float Cs[BM * BN];                                             \
    for (uint c = 0; c < MC; ++c) {                                            \
        simdgroup_store(acc[0][c], Cs + c0 + c * 8, BN);                       \
    }                                                                          \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    for (uint i = lid; i < BM * BN; i += 64) {                                 \
        const uint r = i / BN, cc = i % BN;                                    \
        const uint gr = br + r, gc = bc + cc;                                  \
        if (gr < M && gc < N) C[(size_t)gr * ldC + gc] = (half)Cs[i];          \
    }

[[host_name("gemm_mma_f16")]]
[[kernel]]
void gemm_mma_f16(
    device const half*  A   [[buffer(0)]],
    device const half*  W   [[buffer(1)]],
    device half*        C   [[buffer(2)]],
    constant uint&      M   [[buffer(3)]],
    constant uint&      N   [[buffer(4)]],
    constant uint&      K   [[buffer(5)]],
    constant uint&      ldC [[buffer(6)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    GEMM_MMA_BODY(load_W_f16)
}

[[host_name("gemm_mma_q8_0")]]
[[kernel]]
void gemm_mma_q8_0(
    device const half*  A   [[buffer(0)]],
    device const uchar* W   [[buffer(1)]],
    device half*        C   [[buffer(2)]],
    constant uint&      M   [[buffer(3)]],
    constant uint&      N   [[buffer(4)]],
    constant uint&      K   [[buffer(5)]],
    constant uint&      ldC [[buffer(6)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    GEMM_MMA_BODY(load_W_q8_0)
}

[[host_name("gemm_mma_q4k")]]
[[kernel]]
void gemm_mma_q4k(
    device const half*  A   [[buffer(0)]],
    device const uchar* W   [[buffer(1)]],
    device half*        C   [[buffer(2)]],
    constant uint&      M   [[buffer(3)]],
    constant uint&      N   [[buffer(4)]],
    constant uint&      K   [[buffer(5)]],
    constant uint&      ldC [[buffer(6)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    GEMM_MMA_BODY(load_W_q4k)
}

[[host_name("gemm_mma_q6k")]]
[[kernel]]
void gemm_mma_q6k(
    device const half*  A   [[buffer(0)]],
    device const uchar* W   [[buffer(1)]],
    device half*        C   [[buffer(2)]],
    constant uint&      M   [[buffer(3)]],
    constant uint&      N   [[buffer(4)]],
    constant uint&      K   [[buffer(5)]],
    constant uint&      ldC [[buffer(6)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    GEMM_MMA_BODY(load_W_q6k)
}

[[host_name("gemm_mma_bf16")]]
[[kernel]]
void gemm_mma_bf16(
    device const half*   A   [[buffer(0)]],
    device const bfloat* W   [[buffer(1)]],
    device half*         C   [[buffer(2)]],
    constant uint&       M   [[buffer(3)]],
    constant uint&       N   [[buffer(4)]],
    constant uint&       K   [[buffer(5)]],
    constant uint&       ldC [[buffer(6)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    GEMM_MMA_BODY(load_W_bf16)
}
