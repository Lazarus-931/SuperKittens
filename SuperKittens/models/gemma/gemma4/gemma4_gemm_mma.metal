// gemma4_gemm_mma.metal — bf16-I/O batched (T>1 prefill) GEMM for the gemma4 body.
//
// WHY: the gemma4 quantized body (SK_GEMMA4_LMHEAD_Q8 / Q4_K body) routes every
// projection through enc_q8/enc_kquant_matvec, which loops the M=1 matvec once
// per row at prefill — each row re-reads the full [N,K] weight, so prefill cost
// is O(M * weight_bytes). This mirrors kernels/gemm/gemm_mma.metal (BM=32 weight
// amortization, bit-exact reduction order) but with bf16 activations/outputs to
// match the gemma4 pipeline (the shared gemm_mma_* take half A/C and would
// misread bf16 bits). Weight loaders + MMA body are copied verbatim from the
// shared kernel (same K-reduction order => numerically equivalent on the W side).
//
// Bindings (mirror gemm_mma_*): 0 A bf16[M,K], 1 W quant[N,K], 2 C bf16[M,N],
//   3 uint M, 4 uint N, 5 uint K, 6 uint ldC.
// Dispatch: grid (ceil(N/BN), ceil(M/BM), 1), threadgroup (32*NSG,1,1).

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#ifndef QK_K
#define QK_K 256
#endif

namespace skg4mma {

enum : uint { BM = 32, BN = 32, BK = 32, NSG = 2, MR = 4, MC = 2 };

struct __attribute__((packed)) q8_block { half d; int8_t qs[32]; };

struct block_q4_K {
    half  d;
    half  dmin;
    uchar scales[12];
    uchar qs[QK_K / 2];
};

struct __attribute__((packed)) block_q6_K {
    uchar  ql[QK_K / 2];
    uchar  qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    half   d;
};

// bf16 activation tile loader: A[br:br+BM][k0:k0+BK] -> As[BM][BK] (row-major).
inline void load_A(threadgroup half* As,
                   device const bfloat* A,
                   uint br, uint k0, uint M, uint K, uint ldA,
                   uint lid)
{
    const uint n = BM * BK;
    for (uint i = lid; i < n; i += 64) {
        const uint r = i / BK;
        const uint c = i % BK;
        const uint gr = br + r, gc = k0 + c;
        As[i] = (gr < M && gc < K) ? (half)(float)A[(size_t)gr * ldA + gc] : half(0);
    }
}

inline void load_W_q8_0(threadgroup half* Ws,
                        device const uchar* W,
                        uint bc, uint k0, uint N, uint K,
                        uint lid)
{
    const uint nb = K / 32;
    const uint kb = k0 / 32;
    for (uint i = lid; i < BN * BK; i += 64) {
        const uint wr = i / BK;
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

inline void load_W_q4k(threadgroup half* Ws,
                       device const uchar* W,
                       uint bc, uint k0, uint N, uint K,
                       uint lid)
{
    const uint nsb = K / QK_K;
    const uint sb  = k0 / QK_K;
    const uint sub = (k0 / 32) % 8;
    const uint wr = lid;
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

inline void load_W_q6k(threadgroup half* Ws,
                       device const uchar* W,
                       uint bc, uint k0, uint N, uint K,
                       uint lid)
{
    const uint nsb = K / QK_K;
    const uint sb  = k0 / QK_K;
    const uint e0  = k0 % QK_K;
    for (uint i = lid; i < BN * BK; i += 64) {
        const uint wr = i / BK;
        const uint wk = i % BK;
        const uint gn = bc + wr;
        half v = half(0);
        if (gn < N) {
            device const block_q6_K* blk =
                (device const block_q6_K*)(W + ((size_t)gn * nsb + sb) * sizeof(block_q6_K));
            const uint e  = e0 + wk;
            const uint h  = e >> 7;
            const uint eh = e & 127;
            const uint sp = eh >> 5;
            const uint l  = eh & 31;
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

} // namespace skg4mma

using namespace skg4mma;

#define G4_GEMM_MMA_BODY(LOAD_W)                                               \
    const uint br = gid.y * BM;                                                \
    const uint bc = gid.x * BN;                                                \
    threadgroup half As[BM * BK];                                              \
    threadgroup half Ws[BK * BN];                                              \
    const uint lid = simd * 32 + lane;                                         \
    const uint c0  = simd * MC * 8;                                            \
    simdgroup_float8x8 acc[MR][MC] = {};                                       \
    for (uint k0 = 0; k0 < K; k0 += BK) {                                      \
        load_A(As, A, br, k0, M, K, K, lid);                                   \
        LOAD_W(Ws, W, bc, k0, N, K, lid);                                      \
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
    threadgroup_barrier(mem_flags::mem_threadgroup);                           \
    for (uint i = lid; i < BM * BN; i += 64) {                                 \
        const uint r = i / BN, cc = i % BN;                                    \
        const uint gr = br + r, gc = bc + cc;                                  \
        if (gr < M && gc < N) C[(size_t)gr * ldC + gc] = (bfloat)Cs[i];        \
    }

[[host_name("gemma4_gemm_mma_q8_bf16")]]
[[kernel]]
void gemma4_gemm_mma_q8_bf16(
    device const bfloat* A   [[buffer(0)]],
    device const uchar*  W   [[buffer(1)]],
    device bfloat*       C   [[buffer(2)]],
    constant uint&       M   [[buffer(3)]],
    constant uint&       N   [[buffer(4)]],
    constant uint&       K   [[buffer(5)]],
    constant uint&       ldC [[buffer(6)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    G4_GEMM_MMA_BODY(load_W_q8_0)
}

[[host_name("gemma4_gemm_mma_q4k_bf16")]]
[[kernel]]
void gemma4_gemm_mma_q4k_bf16(
    device const bfloat* A   [[buffer(0)]],
    device const uchar*  W   [[buffer(1)]],
    device bfloat*       C   [[buffer(2)]],
    constant uint&       M   [[buffer(3)]],
    constant uint&       N   [[buffer(4)]],
    constant uint&       K   [[buffer(5)]],
    constant uint&       ldC [[buffer(6)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    G4_GEMM_MMA_BODY(load_W_q4k)
}

[[host_name("gemma4_gemm_mma_q6k_bf16")]]
[[kernel]]
void gemma4_gemm_mma_q6k_bf16(
    device const bfloat* A   [[buffer(0)]],
    device const uchar*  W   [[buffer(1)]],
    device bfloat*       C   [[buffer(2)]],
    constant uint&       M   [[buffer(3)]],
    constant uint&       N   [[buffer(4)]],
    constant uint&       K   [[buffer(5)]],
    constant uint&       ldC [[buffer(6)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    G4_GEMM_MMA_BODY(load_W_q6k)
}
