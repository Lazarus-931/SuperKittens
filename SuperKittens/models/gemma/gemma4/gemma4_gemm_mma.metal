// gemma4_gemm_mma.metal — bf16-I/O batched (T>1 prefill) GEMM for the gemma4 body.
//
// WHY: the gemma4 quantized body (SK_GEMMA4_LMHEAD_Q8 / Q4_K body) routes every
// projection through enc_q8/enc_kquant_matvec, which loops the M=1 matvec once
// per row at prefill — each row re-reads the full [N,K] weight, so prefill cost
// is O(M * weight_bytes). This mirrors kernels/gemm/gemm_mma.metal (weight
// amortization, bit-exact reduction order) but with bf16 activations/outputs to
// match the gemma4 pipeline (the shared gemm_mma_* take half A/C and would
// misread bf16 bits). Weight loaders + MMA body are copied verbatim from the
// shared kernel (same K-reduction order => numerically equivalent on the W side).
//
// Two tile configs coexist so the launcher can A/B one against the other in a
// single loaded process (thermal-controlled, env SK_GEMMA4_MMA_T64):
//   _t32 : BM=32 BN=32 NSG=2 — the original config (default, unchanged).
//   _t64 : BM=64 BN=64 NSG=4 — W=[N,K] is the prefill bandwidth bottleneck and
//          each BM-row M-tile re-reads all of W, so BM 32->64 halves total
//          weight bytes read (seq=512: 16->8 W passes); NSG 2->4 raises occupancy
//          and hides the device->shared load latency. Column-split layout
//          preserved (each simdgroup owns MC cols, all MR rows) so the per-k
//          simdgroup_multiply_accumulate reduction order is byte-identical to the
//          BM=32 kernel — coherence-neutral. BM=MR*8, BN=NSG*MC*8, NT=NSG*32.
//
// Bindings (mirror gemm_mma_*): 0 A bf16[M,K], 1 W quant[N,K], 2 C bf16[M,N],
//   3 uint M, 4 uint N, 5 uint K, 6 uint ldC.
// Dispatch: grid (ceil(N/BN), ceil(M/BM), 1), threadgroup (NT,1,1).

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#ifndef QK_K
#define QK_K 256
#endif

namespace skg4mma {

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
template <uint BM, uint BN, uint BK, uint NT>
inline void load_A(threadgroup half* As,
                   device const bfloat* A,
                   uint br, uint k0, uint M, uint K, uint ldA,
                   uint lid)
{
    const uint n = BM * BK;
    for (uint i = lid; i < n; i += NT) {
        const uint r = i / BK;
        const uint c = i % BK;
        const uint gr = br + r, gc = k0 + c;
        As[i] = (gr < M && gc < K) ? (half)(float)A[(size_t)gr * ldA + gc] : half(0);
    }
}

template <uint BM, uint BN, uint BK, uint NT>
inline void load_W_q8_0(threadgroup half* Ws,
                        device const uchar* W,
                        uint bc, uint k0, uint N, uint K,
                        uint lid)
{
    const uint nb = K / 32;
    const uint kb = k0 / 32;
    for (uint i = lid; i < BN * BK; i += NT) {
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

template <uint BM, uint BN, uint BK, uint NT>
inline void load_W_q4k(threadgroup half* Ws,
                       device const uchar* W,
                       uint bc, uint k0, uint N, uint K,
                       uint lid)
{
    const uint nsb = K / QK_K;
    const uint sb  = k0 / QK_K;
    const uint sub = (k0 / 32) % 8;
    // One thread per output column (Ws row). NT may exceed BN — extra lanes idle.
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

template <uint BM, uint BN, uint BK, uint NT>
inline void load_W_q6k(threadgroup half* Ws,
                       device const uchar* W,
                       uint bc, uint k0, uint N, uint K,
                       uint lid)
{
    const uint nsb = K / QK_K;
    const uint sb  = k0 / QK_K;
    const uint e0  = k0 % QK_K;
    for (uint i = lid; i < BN * BK; i += NT) {
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

// Tiled bf16-I/O GEMM. LOADW is the dequant tile loader (q8/q4k/q6k). The
// column-split MMA (each simdgroup owns MC cols-of-8, all MR rows-of-8) and the
// per-k accumulate order are identical across configs => numerically equivalent.
// Metal forbids threadgroup-address-space locals in a non-kernel function, so
// As/Ws/Cs are declared in each [[kernel]] entry and passed in by pointer.
// CT is the Cs (epilogue staging) element type; all current configs use float so
// the per-k accumulate AND the float-acc -> Cs -> bf16 store are bit-identical
// across tile shapes (only which acc tiles exist changes).
template <uint BM, uint BN, uint BK, uint NSG, uint MR, uint MC, typename CT,
          void (*LOADW)(threadgroup half*, device const uchar*, uint, uint, uint, uint, uint)>
inline void gemm_body(device const bfloat* A, device const uchar* W, device bfloat* C,
                      uint M, uint N, uint K, uint ldC,
                      uint2 gid, uint simd, uint lane,
                      threadgroup half* As, threadgroup half* Ws, threadgroup CT* Cs)
{
    constexpr uint NT = NSG * 32;
    const uint br = gid.y * BM;
    const uint bc = gid.x * BN;
    const uint lid = simd * 32 + lane;
    const uint c0  = simd * MC * 8;
    simdgroup_float8x8 acc[MR][MC] = {};
    for (uint k0 = 0; k0 < K; k0 += BK) {
        load_A<BM, BN, BK, NT>(As, A, br, k0, M, K, K, lid);
        LOADW(Ws, W, bc, k0, N, K, lid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < BK / 8; ++k) {
            simdgroup_half8x8 a[MR];
            for (uint r = 0; r < MR; ++r)
                simdgroup_load(a[r], As + (r * 8) * BK + k * 8, BK);
            for (uint c = 0; c < MC; ++c) {
                simdgroup_half8x8 b;
                simdgroup_load(b, Ws + (k * 8) * BN + c0 + c * 8, BN);
                for (uint r = 0; r < MR; ++r)
                    simdgroup_multiply_accumulate(acc[r][c], a[r], b, acc[r][c]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r = 0; r < MR; ++r)
        for (uint c = 0; c < MC; ++c)
            simdgroup_store(acc[r][c], Cs + (r * 8) * BN + c0 + c * 8, BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lid; i < BM * BN; i += NT) {
        const uint r = i / BN, cc = i % BN;
        const uint gr = br + r, gc = bc + cc;
        if (gr < M && gc < N) C[(size_t)gr * ldC + gc] = (bfloat)Cs[i];
    }
}

} // namespace skg4mma

using namespace skg4mma;

// TAG: bare-token C++ name suffix; HN: host_name string suffix; QSUF/LOADW:
// quant tag + dequant loader; CT: Cs element type (float=byte-identical default,
// half=smaller epilogue buffer); BM/BN/BK/NSG/MR/MC: tile config (BM=MR*8,
// BN=NSG*MC*8). As/Ws/Cs live here (threadgroup locals are illegal in helpers).
#define G4_GEMM_KERNEL(QSUF, LOADW, TAG, HN, CT, BM, BN, BK, NSG, MR, MC)      \
[[host_name("gemma4_gemm_mma_" #QSUF "_bf16" HN)]] [[kernel]]                  \
void gemma4_gemm_mma_##QSUF##_bf16_##TAG(                                      \
    device const bfloat* A [[buffer(0)]], device const uchar* W [[buffer(1)]], \
    device bfloat* C [[buffer(2)]], constant uint& M [[buffer(3)]],            \
    constant uint& N [[buffer(4)]], constant uint& K [[buffer(5)]],            \
    constant uint& ldC [[buffer(6)]], uint2 gid [[threadgroup_position_in_grid]], \
    uint simd [[simdgroup_index_in_threadgroup]],                             \
    uint lane [[thread_index_in_simdgroup]]) {                                 \
    threadgroup half As[BM * BK];                                             \
    threadgroup half Ws[BK * BN];                                             \
    threadgroup CT   Cs[BM * BN];                                             \
    gemm_body<BM, BN, BK, NSG, MR, MC, CT, LOADW<BM, BN, BK, NSG*32>>(         \
        A, W, C, M, N, K, ldC, gid, simd, lane, As, Ws, Cs); }

#define G4_GEMM_FAMILY(TAG, HN, CT, BM, BN, BK, NSG, MR, MC)                   \
    G4_GEMM_KERNEL(q8, load_W_q8_0, TAG, HN, CT, BM, BN, BK, NSG, MR, MC)      \
    G4_GEMM_KERNEL(q4k, load_W_q4k, TAG, HN, CT, BM, BN, BK, NSG, MR, MC)      \
    G4_GEMM_KERNEL(q6k, load_W_q6k, TAG, HN, CT, BM, BN, BK, NSG, MR, MC)

// Original config (kept for rollback via SK_GEMMA4_MMA_T64=0): BM=32 BN=32 BK=32
// NSG=2 MR=4 MC=2. host_names unchanged so the metallib path resolves identically.
G4_GEMM_FAMILY(t32, "", float, 32, 32, 32, 2, 4, 2)
// Default prefill config: BM=64 (halves W re-reads) BN=32 BK=32 NSG=4 MR=8 MC=1.
// 14 KB tgmem -> ~2 threadgroups/core for latency-hiding at large M. Measured
// E4B prefill TTFT +14-18% over the BM=32 kernel; coherence/decode byte-identical.
G4_GEMM_FAMILY(t64n, "_t64n", float, 64, 32, 32, 4, 8, 1)
