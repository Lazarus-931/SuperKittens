//
//  gated_mlp_gelu.metal — fused GeLU-gated MLP (GeGLU)
//
//  gate = x @ w_gate, up = x @ w_up
//  out  = (GeLU(gate) * up) @ w_down
//
//  Used by Gemma 2, Gemma 3, Gemma 4. Tile geometry:
//      BM × BN output per TG, BM × BN intermediate (BN-wide chunk of N_int)
//      OUTER LOOP over N_int in BN-chunks accumulating into acc_out.
//
//  GeLU form: tanh approximation, matching HuggingFace and PyTorch's
//  approximate=True. Closed form:
//      gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
//  IMPORTANT: this kernel was reworked on 2026-05-10 to fix a correctness
//  bug at N_int > BN (intermediate-overflow + missing chunk loop). The
//  earlier version computed only one BN-wide slice of intermediate and
//  reused it across all GEMM 3 K-iterations, producing wrong output for
//  any N_int > BN, plus producing NaN/garbage at small M with non-zero
//  buffer offsets. The current structure: full chunk loop over N_int,
//  intermediate sized BM × BN per chunk, GEMM 3 reads with proper bk
//  offset within the chunk.
//
//  Tile constants: BM=BN=64, BK=32, MR=2 MC=8, 128 threads (4 simdgroups).
//  Threadgroup memory: As + Bs + intermediate = 8 KB + 8 KB + 8 KB = 24 KB.
//

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

enum : uint { BM = 64, BN = 64, BK = 32, MR = 2, MC = 8, THREADS = 128 };

// GeLU constants (tanh approximation).
constant constexpr float GELU_K0 = 0.7978845608028654f;   // sqrt(2 / pi)
constant constexpr float GELU_K1 = 0.044715f;

inline float gelu_approx(float x) {
    float x3 = x * x * x;
    float u  = GELU_K0 * (x + GELU_K1 * x3);
    return 0.5f * x * (1.0f + metal::precise::tanh(u));
}

[[host_name("gated_mlp_bf16")]]
[[kernel, max_total_threads_per_threadgroup(THREADS)]]
void gated_mlp_bf16(
    device const bfloat* x,          // (M, K)
    device const bfloat* w_gate,     // (K, N_int)
    device const bfloat* w_up,       // (K, N_int)
    device const bfloat* w_down,     // (N_int, N)
    device bfloat* out,              // (M, N)
    constant uint& M, constant uint& N, constant uint& K, constant uint& N_int,
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    const uint row = gid.y * BM + simd * 16;
    const uint col = gid.x * BN;
    const uint tid = simd * 32 + lane;

    const device bfloat4* x4 = reinterpret_cast<const device bfloat4*>(x);
    const device bfloat4* g4 = reinterpret_cast<const device bfloat4*>(w_gate);
    const device bfloat4* u4 = reinterpret_cast<const device bfloat4*>(w_up);
    const device bfloat4* d4 = reinterpret_cast<const device bfloat4*>(w_down);

    threadgroup bfloat As[BM * BK];          // 4 KB — reused across passes
    threadgroup bfloat Bs[BK * BN];          // 4 KB — reused across passes
    threadgroup bfloat intermediate[BM * BN]; // 8 KB — one BN-chunk of GeGLU output

    simdgroup_float8x8 acc_out[MR][MC] = {};

    // ── OUTER LOOP: chunk N_int into BN slices ──
    // For each chunk, recompute the gate/up/intermediate tile and accumulate
    // its GEMM-3 contribution. acc_out persists across chunks.
    for (uint nint_chunk = 0; nint_chunk < N_int; nint_chunk += BN) {

        // ── GEMM 1: acc_gate (BM×BN) = x @ w_gate[:, nint_chunk:+BN] ──
        simdgroup_float8x8 acc_gate[MR][MC] = {};
        for (uint bk = 0; bk < K; bk += BK) {
            for (uint i = tid; i < (BM * BK) / 4; i += THREADS) {
                uint r = (i * 4) / BK, c = (i * 4) % BK;
                uint gr = row + r, gc = bk + c;
                reinterpret_cast<threadgroup bfloat4*>(As)[i] =
                    (gr < M && gc < K) ? x4[(gr * K + gc) / 4] : bfloat4(0);
            }
            for (uint i = tid; i < (BK * BN) / 4; i += THREADS) {
                uint r = (i * 4) / BN, c = (i * 4) % BN;
                uint gr = bk + r, gc = nint_chunk + c;
                reinterpret_cast<threadgroup bfloat4*>(Bs)[i] =
                    (gr < K && gc < N_int) ? g4[(gr * N_int + gc) / 4] : bfloat4(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = 0; k < BK / 8; k++)
                for (uint r = 0; r < MR; r++) {
                    simdgroup_bfloat8x8 a;
                    simdgroup_load(a, As + (simd * 16 + r * 8) * BK + k * 8, BK);
                    for (uint c = 0; c < MC; c++) {
                        simdgroup_bfloat8x8 b;
                        simdgroup_load(b, Bs + k * 8 * BN + c * 8, BN);
                        simdgroup_multiply_accumulate(acc_gate[r][c], a, b, acc_gate[r][c]);
                    }
                }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ── GEMM 2: acc_up (BM×BN) = x @ w_up[:, nint_chunk:+BN] ──
        simdgroup_float8x8 acc_up[MR][MC] = {};
        for (uint bk = 0; bk < K; bk += BK) {
            for (uint i = tid; i < (BM * BK) / 4; i += THREADS) {
                uint r = (i * 4) / BK, c = (i * 4) % BK;
                uint gr = row + r, gc = bk + c;
                reinterpret_cast<threadgroup bfloat4*>(As)[i] =
                    (gr < M && gc < K) ? x4[(gr * K + gc) / 4] : bfloat4(0);
            }
            for (uint i = tid; i < (BK * BN) / 4; i += THREADS) {
                uint r = (i * 4) / BN, c = (i * 4) % BN;
                uint gr = bk + r, gc = nint_chunk + c;
                reinterpret_cast<threadgroup bfloat4*>(Bs)[i] =
                    (gr < K && gc < N_int) ? u4[(gr * N_int + gc) / 4] : bfloat4(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = 0; k < BK / 8; k++)
                for (uint r = 0; r < MR; r++) {
                    simdgroup_bfloat8x8 a;
                    simdgroup_load(a, As + (simd * 16 + r * 8) * BK + k * 8, BK);
                    for (uint c = 0; c < MC; c++) {
                        simdgroup_bfloat8x8 b;
                        simdgroup_load(b, Bs + k * 8 * BN + c * 8, BN);
                        simdgroup_multiply_accumulate(acc_up[r][c], a, b, acc_up[r][c]);
                    }
                }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ── Activation: intermediate[BM×BN] = gelu(acc_gate) * acc_up ──
        // intermediate is local to this chunk: column index = gc_local in [0, BN).
        for (uint r = 0; r < MR; r++)
            for (uint c = 0; c < MC; c++) {
                float2 g = reinterpret_cast<thread float2&>(acc_gate[r][c].thread_elements());
                float2 up_v = reinterpret_cast<thread float2&>(acc_up[r][c].thread_elements());
                uint qid = lane / 4;
                uint lr = (qid & 4) + ((lane / 2) % 4);
                uint lc = (qid & 2) * 2 + (lane % 2) * 2;
                uint gr_abs   = row + r * 8 + lr;            // absolute row in M
                uint gc_local = c * 8 + lc;                  // local col in BN
                uint gc_abs   = nint_chunk + gc_local;       // absolute col in N_int

                bool valid = (gr_abs < M) && (gc_abs < N_int);
                bool valid1 = (gr_abs < M) && (gc_abs + 1 < N_int);

                // Both pair elements: zero-fill out-of-range so intermediate
                // has no stale data from a prior chunk's iteration.
                intermediate[(gr_abs % BM) * BN + gc_local]
                    = valid ? bfloat(gelu_approx(g.x) * up_v.x) : bfloat(0);
                intermediate[(gr_abs % BM) * BN + gc_local + 1]
                    = valid1 ? bfloat(gelu_approx(g.y) * up_v.y) : bfloat(0);
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── GEMM 3 contribution: acc_out += intermediate × w_down[chunk:+BN, col:+BN] ──
        // Inner bk steps through BN (the chunk width) by BK. As load reads
        // the intermediate row-slice [bk, bk+BK) — using bk as a column
        // offset into the chunk-local intermediate buffer.
        for (uint bk = 0; bk < BN; bk += BK) {
            for (uint i = tid; i < (BM * BK) / 4; i += THREADS) {
                uint r = (i * 4) / BK, c = (i * 4) % BK;
                uint gr_local = (row % BM) + r;
                // Column within the BN-wide intermediate is bk + c.
                reinterpret_cast<threadgroup bfloat4*>(As)[i] =
                    (gr_local < BM)
                        ? reinterpret_cast<threadgroup bfloat4*>(intermediate + gr_local * BN + bk)[c / 4]
                        : bfloat4(0);
            }
            for (uint i = tid; i < (BK * BN) / 4; i += THREADS) {
                uint r = (i * 4) / BN, c = (i * 4) % BN;
                uint gr = nint_chunk + bk + r, gc = col + c;
                reinterpret_cast<threadgroup bfloat4*>(Bs)[i] =
                    (gr < N_int && gc < N) ? d4[(gr * N + gc) / 4] : bfloat4(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = 0; k < BK / 8; k++)
                for (uint r = 0; r < MR; r++) {
                    simdgroup_bfloat8x8 a;
                    simdgroup_load(a, As + (simd * 16 + r * 8) * BK + k * 8, BK);
                    for (uint c = 0; c < MC; c++) {
                        simdgroup_bfloat8x8 b;
                        simdgroup_load(b, Bs + k * 8 * BN + c * 8, BN);
                        simdgroup_multiply_accumulate(acc_out[r][c], a, b, acc_out[r][c]);
                    }
                }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // ── Write output ──
    for (uint r = 0; r < MR; r++)
        for (uint c = 0; c < MC; c++) {
            float2 v = reinterpret_cast<thread float2&>(acc_out[r][c].thread_elements());
            uint qid = lane / 4;
            uint lr = (qid & 4) + ((lane / 2) % 4);
            uint lc = (qid & 2) * 2 + (lane % 2) * 2;
            uint gr = row + r * 8 + lr;
            uint gc = col + c * 8 + lc;
            if (gr < M && gc < N) {
                out[(size_t)gr * N + gc] = bfloat(v.x);
                if (gc + 1 < N)
                    out[(size_t)gr * N + gc + 1] = bfloat(v.y);
            }
        }
}
