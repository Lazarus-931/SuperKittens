//
//  gated_mlp.metal — fused SiLU-gated MLP
//
//  gate = x @ w_gate, up = x @ w_up
//  out = (SiLU(gate) * up) @ w_down
//
//  Two GEMMs (gate, up) share the same input x, then fuse activation +
//  element-wise multiply + third GEMM into one kernel.
//

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

enum : uint { BM = 64, BN = 64, BK = 32, MR = 2, MC = 8, THREADS = 128 };

[[host_name("gated_mlp")]]
[[kernel, max_total_threads_per_threadgroup(THREADS)]]
void gated_mlp(
    device const half* x,          // (M, K)
    device const half* w_gate,     // (K, N_intermediate)
    device const half* w_up,       // (K, N_intermediate)
    device const half* w_down,     // (N_intermediate, N)
    device half* out,              // (M, N)
    constant uint& M, constant uint& N, constant uint& K, constant uint& N_int,
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    const uint row = gid.y * BM + simd * 16;
    const uint col = gid.x * BN;

    // ── GEMM 1: gate = x @ w_gate ──
    simdgroup_float8x8 acc_gate[MR][MC] = {};
    threadgroup half As[BM * BK], Bs[BK * BN];
    const uint tid = simd * 32 + lane;
    const device half4* x4 = reinterpret_cast<const device half4*>(x);
    const device half4* g4 = reinterpret_cast<const device half4*>(w_gate);

    for (uint bk = 0; bk < K; bk += BK) {
        for (uint i = tid; i < (BM * BK) / 4; i += THREADS) {
            uint r = (i * 4) / BK, c = (i * 4) % BK;
            uint gr = row + r, gc = bk + c;
            reinterpret_cast<threadgroup half4*>(As)[i] =
                (gr < M && gc < K) ? x4[(gr * K + gc) / 4] : half4(0);
        }
        for (uint i = tid; i < (BK * BN) / 4; i += THREADS) {
            uint r = (i * 4) / BN, c = (i * 4) % BN;
            uint gr = bk + r, gc = col + c;
            reinterpret_cast<threadgroup half4*>(Bs)[i] =
                (gr < K && gc < N_int) ? g4[(gr * N_int + gc) / 4] : half4(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < BK / 8; k++)
            for (uint r = 0; r < MR; r++) {
                simdgroup_half8x8 a;
                simdgroup_load(a, As + (simd * 16 + r * 8) * BK + k * 8, BK);
                for (uint c = 0; c < MC; c++) {
                    simdgroup_half8x8 b;
                    simdgroup_load(b, Bs + k * 8 * BN + c * 8, BN);
                    simdgroup_multiply_accumulate(acc_gate[r][c], a, b, acc_gate[r][c]);
                }
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── GEMM 2: up = x @ w_up (reuse As, reload Bs from w_up) ──
    simdgroup_float8x8 acc_up[MR][MC] = {};
    const device half4* u4 = reinterpret_cast<const device half4*>(w_up);

    for (uint bk = 0; bk < K; bk += BK) {
        // Reload As (same as before, but we can reuse from the last GEMM pass... complicated)
        // For simplicity: reload As
        for (uint i = tid; i < (BM * BK) / 4; i += THREADS) {
            uint r = (i * 4) / BK, c = (i * 4) % BK;
            uint gr = row + r, gc = bk + c;
            reinterpret_cast<threadgroup half4*>(As)[i] =
                (gr < M && gc < K) ? x4[(gr * K + gc) / 4] : half4(0);
        }
        for (uint i = tid; i < (BK * BN) / 4; i += THREADS) {
            uint r = (i * 4) / BN, c = (i * 4) % BN;
            uint gr = bk + r, gc = col + c;
            reinterpret_cast<threadgroup half4*>(Bs)[i] =
                (gr < K && gc < N_int) ? u4[(gr * N_int + gc) / 4] : half4(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < BK / 8; k++)
            for (uint r = 0; r < MR; r++) {
                simdgroup_half8x8 a;
                simdgroup_load(a, As + (simd * 16 + r * 8) * BK + k * 8, BK);
                for (uint c = 0; c < MC; c++) {
                    simdgroup_half8x8 b;
                    simdgroup_load(b, Bs + k * 8 * BN + c * 8, BN);
                    simdgroup_multiply_accumulate(acc_up[r][c], a, b, acc_up[r][c]);
                }
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── SiLU(gate) * up ──
    // Store intermediate to threadgroup scratch, then GEMM with w_down
    threadgroup half intermediate[BM * BN];  // gate_act * up

    for (uint r = 0; r < MR; r++)
        for (uint c = 0; c < MC; c++) {
            float2 g = reinterpret_cast<thread float2&>(acc_gate[r][c].thread_elements());
            float2 u = reinterpret_cast<thread float2&>(acc_up[r][c].thread_elements());
            uint qid = lane / 4, lr = (qid & 4) + ((lane / 2) % 4), lc = (qid & 2) * 2 + (lane % 2) * 2;
            uint gr = row + r * 8 + lr;
            uint gc = col + c * 8 + lc;

            if (gr < M && gc < N_int) {
                // SiLU activation
                float g0 = g.x / (1.0f + metal::fast::exp(-g.x));
                float g1 = g.y / (1.0f + metal::fast::exp(-g.y));
                intermediate[(gr % BM) * BN + gc]     = half(g0 * u.x);
                intermediate[(gr % BM) * BN + gc + 1] = half(g1 * u.y);
            }
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── GEMM 3: out = intermediate @ w_down ──
    simdgroup_float8x8 acc_out[MR][MC] = {};
    const device half4* d4 = reinterpret_cast<const device half4*>(w_down);

    for (uint bk = 0; bk < N_int; bk += BK) {
        // Load A from intermediate scratch
        for (uint i = tid; i < (BM * BK) / 4; i += THREADS) {
            uint r = (i * 4) / BK, c = (i * 4) % BK;
            uint gr = (row % BM) + r, gc = bk + c;
            reinterpret_cast<threadgroup half4*>(As)[i] =
                (gr < BM && gc < N_int) ? reinterpret_cast<threadgroup half4*>(intermediate + gr * BN)[c / 4] : half4(0);
        }
        for (uint i = tid; i < (BK * BN) / 4; i += THREADS) {
            uint r = (i * 4) / BN, c = (i * 4) % BN;
            uint gr = bk + r, gc = col + c;
            reinterpret_cast<threadgroup half4*>(Bs)[i] =
                (gr < N_int && gc < N) ? d4[(gr * N + gc) / 4] : half4(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < BK / 8; k++)
            for (uint r = 0; r < MR; r++) {
                simdgroup_half8x8 a;
                simdgroup_load(a, As + (simd * 16 + r * 8) * BK + k * 8, BK);
                for (uint c = 0; c < MC; c++) {
                    simdgroup_half8x8 b;
                    simdgroup_load(b, Bs + k * 8 * BN + c * 8, BN);
                    simdgroup_multiply_accumulate(acc_out[r][c], a, b, acc_out[r][c]);
                }
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Write output ──
    for (uint r = 0; r < MR; r++)
        for (uint c = 0; c < MC; c++) {
            float2 v = reinterpret_cast<thread float2&>(acc_out[r][c].thread_elements());
            uint qid = lane / 4, lr = (qid & 4) + ((lane / 2) % 4), lc = (qid & 2) * 2 + (lane % 2) * 2;
            uint gr = row + r * 8 + lr;
            uint gc = col + c * 8 + lc;
            if (gr < M && gc < N) {
                size_t off = (size_t)gr * N + gc;
                (reinterpret_cast<device half*>(out))[off]     = half(v.x);
                (reinterpret_cast<device half*>(out))[off + 1] = half(v.y);
            }
        }
}
