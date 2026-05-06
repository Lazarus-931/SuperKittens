//
//  mma.h — SuperKittens DSL: simdgroup matrix multiply-accumulate
//
//  Register-aware MR/MC derivation per Apple GPU family.
//  Metal 3.1 compatible: thread refs, enums, no namespace constexpr.
//

#pragma once

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include "feature.h"

using namespace metal;

namespace sk {
namespace dsl {

// ── GemmConfig: the ONLY type the user sees ────────────────────────
//
//  User specifies threadgroup tile sizes and thread count.
//  The compiler derives the 8×8 decomposition internally.
//
//     GemmConfig<BM<64>, BN<64>, BK<32>, Threads<128>> gemm;
//     auto acc = gemm.make_acc();
//     gemm.mma(acc, As, Bs, simd);
//     gemm.store(C, N, 0, 0, M, N, acc, simd, lane);
//

template<int BM_, int BN_, int BK_, int THREADS_ = 128>
struct GemmConfig {
    enum : int {
        BM = BM_, BN = BN_, BK = BK_, THREADS = THREADS_,
        N_SIMDS = THREADS / 32
    };

    static_assert(BM % (N_SIMDS * 8) == 0, "BM must be multiple of N_SIMDS × 8");
    static_assert(BN % 8 == 0, "BN must be multiple of 8");

    // ── Derived from BM, BN, THREADS (8×8 mapping is internal) ────
    enum : int {
        MR = BM / (N_SIMDS * 8),         // 8×8 tiles vertically per SIMD
        MC = BN / 8,                      // 8×8 tiles horizontally per SIMD
        ROWS_PER_SIMD = BM / N_SIMDS,     // rows owned by one SIMD group
        REGS_PER_LANE = MR * MC * 2       // float register pressure per lane
    };

    // Conservative floor: M1's 96 regs.  Compiler spills above this.
    // M2=112, M3+=128.  Use sk::compiler::FamilyBudget for per-family tuning.
    enum : int { MIN_REGS_PER_LANE = 96 };
    static_assert(REGS_PER_LANE <= MIN_REGS_PER_LANE,
        "tile exceeds M1 register budget — reduce BM or BN");

    // Accumulator type (user doesn't declare simdgroup_float8x8 directly)
    struct Accumulator {
        simdgroup_float8x8 data[MR][MC];
        void zero() {
            for (int r = 0; r < MR; r++)
                for (int c = 0; c < MC; c++)
                    data[r][c] = simdgroup_float8x8(0.0f);
        }
    };

    static Accumulator make_acc() { Accumulator a; a.zero(); return a; }

    // MMA: As[BM][BK] × Bs[BK][BN] → acc (8×8 loop internal)
    static void mma(
        thread Accumulator& acc,
        const threadgroup half* As,
        const threadgroup half* Bs,
        uint simd_id)
    {
        const uint row_base = simd_id * ROWS_PER_SIMD;
        for (int k = 0; k < BK / 8; k++) {
            for (int r = 0; r < MR; r++) {
                simdgroup_half8x8 a;
                simdgroup_load(a, As + (row_base + r * 8) * BK + k * 8, BK);
                for (int c = 0; c < MC; c++) {
                    simdgroup_half8x8 b;
                    simdgroup_load(b, Bs + k * 8 * BN + c * 8, BN);
                    simdgroup_multiply_accumulate(acc.data[r][c], a, b, acc.data[r][c]);
                }
            }
        }
    }

    // Writeback: store accumulator rows to device memory
    static void store(
        device half* C, uint ldC,
        uint row_start, uint col_start,
        uint max_rows, uint max_cols,
        const thread Accumulator& acc,
        uint simd_id, uint lane_id)
    {
        const uint row_base = simd_id * ROWS_PER_SIMD;
        for (int r = 0; r < MR; r++) {
            for (int c = 0; c < MC; c++) {
                auto raw = acc.data[r][c].thread_elements();
                float2 v = *(const thread float2*)&raw;
                uint qid = lane_id / 4;
                uint lr = (qid & 4) + ((lane_id / 2) % 4);
                uint lc = (qid & 2) * 2 + (lane_id % 2) * 2;
                uint gr = row_start + row_base + r * 8 + lr;
                uint gc = col_start + c * 8 + lc;
                if (gr < max_rows && gc < max_cols) {
                    C[gr * ldC + gc]     = half(v.x);
                    C[gr * ldC + gc + 1] = half(v.y);
                }
            }
        }
    }
};

} // namespace dsl
} // namespace sk
