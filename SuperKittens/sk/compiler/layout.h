//
//  layout.h — SuperKittens compiler: 8×8 tile → SIMD → threadgroup
//
//  The layout algebra.  Every kernel decomposes into 8×8 tiles
//  distributed across SIMD groups.  All constants are enums
//  (Metal 3.1 rejects constexpr members at struct scope).
//

#pragma once

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

namespace sk {
namespace compiler {

// ── The atom: 8×8 tile ────────────────────────────────────────────

// Physical unit: simdgroup_half8x8 (load) / simdgroup_float8x8 (acc).
// 64 elements across 32 lanes → 2 elements per lane.

template<typename ElementT, typename AccumulatorT = float>
struct Tile8x8 {
    enum : int {
        rows     = 8,
        cols     = 8,
        elements = 64,
        per_lane = 2,
        // Threadgroup memory: 64 × sizeof(ElementT). Half=128 bytes.
        tg_bytes = elements * (int)sizeof(ElementT),
        // Register: 8 per lane for float accum, 4 for half
        regs = ((int)sizeof(AccumulatorT) == 4) ? 8 : 4
    };
};

// ── SIMD group layout ─────────────────────────────────────────────

// 32 lanes × MR×MC tiles.  Total rows = MR*8, cols = MC*8.
// Register usage = MR * MC * Tile8x8::regs per lane.

template<int MR_, int MC_>
struct SimdLayout {
    enum : int {
        MR   = MR_,
        MC   = MC_,
        rows = MR * 8,
        cols = MC * 8
    };

    template<typename AccT = float>
    static constexpr int regs_used() {
        constexpr int per = ((int)sizeof(AccT) == 4) ? 8 : 4;
        return MR * MC * per;
    }
};

// ── Threadgroup decomposition ─────────────────────────────────────

// A 2D grid of SIMD groups: grid_rows × grid_cols.
// Total SIMD groups ≤ 32 (1024 threads max).

template<int GridRows_, int GridCols_, typename SimdLayout_>
struct TgDecomposition {
    using Simd = SimdLayout_;

    enum : int {
        grid_rows  = GridRows_,
        grid_cols  = GridCols_,
        n_simds    = grid_rows * grid_cols,
        n_threads  = n_simds * 32,
        total_rows = grid_rows * Simd::rows,
        total_cols = grid_cols * Simd::cols,
        total_tiles = grid_rows * grid_cols * Simd::MR * Simd::MC
    };

    static_assert(n_simds <= 32, "max 32 SIMD groups per threadgroup");

    template<typename AccT = float>
    static constexpr bool fits_in_regs(int budget) {
        return Simd::template regs_used<AccT>() <= budget;
    }

    template<int TGBytes>
    static constexpr bool fits_in_tgmem() {
        return TGBytes <= 32768;
    }
};

// ── Preset decompositions ─────────────────────────────────────────

// GEMM 128: 2×2 grid, MR=2,MC=8 → 32×64 window
using GemmDecomp_128 = TgDecomposition<2, 2, SimdLayout<2, 8>>;

// GEMM 1024: 4×8 grid, MR=1,MC=1 → 32×64 window
using GemmDecomp_1024 = TgDecomposition<4, 8, SimdLayout<1, 1>>;

// FA row-parallel: 1×32 grid, MR=1,MC=1 → 32 Q rows each
using FaRowDecomp_1024 = TgDecomposition<1, 32, SimdLayout<1, 1>>;

// ── Kernel params ─────────────────────────────────────────────────

template<typename Decomp, int BK_>
struct GemmParams {
    using Tg = Decomp;
    enum : int {
        BK = BK_,
        BM = Tg::total_rows,
        BN = Tg::total_cols,
        n_threads = Tg::n_threads
    };
};

} // namespace compiler
} // namespace sk
