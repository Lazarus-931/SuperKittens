//
//  ops.h
//  SuperKittens
//
//  MMA operations: mm_AB and mm_ABt.
//  All operate on Tile<R,C> accumulators with threadgroup half sources.
//

#ifndef MEOW_MMA_OPS_H
#define MEOW_MMA_OPS_H

#include "base.h"

namespace meow {
namespace mma {

// ── mm_AB: C[M×N] += A[M×K] × B[K×N] ───────────────────────
// A is (ROWS*8) × (BK) at src_a with leading dim lda
// B is (BK) × (COLS*8) at src_b with leading dim ldb
// Accumulates into tile.data[r][c]

template <int BK, int ROWS, int COLS>
METAL_FUNC void mm_AB(thread Tile<ROWS, COLS>& tile,
                      const threadgroup half* src_a, uint lda,
                      const threadgroup half* src_b, uint ldb) {
    static_assert(BK % 8 == 0, "BK must be multiple of 8");
    for (int k = 0; k < BK / 8; k++) {
        for (int r = 0; r < ROWS; r++) {
            simdgroup_half8x8 a;
            simdgroup_load(a, src_a + r * 8 * lda + k * 8, lda);
            for (int c = 0; c < COLS; c++) {
                simdgroup_half8x8 b;
                simdgroup_load(b, src_b + k * 8 * ldb + c * 8, ldb);
                simdgroup_multiply_accumulate(tile.data[r][c], a, b, tile.data[r][c]);
            }
        }
    }
}

// ── mm_ABt: C[M×N] += A[M×K] × B^T[N×K] ────────────────────
// A is (ROWS*8) × (BK) at src_a with leading dim lda
// B is (COLS*8) × (BK) at src_b with leading dim ldb  (stored row-major, we transpose)
// i.e. B[n][k] is at src_b[n * ldb + k], we want B^T[k][n]

template <int BK, int ROWS, int COLS>
METAL_FUNC void mm_ABt(thread Tile<ROWS, COLS>& tile,
                       const threadgroup half* src_a, uint lda,
                       const threadgroup half* src_b, uint ldb) {
    static_assert(BK % 8 == 0, "BK must be multiple of 8");
    for (int k = 0; k < BK / 8; k++) {
        for (int r = 0; r < ROWS; r++) {
            simdgroup_half8x8 a;
            simdgroup_load(a, src_a + r * 8 * lda + k * 8, lda);
            for (int c = 0; c < COLS; c++) {
                simdgroup_half8x8 b;
                simdgroup_load(b, src_b + c * 8 * ldb + k * 8, ldb, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(tile.data[r][c], a, b, tile.data[r][c]);
            }
        }
    }
}

// ── mm_AtB: C[K×N] += A^T[M×K] × B[M×N] ────────────────────
// A is (BM) × (ROWS*8) at src_a with leading dim lda  (stored row-major, we transpose)
// B is (BM) × (COLS*8) at src_b with leading dim ldb
// Used for: state += K^T @ V in Mamba-2 inter-chunk update

template <int BM, int ROWS, int COLS>
METAL_FUNC void mm_AtB(thread Tile<ROWS, COLS>& tile,
                       const threadgroup half* src_a, uint lda,
                       const threadgroup half* src_b, uint ldb) {
    static_assert(BM % 8 == 0, "BM must be multiple of 8");
    for (int m = 0; m < BM / 8; m++) {
        for (int r = 0; r < ROWS; r++) {
            simdgroup_half8x8 a;
            simdgroup_load(a, src_a + m * 8 * lda + r * 8, lda, ulong2(0, 0), true);
            for (int c = 0; c < COLS; c++) {
                simdgroup_half8x8 b;
                simdgroup_load(b, src_b + m * 8 * ldb + c * 8, ldb);
                simdgroup_multiply_accumulate(tile.data[r][c], a, b, tile.data[r][c]);
            }
        }
    }
}

} // namespace mma
} // namespace meow

#endif // MEOW_MMA_OPS_H
