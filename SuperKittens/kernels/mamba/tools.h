//
//  tools.h
//  SuperKittens
//
//  Mamba-specific utility functions.
//

#ifndef SUPERKITTENS_MAMBA_TOOLS_H
#define SUPERKITTENS_MAMBA_TOOLS_H

#include <metal_stdlib>
using namespace metal;

namespace superkittens {
namespace tools {

////////////////////////////////////////////////////////////////////////////////////////////
/// Hillis-Steele cumsum
////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
METAL_FUNC void cumsum_simd(thread T& val, uint lid) {
    T n = simd_shuffle_up(val, 1);
    val += (lid >= 1) ? n : T(0);
    n = simd_shuffle_up(val, 2);
    val += (lid >= 2) ? n : T(0);
    n = simd_shuffle_up(val, 4);
    val += (lid >= 4) ? n : T(0);
    n = simd_shuffle_up(val, 8);
    val += (lid >= 8) ? n : T(0);
    n = simd_shuffle_up(val, 16);
    val += (lid >= 16) ? n : T(0);
}

////////////////////////////////////////////////////////////////////////////////////////////
/// Causal Decay — applies exp(cumsum[row] - cumsum[col]) * causal mask to an MMA Tile
////////////////////////////////////////////////////////////////////////////////////////////

// lane_id → (local_row, local_col) within each 8×8 block via Apple's layout:
//   qid = lane_id / 4
//   local_row = (qid & 4) + (lane_id / 2) % 4
//   local_col = (qid & 2) * 2 + (lane_id % 2) * 2
//   thread owns (local_row, local_col) and (local_row, local_col + 1)

template <int ROWS, int COLS>
METAL_FUNC void apply_causal_decay(thread superkittens::mma::Tile<ROWS, COLS>& tile,
                                    const threadgroup float* a_cumsum,
                                    uint row_base, uint lane_id) {
    uint qid = lane_id / 4;
    uint local_row = (qid & 4) + (lane_id / 2) % 4;
    uint local_col = (qid & 2) * 2 + (lane_id % 2) * 2;

    for (int r = 0; r < ROWS; r++) {
        uint abs_row = row_base + r * 8 + local_row;
        float cs_row = a_cumsum[abs_row];
        for (int c = 0; c < COLS; c++) {
            uint abs_col0 = c * 8 + local_col;
            uint abs_col1 = abs_col0 + 1;

            auto& d = reinterpret_cast<thread float2&>(tile.data[r][c].thread_elements());

            d.x *= (abs_col0 <= abs_row) ? exp(cs_row - a_cumsum[abs_col0]) : 0.0f;
            d.y *= (abs_col1 <= abs_row) ? exp(cs_row - a_cumsum[abs_col1]) : 0.0f;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////
/// Contiguous Check — true if strides match a canonical row-major packed layout
////////////////////////////////////////////////////////////////////////////////////////////

template <int RANK>
METAL_FUNC bool is_contiguous(const thread int* shape, const thread int* strides) {
    int expected = 1;
    for (int i = RANK - 1; i >= 0; i--) {
        if (shape[i] > 1 && strides[i] != expected) return false;
        expected *= shape[i];
    }
    return true;
}

} // namespace tools
} // namespace superkittens

#endif // SUPERKITTENS_MAMBA_TOOLS_H
