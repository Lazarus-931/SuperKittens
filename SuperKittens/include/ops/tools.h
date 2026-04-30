//
//  tools.h
//  SuperKittens
//
//  General-purpose GPU utility functions (cumsum, causal decay, contiguity check).
//

#ifndef MEOW_OPS_TOOLS_H
#define MEOW_OPS_TOOLS_H

#include <metal_stdlib>
using namespace metal;

namespace meow {
namespace tools {

////////////////////////////////////////////////////////////////////////////////////////////
/// Softplus: log(1 + exp(x)), with bypass for large x to avoid overflow
////////////////////////////////////////////////////////////////////////////////////////////

METAL_FUNC float softplus(float x) {
    return (x > 20.0f) ? x : log(1.0f + metal::fast::exp(x));
}

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

template <typename T>
METAL_FUNC void reverse_cumsum_simd(thread T& val, uint lid) {
    T n = simd_shuffle_down(val, 1);
    val += (lid + 1 < 32) ? n : T(0);
    n = simd_shuffle_down(val, 2);
    val += (lid + 2 < 32) ? n : T(0);
    n = simd_shuffle_down(val, 4);
    val += (lid + 4 < 32) ? n : T(0);
    n = simd_shuffle_down(val, 8);
    val += (lid + 8 < 32) ? n : T(0);
    n = simd_shuffle_down(val, 16);
    val += (lid + 16 < 32) ? n : T(0);
}

////////////////////////////////////////////////////////////////////////////////////////////
/// Causal Decay — applies exp(cumsum[row] - cumsum[col]) * causal mask to an MMA Tile
////////////////////////////////////////////////////////////////////////////////////////////


template <int ROWS, int COLS>
METAL_FUNC void apply_causal_decay(thread meow::mma::Tile<ROWS, COLS>& tile,
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

            auto d = reinterpret_cast<thread float2&>(tile.data[r][c].thread_elements());

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

////////////////////////////////////////////////////////////////////////////////////////////
/// SIMD-based threadgroup cumsum helper
////////////////////////////////////////////////////////////////////////////////////////////
///

METAL_FUNC float min(float lhs, float rhs) {
    return (lhs < rhs) ? lhs : rhs;
}

METAL_FUNC float max(float lhs, float rhs) {
    return (lhs > rhs) ? lhs : rhs;
}

METAL_FUNC float clamp(float val, float min_val, float max_val) {
    if (val < min_val) {
        return min_val;
    } else if (val > max_val) {
        return max_val;
    } else {
        return val;
    }
}


////////////////////////////////////////////////////////////////////////////////////////////
/// SIMD-based threadgroup cumsum helper
////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int N>
METAL_FUNC void threadgroup_cumsum(
    threadgroup T* data,
    threadgroup T* simd_totals,
    uint lid, uint lane_id, uint simd_id)
{
    static_assert(N % 32 == 0, "N must be a multiple of SIMD width (32)");
    constexpr int N_SIMDS = N / 32;

    // Stage 1: each SIMD scans its own 32-lane slice
    if (lid < N) {
        T val = data[lid];
        cumsum_simd<T>(val, lane_id);
        data[lid] = val;
        if (lane_id == 31) simd_totals[simd_id] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 2: SIMD 0 scans the per-SIMD totals (N_SIMDS <= 32 by construction)
    if (simd_id == 0 && lane_id < N_SIMDS) {
        T t = simd_totals[lane_id];
        cumsum_simd<T>(t, lane_id);
        simd_totals[lane_id] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 3: add the prefix of preceding SIMDs back into each slice
    if (lid < N && simd_id > 0) {
        data[lid] += simd_totals[simd_id - 1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <typename T, int N>
METAL_FUNC void threadgroup_reverse_cumsum(
    threadgroup T* data,
    threadgroup T* simd_totals,
    uint lid, uint lane_id, uint simd_id)
{
    static_assert(N % 32 == 0, "N must be a multiple of SIMD width (32)");
    constexpr int N_SIMDS = N / 32;

    if (lid < N) {
        T val = data[lid];
        reverse_cumsum_simd<T>(val, lane_id);
        data[lid] = val;
        if (lane_id == 0) simd_totals[simd_id] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0 && lane_id < N_SIMDS) {
        T t = simd_totals[lane_id];
        reverse_cumsum_simd<T>(t, lane_id);
        simd_totals[lane_id] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid < N && simd_id + 1 < N_SIMDS) {
        data[lid] += simd_totals[simd_id + 1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

} // namespace tools
} // namespace meow




#endif // MEOW_OPS_TOOLS_H
