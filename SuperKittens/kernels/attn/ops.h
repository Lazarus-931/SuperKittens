//
//  ops.h
//  SuperKittens — attention-specific ops
//
//  By Alazar Manakelew
//
//  Tile-based softmax, causal mask, and online state update.

#ifndef SUPERKITTENS_ATTN_OPS_H
#define SUPERKITTENS_ATTN_OPS_H

#include <metal_stdlib>
#include "../../include/ops/simdgroup.h"
#include "../../include/ops/math.h"

using namespace metal;

namespace meow {
namespace attn {

struct SoftmaxState {
    float row_max;
    float row_sum;
};

// Apply causal mask to a tile of scores in threadgroup memory.
// Sets scores[row][col] = -INFINITY where key_pos > query_pos.
inline void apply_causal_mask(
    threadgroup float* scores,
    uint stride,
    uint row_start,
    uint tile_col,
    uint tile_m,
    uint tile_n,
    uint lid,
    uint num_threads)
{
    for (uint i = lid; i < tile_m * tile_n; i += num_threads) {
        uint r = i / tile_n;
        uint c = i % tile_n;
        uint query_pos = row_start + r;
        uint key_pos = tile_col + c;
        if (key_pos > query_pos)
            scores[r * stride + c] = -INFINITY;
    }
}

// Scale scores by rsqrt_d and find the per-row max.
// Each row's max is stored in row_maxes[row] (threadgroup float array, tile_m elements).
inline void scale_and_find_row_max(
    threadgroup float* scores,
    uint stride,
    uint tile_m,
    uint tile_n,
    float scale,
    threadgroup float* row_maxes,
    uint lid,
    uint num_threads)
{
    for (uint i = lid; i < tile_m * tile_n; i += num_threads) {
        uint r = i / tile_n;
        uint c = i % tile_n;
        uint idx = r * stride + c;
        scores[idx] *= scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // One thread per row initializes max
    if (lid < tile_m) {
        float m = -INFINITY;
        for (uint c = 0; c < tile_n; c++)
            m = max(m, scores[lid * stride + c]);
        row_maxes[lid] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Apply exp(score - row_max) and compute per-row sum.
// Stores exp values back into scores, sums into row_sums[row].
inline void exp_and_sum_rows(
    threadgroup float* scores,
    uint stride,
    uint tile_m,
    uint tile_n,
    threadgroup float* row_maxes,
    threadgroup float* row_sums,
    uint lid,
    uint num_threads)
{
    if (lid < tile_m) {
        float s = 0.0f;
        float rmax = row_maxes[lid];
        for (uint c = 0; c < tile_n; c++) {
            uint idx = lid * stride + c;
            float val = meow::ops::fast_exp(scores[idx] - rmax);
            scores[idx] = val;
            s += val;
        }
        row_sums[lid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Update online softmax state with a new tile's statistics.
// Returns the rescale factor to apply to previously accumulated output.
inline float update_state(
    thread SoftmaxState& state,
    float tile_max,
    float tile_sum)
{
    float new_max = max(state.row_max, tile_max);
    float rescale = meow::ops::fast_exp(state.row_max - new_max);
    state.row_sum = state.row_sum * rescale + tile_sum;
    state.row_max = new_max;
    return rescale;
}

} // namespace attn
} // namespace meow

#endif // SUPERKITTENS_ATTN_OPS_H
