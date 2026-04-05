//
//  ops.h
//  SuperKittens — attention-specific ops
//
//  By Alazar Manakelew
//
//  Online softmax update, score scaling, causal mask.

#ifndef SUPERKITTENS_ATTN_OPS_H
#define SUPERKITTENS_ATTN_OPS_H

#include <metal_stdlib>
#include "../../ops/global.h"
#include "../../ops/math.h"

using namespace metal;

namespace superkittens {
namespace attn {

struct SoftmaxState {
    float row_max;
    float row_sum;
};

// Scale scores + find tile_max in one pass over shared memory.
inline float scale_and_max(
    threadgroup float* scores, uint stride,
    uint tile_size,
    float rsqrt_d,
    uint lid, uint num_threads)
{
    float tile_max = -INFINITY;
    for (uint i = lid; i < tile_size; i += num_threads) {
        uint idx = (i / 128) * stride + (i % 128);
        scores[idx] *= rsqrt_d;
        tile_max = max(tile_max, scores[idx]);
    }
    return tile_max;
}

// Apply exp(score - max) and accumulate sum in one pass.
inline float exp_and_sum(
    threadgroup float* scores, uint stride,
    uint tile_size,
    float row_max,
    uint lid, uint num_threads)
{
    float tile_sum = 0.0f;
    for (uint i = lid; i < tile_size; i += num_threads) {
        uint idx = (i / 128) * stride + (i % 128);
        float val = superkittens::ops::fast_exp(scores[idx] - row_max);
        scores[idx] = val;
        tile_sum += val;
    }
    return tile_sum;
}

// Apply causal mask: set scores where key_pos > query_pos to -inf.
inline void apply_causal_mask(
    threadgroup float* scores, uint stride,
    uint tileRow, uint tileCol,
    uint lid, uint num_threads)
{
    for (uint i = lid; i < 16 * 128; i += num_threads) {
        uint r = i / 128, c = i % 128;
        uint query_pos = tileRow + r;
        uint key_pos = tileCol + c;
        if (key_pos > query_pos)
            scores[r * stride + c] = -INFINITY;
    }
}

// Update online softmax state with new tile stats.
inline void update_state(
    thread SoftmaxState& state,
    float tile_max,
    float tile_sum,
    thread float& scale)
{
    float old_max = state.row_max;
    state.row_max = max(state.row_max, tile_max);
    scale = superkittens::ops::fast_exp(old_max - state.row_max);
    state.row_sum = state.row_sum * scale + tile_sum;
}

} // namespace attn
} // namespace superkittens

#endif // SUPERKITTENS_ATTN_OPS_H
