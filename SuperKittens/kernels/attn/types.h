//
//  types.h
//  SuperKittens — attention tile configs
//
//  By Alazar Manakelew
//
//  Tile geometry constants for future templated attention kernels.
//  Pattern: define a config struct, then static_assert tile validity.
//  Not yet wired into the row-per-SIMD kernels (which use fixed enums).

#ifndef SUPERKITTENS_ATTN_TYPES_H
#define SUPERKITTENS_ATTN_TYPES_H

namespace meow {
namespace attn {

// Config for d=128 attention on M2+.
// TILE_M = query rows per threadgroup
// TILE_N = key columns per score tile (when materializing scores)
// TILE_K = head-dim chunk for tiled dot products
// TG_SIZE = threads per threadgroup

struct AttnConfig_128 {
    static constexpr unsigned int TILE_M   = 16;
    static constexpr unsigned int TILE_N   = 64;
    static constexpr unsigned int TILE_K   = 32;
    static constexpr unsigned int TG_SIZE  = 128;
    static constexpr unsigned int AS_STRIDE = 17;   // padded for bank conflicts
    static constexpr unsigned int BS_STRIDE = 65;
};

// Verify config fits in 32KB threadgroup memory before use.
static_assert(
    AttnConfig_128::TILE_M * AttnConfig_128::TILE_K * 2 +   // Q tile
    AttnConfig_128::TILE_N * AttnConfig_128::TILE_K * 2 +   // K tile
    AttnConfig_128::TILE_N * AttnConfig_128::TILE_K * 2 +   // V tile
    AttnConfig_128::TILE_M * AttnConfig_128::BS_STRIDE * 4  // scores
    <= 32768,
    "AttnConfig_128 exceeds 32KB threadgroup memory");

} // namespace attn
} // namespace meow

#endif // SUPERKITTENS_ATTN_TYPES_H
