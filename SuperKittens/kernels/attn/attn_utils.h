//
//  attn_utils.h
//  SuperKittens — attention utility constants and tile configs
//
//  By Alazar Manakelew

#ifndef SUPERKITTENS_ATTN_UTILS_H
#define SUPERKITTENS_ATTN_UTILS_H

namespace meow {
namespace attn {

// Tile configuration for attention kernels
// Tuned for M4: 9.3 KB shared mem, leaves 22 KB for dynamic caching
struct TileConfig {
    static constexpr unsigned int TILE_M = 16;       // query rows per threadgroup
    static constexpr unsigned int TILE_N = 128;      // key columns per tile
    static constexpr unsigned int TILE_K = 16;       // K-block depth
    static constexpr unsigned int TG_SIZE = 128;     // threads per threadgroup (4 SIMDs)
    static constexpr unsigned int NUM_SIMDS = 4;
    static constexpr unsigned int AS_STRIDE = 17;    // padded for bank conflicts
    static constexpr unsigned int BS_STRIDE = 129;   // padded for bank conflicts
};

} // namespace attn
} // namespace meow

#endif // SUPERKITTENS_ATTN_UTILS_H
