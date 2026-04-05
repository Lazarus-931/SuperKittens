//
//  types.h
//  SuperKittens — attention utility constants and tile configs
//
//  By Alazar Manakelew

#ifndef SUPERKITTENS_ATTN_UTILS_H
#define SUPERKITTENS_ATTN_UTILS_H

namespace superkittens {
namespace attn {


/// TILE_M = query rows per threadgroup
/// TILE_N = key columns per tile
/// TILE_K = k-block depth
/// TG_SIZE = threads per threadgroup
/// AS_STRIDE  and BS_STRIDE= padded for bank conficts


struct A_2048_128_Config {
    static constant unsigned int TILE_M = 16;
    static constant unsigned int TILE_N = 128;
    static constant unsigned int TILE_K = 16;
    static constant unsigned int TG_SIZE = 128;
    static constant unsigned int NUM_SIMDS = 4;
    static constant unsigned int AS_STRIDE = 17;
    static constant unsigned int BS_STRIDE = 129;
};


} // namespace attn
} // namespace superkittens

#endif // SUPERKITTENS_ATTN_UTILS_H
