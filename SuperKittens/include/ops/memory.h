//
//  memory.h
//  SuperKittens
//
//  By Alazar Manakelew
//
//  Tile loaders: safe/unsafe, fused scale, coalesced write-back.

#ifndef MEOW_OPS_MEMORY_H
#define MEOW_OPS_MEMORY_H

#include <metal_stdlib>
using namespace metal;

namespace meow {
namespace ops {

// Load tile from device half* → threadgroup float*, with scale applied.
// Use for Q tiles where rsqrt_d is baked into the load.
inline void load_tile_scaled(
    device const half* src, uint ld,
    threadgroup float* dst, uint dst_stride,
    uint rows, uint cols,
    uint tileRow, uint tileCol,
    uint lid, uint num_threads,
    uint maxR, uint maxC,
    float scale)
{
    for (uint i = lid; i < rows * cols; i += num_threads) {
        uint r = i / cols, c = i % cols;
        uint gr = tileRow + r, gc = tileCol + c;
        dst[r * dst_stride + c] = (gr < maxR && gc < maxC) ? float(src[gr * ld + gc]) * scale : 0.0f;
    }
}

// Load tile without scale. Half → float promotion.
inline void load_tile(
    device const half* src, uint ld,
    threadgroup float* dst, uint dst_stride,
    uint rows, uint cols,
    uint tileRow, uint tileCol,
    uint lid, uint num_threads,
    uint maxR, uint maxC)
{
    for (uint i = lid; i < rows * cols; i += num_threads) {
        uint r = i / cols, c = i % cols;
        uint gr = tileRow + r, gc = tileCol + c;
        dst[r * dst_stride + c] = (gr < maxR && gc < maxC) ? float(src[gr * ld + gc]) : 0.0f;
    }
}

// Load transposed: B[gc][gr] instead of B[gr][gc]. For K^T.
inline void load_tile_transposed(
    device const half* src, uint ld,
    threadgroup float* dst, uint dst_stride,
    uint rows, uint cols,
    uint tileRow, uint tileCol,
    uint lid, uint num_threads,
    uint maxR, uint maxC)
{
    for (uint i = lid; i < rows * cols; i += num_threads) {
        uint r = i / cols, c = i % cols;
        uint gr = tileRow + r, gc = tileCol + c;
        dst[r * dst_stride + c] = (gc < maxR && gr < maxC) ? float(src[gc * ld + gr]) : 0.0f;
    }
}

// Unsafe load — no bounds checks. For interior tiles only.
inline void load_tile_unsafe(
    device const half* src, uint ld,
    threadgroup float* dst, uint dst_stride,
    uint rows, uint cols,
    uint tileRow, uint tileCol,
    uint lid, uint num_threads)
{
    for (uint i = lid; i < rows * cols; i += num_threads) {
        uint r = i / cols, c = i % cols;
        dst[r * dst_stride + c] = float(src[(tileRow + r) * ld + (tileCol + c)]);
    }
}

// Unsafe transposed load — no bounds checks.
inline void load_tile_transposed_unsafe(
    device const half* src, uint ld,
    threadgroup float* dst, uint dst_stride,
    uint rows, uint cols,
    uint tileRow, uint tileCol,
    uint lid, uint num_threads)
{
    for (uint i = lid; i < rows * cols; i += num_threads) {
        uint r = i / cols, c = i % cols;
        dst[r * dst_stride + c] = float(src[(tileCol + c) * ld + (tileRow + r)]);
    }
}

// Coalesced write-back: float shared mem → half device, 4 elements at a time.
inline void store_tile_half4(
    threadgroup float* src, uint src_stride,
    device half* dst, uint ld,
    uint rows, uint cols,
    uint tileRow,
    uint lid, uint num_threads,
    uint maxR)
{
    for (uint i = lid; i < rows * cols; i += num_threads) {
        uint r = i / cols, c = i % cols;
        uint gr = tileRow + r;
        if (gr < maxR)
            dst[gr * ld + c] = half(src[r * src_stride + c]);
    }
}





} // namespace ops
} // namespace meow

#endif // MEOW_OPS_MEMORY_H
