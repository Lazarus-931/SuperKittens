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

inline float4 load_half4(
    device const half* src,
    uint base)
{
    return float4(float(src[base + 0]),
                  float(src[base + 1]),
                  float(src[base + 2]),
                  float(src[base + 3]));
}

inline float4 load_half4_or_zero(
    device const half* src,
    uint base,
    uint valid)
{
    return float4(valid > 0 ? float(src[base + 0]) : 0.0f,
                  valid > 1 ? float(src[base + 1]) : 0.0f,
                  valid > 2 ? float(src[base + 2]) : 0.0f,
                  valid > 3 ? float(src[base + 3]) : 0.0f);
}

inline void store_half4(
    device half* dst,
    uint base,
    float4 val)
{
    dst[base + 0] = half(val.x);
    dst[base + 1] = half(val.y);
    dst[base + 2] = half(val.z);
    dst[base + 3] = half(val.w);
}

inline void store_half4_masked(
    device half* dst,
    uint base,
    float4 val,
    uint valid)
{
    if (valid > 0) dst[base + 0] = half(val.x);
    if (valid > 1) dst[base + 1] = half(val.y);
    if (valid > 2) dst[base + 2] = half(val.z);
    if (valid > 3) dst[base + 3] = half(val.w);
}

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
