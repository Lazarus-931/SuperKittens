//
//  tile.h
//  SuperKittens
//
//  Created by Alazar Manakelew on 4/4/26.
//

#ifndef SUPERKITTENS_TOOLS_TILE_H
#define SUPERKITTENS_TOOLS_TILE_H

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

namespace superkittens {
namespace tools {


/// simdgroup_8x8 is not accessible without simd_group, so like mlx, we choose to construct a abstract tile to allow direct access, and more importantly(a huge boost in perf!),
/// that this leads to not having us use shared memory!
/// This means defining fragments as well as the tiles that will use them
/// float2 is perfect because this holds 8 bytes (.x and .y x 2), defined as a fragment used in the tile

// Reduction op types for templated row_reduce
struct MaxOp { static float apply(float a, float b) { return metal::max(a, b); } };
struct SumOp { static float apply(float a, float b) { return a + b; } };

struct Frag {

    /// each thread get's two consecutive cols in the same row
    float2 data;
    short2 coord;  // set once per kernel, reused for all load/store

    Frag() : data(float2(0)), coord(short2(0)) {}

    void clear() { data = float2(0); }

    /// maps simd_lane_id to row and col position within 8x8, called once in kernel
    static short2 get_coord(uint simd_lane_id) {
        short qid = simd_lane_id / 4;
        short row = (qid & 4) + ((simd_lane_id / 2) % 4);
        short col = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
        return short2(col, row);
    }

    void set_coord(uint lane_id) { coord = get_coord(lane_id); }

    void load(const threadgroup float* src, uint stride) {
        data.x = src[coord.y * stride + coord.x];
        data.y = src[coord.y * stride + coord.x + 1];
    }

    void store(threadgroup float* dst, uint stride) const {
        dst[coord.y * stride + coord.x]     = data.x;
        dst[coord.y * stride + coord.x + 1] = data.y;
    }

    void store(device half* dst, uint stride) const {
        dst[coord.y * stride + coord.x]     = half(data.x);
        dst[coord.y * stride + coord.x + 1] = half(data.y);
    }

    /// templated row reduction — handles max, sum, any associative op
    template <typename Op>
    float row_reduce() const {
        float v = Op::apply(data.x, data.y);
        v = Op::apply(v, simd_shuffle_xor(v, 1));
        v = Op::apply(v, simd_shuffle_xor(v, 8));
        return v;
    }

    float row_max() const { return row_reduce<MaxOp>(); }
    float row_sum() const { return row_reduce<SumOp>(); }

    /// in-register arithmetic — no shared memory
    void scale(float s)    { data *= s; }
    void add(float s)      { data += s; }
    void add(Frag f)       { data += f.data; }
    void sub(float s)      { data -= s; }

    void exp()  { data = float2(metal::fast::exp(data.x), metal::fast::exp(data.y)); }
    void exp2() { data = float2(metal::fast::exp2(data.x), metal::fast::exp2(data.y)); }

    /// convert from native simdgroup
    void from_simd(thread simdgroup_float8x8& m) {
        data = reinterpret_cast<thread float2&>(m.thread_elements());
    }

    /// convert to native simdgroup
    void to_simd(thread simdgroup_float8x8& m) const {
        reinterpret_cast<thread float2&>(m.thread_elements()) = data;
    }

    /// MMA without leaving Frag — no from_simd/to_simd overhead
    static void mma(thread Frag& D, thread Frag& A, thread Frag& B) {
        simdgroup_float8x8 dM, aM, bM;
        reinterpret_cast<thread float2&>(aM.thread_elements()) = A.data;
        reinterpret_cast<thread float2&>(bM.thread_elements()) = B.data;
        reinterpret_cast<thread float2&>(dM.thread_elements()) = D.data;
        simdgroup_multiply_accumulate(dM, aM, bM, dM);
        D.data = reinterpret_cast<thread float2&>(dM.thread_elements());
    }
};
    

template <int ROWS, int COLS>
struct Tile {

    Frag frags[ROWS * COLS];

    void clear() {
        for (int i = 0; i < ROWS * COLS; i++) frags[i].clear();
    }

    /// call once at kernel start — propagates coord to all frags
    void set_coord(uint lane_id) {
        for (int i = 0; i < ROWS * COLS; i++) frags[i].set_coord(lane_id);
    }

    thread Frag& at(int r, int c) { return frags[r * COLS + c]; }
    const thread Frag& at(int r, int c) const { return frags[r * COLS + c]; }

    /// templated row reduction — works for max, sum, any Op
    template <typename Op>
    void row_reduce(thread float* out) const {
        for (int i = 0; i < ROWS; i++) {
            out[i] = at(i, 0).template row_reduce<Op>();
            for (int j = 1; j < COLS; j++)
                out[i] = Op::apply(out[i], at(i, j).template row_reduce<Op>());
        }
    }

    void row_max(thread float* out) const { row_reduce<MaxOp>(out); }
    void row_sum(thread float* out) const { row_reduce<SumOp>(out); }

    /// multiply each row by a per-row scalar
    void row_scale(const thread float* vals) {
        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++)
                frags[i * COLS + j].scale(vals[i]);
    }

    /// subtract per-row scalar then exp — online softmax in one call
    void row_softmax_exp(const thread float* maxes) {
        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++) {
                frags[i * COLS + j].sub(maxes[i]);
                frags[i * COLS + j].exp();
            }
    }

    /// same but exp2 — faster on Apple GPU
    void row_softmax_exp2(const thread float* maxes) {
        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++) {
                frags[i * COLS + j].sub(maxes[i]);
                frags[i * COLS + j].exp2();
            }
    }

    /// scale entire tile by one scalar
    void scale(float s) {
        for (int i = 0; i < ROWS * COLS; i++) frags[i].scale(s);
    }

    /// convert from native simdgroup MMA accumulators (call after K-loop)
    void from_simd(thread simdgroup_float8x8* mats) {
        for (int i = 0; i < ROWS * COLS; i++)
            frags[i].from_simd(mats[i]);
    }

    /// load all frags from shared memory (frags laid out at frag_stride apart)
    void load(const threadgroup float* src, uint stride, uint frag_stride) {
        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++)
                at(i, j).load(src + i * 8 * stride + j * frag_stride, stride);
    }

    /// store all frags to shared memory
    void store(threadgroup float* dst, uint stride, uint frag_stride) const {
        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++)
                at(i, j).store(dst + i * 8 * stride + j * frag_stride, stride);
    }

    /// store all frags to device half output
    void store(device half* dst, uint stride, uint frag_stride) const {
        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++)
                at(i, j).store(dst + i * 8 * stride + j * frag_stride, stride);
    }
};

} // namespace tools
} // namespace superkittens

/// ──────────────────────────────────────────────────────────
/// Future improvements:
/// 1. Compile-time strides via template params (like MLX's load<T, w_x, w_y, str_x, str_y>)
///    — lets the compiler replace stride multiplies with fixed offsets, ~5-10% gain on loads
/// 2. BlockMMA-style constructor that computes shared memory offsets once from simd_group_id
///    — eliminates per-load offset math, cleaner kernel code
/// 3. Frag::mma that keeps data as frag_type throughout a chain of MMAs without cast overhead
///    — avoids the float2→simdgroup_matrix→float2 round-trip when doing QK^T then score×V back-to-back
/// ──────────────────────────────────────────────────────────

#endif // SUPERKITTENS_TOOLS_TILE_H
