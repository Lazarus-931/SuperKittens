//
//  tile.h — SuperKittens DSL: threadgroup tile primitives
//
//  Tile<T,Rows,Cols> — compile-time-sized threadgroup tile.
//  Auto half4 vectorization.  Metal 3.1 compatible (enums, pointers).
//

#pragma once

#include <metal_stdlib>

using namespace metal;

namespace sk {
namespace dsl {

// ── Threadgroup tile ──────────────────────────────────────────────

template<typename T, int Rows_, int Cols_>
struct Tile {
    enum : int { Rows = Rows_, Cols = Cols_, Elements = Rows * Cols,
                 SizeBytes = Elements * (int)sizeof(T) };

    T data[Rows * Cols];
};

template<int Rows, int Cols> using Fp16Tile = Tile<half, Rows, Cols>;
template<int Rows, int Cols> using Fp32Tile = Tile<float, Rows, Cols>;

// ── Budget check (variadic template, no fold expressions) ─────────

template<int Total, int... Rest>
struct BudgetSum;

template<int Total>
struct BudgetSum<Total> { enum : int { value = Total }; };

template<int Total, int Head, int... Tail>
struct BudgetSum<Total, Head, Tail...> {
    enum : int { value = BudgetSum<Total + Head, Tail...>::value };
};

template<int... Bytes>
constexpr int tg_budget() {
    // 32768 bytes = universal Apple Silicon threadgroup memory limit
    static_assert(BudgetSum<0, Bytes...>::value <= 32768,
        "threadgroup memory exceeds 32KB limit");
    return BudgetSum<0, Bytes...>::value;
}

// ── Half4-vectorized load (pass by pointer — Metal rule) ──────────

template<int Rows, int Cols, int N_THREADS>
void load_tile(
    device const half* src, uint ld,
    uint row_start, uint col_start,
    uint max_rows, uint max_cols,
    threadgroup Tile<half, Rows, Cols>* dst,
    uint lid)
{
    static_assert(Cols % 4 == 0, "half4 load needs Cols %% 4 == 0");
    constexpr uint C4 = Cols / 4;
    constexpr uint total = Rows * C4;

    const device half4* src4 = reinterpret_cast<const device half4*>(src);
    threadgroup half4* dst4  = reinterpret_cast<threadgroup half4*>(dst->data);

    for (uint i = lid; i < total; i += N_THREADS) {
        uint r = i / C4, c4 = i % C4;
        uint gr = row_start + r, gc = col_start + c4 * 4;
        dst4[i] = (gr < max_rows && gc < max_cols)
            ? src4[(gr * ld + gc) / 4]
            : half4(0.0h);
    }
}

// ── Half4-vectorized store ────────────────────────────────────────

template<int Rows, int Cols, int N_THREADS>
void store_tile(
    device half* dst, uint ld,
    uint row_start, uint col_start,
    uint max_rows, uint max_cols,
    const threadgroup Tile<half, Rows, Cols>* src,
    uint lid)
{
    static_assert(Cols % 4 == 0, "half4 store needs Cols %% 4 == 0");
    constexpr uint C4 = Cols / 4;
    constexpr uint total = Rows * C4;

    device half4* dst4 = reinterpret_cast<device half4*>(dst);
    const threadgroup half4* src4 = reinterpret_cast<const threadgroup half4*>(src->data);

    for (uint i = lid; i < total; i += N_THREADS) {
        uint r = i / C4, c4 = i % C4;
        uint gr = row_start + r, gc = col_start + c4 * 4;
        if (gr < max_rows && gc < max_cols) dst4[(gr * ld + gc) / 4] = src4[i];
    }
}

} // namespace dsl
} // namespace sk
