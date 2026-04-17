//
//  base.h
//  SuperKittens
//
//  Core 8×8 SIMD matrix types for Metal MMA.
//  Everything builds on simdgroup_{half,float}8x8.
//

#ifndef MEOW_MMA_BASE_H
#define MEOW_MMA_BASE_H

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

namespace meow {
namespace mma {

// ── Register tile: ROWS×COLS grid of 8×8 accumulators ────────
// Each simdgroup_float8x8 is one 8×8 block.
// A Tile<2,8> = 16×64 output region owned by one SIMD group.

template <int ROWS, int COLS>
struct Tile {
    static_assert(ROWS > 0 && COLS > 0, "Tile dims must be positive");
    simdgroup_float8x8 data[ROWS][COLS];

    METAL_FUNC void clear() {
        for (int r = 0; r < ROWS; r++)
            for (int c = 0; c < COLS; c++)
                data[r][c] = simdgroup_float8x8(0);
    }

    // Store accumulator to threadgroup memory as half
    METAL_FUNC void store(threadgroup half* dst, uint ld) const {
        for (int r = 0; r < ROWS; r++)
            for (int c = 0; c < COLS; c++)
                simdgroup_store(data[r][c], dst + r * 8 * ld + c * 8, ld);
    }

    // Store accumulator to device memory as half
    METAL_FUNC void store(device half* dst, uint ld) const {
        for (int r = 0; r < ROWS; r++)
            for (int c = 0; c < COLS; c++)
                simdgroup_store(data[r][c], dst + r * 8 * ld + c * 8, ld);
    }

    // Element-wise multiply with another tile (e.g. decay mask)
    METAL_FUNC void mul(const thread Tile& other) {
        for (int r = 0; r < ROWS; r++)
            for (int c = 0; c < COLS; c++) {
                // Extract, multiply, write back via thread_elements
                auto d = reinterpret_cast<thread float2&>(data[r][c].thread_elements());
                auto o = reinterpret_cast<const thread float2&>(other.data[r][c].thread_elements());
                d *= o;
            }
    }

    // Scale entire tile by scalar
    METAL_FUNC void scale(float s) {
        for (int r = 0; r < ROWS; r++)
            for (int c = 0; c < COLS; c++) {
                auto&\ d = reinterpret_cast<thread float2&>(data[r][c].thread_elements());
                d *= s;
            }
    }

    // Copy float32 acc → half in threadgroup (for feeding back into MMA as operand)
    METAL_FUNC void copy_to_half(threadgroup half* dst, uint ld) const {
        store(dst, ld);  // simdgroup_store already converts float→half
    }

    // Load from threadgroup half into this float32 accumulator
    METAL_FUNC void load(const threadgroup half* src, uint ld) {
        for (int r = 0; r < ROWS; r++)
            for (int c = 0; c < COLS; c++) {
                simdgroup_half8x8 tmp;
                simdgroup_load(tmp, src + r * 8 * ld + c * 8, ld);
                simdgroup_multiply_accumulate(data[r][c], tmp,
                    simdgroup_half8x8(1), simdgroup_float8x8(0));
            }
    }

    // Copy from another tile (same dimensions)
    METAL_FUNC void copy(const thread Tile& src) {
        for (int r = 0; r < ROWS; r++)
            for (int c = 0; c < COLS; c++) {
                auto d = reinterpret_cast<thread float2&>(data[r][c].thread_elements());
                auto s = reinterpret_cast<const thread float2&>(src.data[r][c].thread_elements());
                d = s;
            }
    }
    
    METAL_FUNC void copy_to_global(device half* dst, uint ld) const {
        
    }


};

} // namespace mma
} // namespace meow

#endif // MEOW_MMA_BASE_H
