//
//  runtime/threadgroup.h — Threadgroup memory budget tracker
//  Compile-time-safe: static_assert if allocation exceeds 32KB.
//
//  Usage:
//    using TG = Threadgroup<16_KB>;  // 16KB used so far
//    using WithAs = TG::add<4_KB>;   // 20KB — compiles
//    using WithCs = WithAs::add<16_KB>;  // 36KB — static_assert fires
//

#ifndef SK_RUNTIME_THREADGROUP_H
#define SK_RUNTIME_THREADGROUP_H

#include <cstdint>

namespace sk::runtime {

// ── constants ──
inline constexpr uint32_t TG_LIMIT = 32768;  // 32KB — all Apple GPUs
inline constexpr uint32_t operator""_KB(unsigned long long n) { return (uint32_t)(n * 1024); }

// ── compile-time tracker ──
template<uint32_t Used>
struct Threadgroup {
    static constexpr uint32_t used  = Used;
    static constexpr uint32_t free  = TG_LIMIT - Used;
    static constexpr bool     valid = Used <= TG_LIMIT;

    template<uint32_t Extra>
    using add = Threadgroup<Used + Extra>;

    // Verify at compile time
    static_assert(valid, "Threadgroup memory exceeds 32KB limit");
};

// ── runtime helper: compute threadgroup memory for a kernel ──
struct TGUsage {
    uint32_t bytes;
    uint32_t percent;

    static TGUsage from_elements(uint32_t elements, uint32_t elem_size) {
        return { elements * elem_size, (elements * elem_size * 100) / TG_LIMIT };
    }
};

// ── common array sizes for kernel design ──
// Q tile [BM, d]:       BM * d * 2 bytes
// K tile [BN, d]:       BN * d * 2 bytes
// V tile [BN, d]:       BN * d * 2 bytes
// Scores [BM, BN]:      BM * BN * 4 bytes (float)
// Accumulator [BM, d]:  BM * d * 4 bytes (float)

} // namespace sk::runtime

#endif
