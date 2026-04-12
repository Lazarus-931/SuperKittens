//
//  convert.h
//  SuperKittens
//
//  Type conversion utilities for threadgroup and register data.
//

#ifndef SUPERKITTENS_OPS_CONVERT_H
#define SUPERKITTENS_OPS_CONVERT_H

#include <metal_stdlib>
using namespace metal;

namespace superkittens {
/**
 * @namespace convert
 *
 * @brief Conversion namespace, where the conversion tools live, used throught SK
 */
namespace convert {



// float → half: N elements, all threads in threadgroup cooperate
METAL_FUNC void float_to_half(threadgroup half* dst,
                               const threadgroup float* src,
                               uint count, uint lid, uint n_threads) {
    for (uint i = lid; i < count; i += n_threads) {
        dst[i] = half(src[i]);
    }
}

// half → float
METAL_FUNC void half_to_float(threadgroup float* dst,
                               const threadgroup half* src,
                               uint count, uint lid, uint n_threads) {
    for (uint i = lid; i < count; i += n_threads) {
        dst[i] = float(src[i]);
    }
}


// Load float from device, store as half in threadgroup
METAL_FUNC void load_float_as_half(threadgroup half* dst,
                                    device const float* src,
                                    uint count, uint lid, uint n_threads) {
    for (uint i = lid; i < count; i += n_threads) {
        dst[i] = half(src[i]);
    }
}

// Load half from device, store as float in threadgroup
METAL_FUNC void load_half_as_float(threadgroup float* dst,
                                    device const half* src,
                                    uint count, uint lid, uint n_threads) {
    for (uint i = lid; i < count; i += n_threads) {
        dst[i] = float(src[i]);
    }
}



// Convert simdgroup_float8x8 accumulator to store as half to threadgroup
METAL_FUNC void acc_to_half(threadgroup half* dst, uint ld,
                             thread simdgroup_float8x8& acc) {
    simdgroup_store(acc, dst, ld);
}

// Load half from threadgroup into simdgroup_float8x8
METAL_FUNC void half_to_acc(thread simdgroup_float8x8& acc,
                             const threadgroup half* src, uint ld) {
    simdgroup_half8x8 tmp;
    simdgroup_load(tmp, src, ld);
    simdgroup_multiply_accumulate(acc, tmp, simdgroup_half8x8(1), simdgroup_float8x8(0));
}

// Generic copy with type conversion
template <typename Dst, typename Src>
METAL_FUNC void copy(threadgroup Dst* dst, const threadgroup Src* src,
                     uint count, uint lid, uint n_threads) {
    for (uint i = lid; i < count; i += n_threads)
        dst[i] = Dst(src[i]);
}

} // namespace convert
} // namespace superkittens

#endif // SUPERKITTENS_OPS_CONVERT_H
