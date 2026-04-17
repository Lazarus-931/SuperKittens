//
//  math.h
//  SuperKittens
//
//  By Alazar Manakelew
//
//  Fast math ops: exp, rsqrt, branchless select.

#ifndef MEOW_OPS_MATH_H
#define MEOW_OPS_MATH_H

#include <metal_stdlib>
using namespace metal;

namespace meow {
namespace ops {

inline float fast_exp(float x) {
    return metal::fast::exp(x);
}

inline float fast_rsqrt(float x) {
    return metal::fast::rsqrt(x);
}

inline float safe_load(float val, bool in_bounds) {
    return select(0.0f, val, in_bounds);
}

inline float sqrt(float x) {
    return metal::fast::sqrt(x);
}

} // namespace ops
} // namespace meow

#endif // MEOW_OPS_MATH_H
