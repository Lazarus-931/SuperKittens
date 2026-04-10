//
//  global.h
//  SuperKittens
//
//  By Alazar Manakelew
//
//  Cross-SIMD reductions and barrier helpers.

#ifndef SUPERKITTENS_OPS_GLOBAL_H
#define SUPERKITTENS_OPS_GLOBAL_H

#include <metal_stdlib>
using namespace metal;

namespace superkittens {
namespace ops {

inline float cross_simd_max(
    float val,
    uint simd_id,
    uint lid,
    threadgroup float* scratch)
{
    val = simd_max(val);
    if (lid % 32 == 0) scratch[simd_id] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return max(max(scratch[0], scratch[1]), max(scratch[2], scratch[3]));
}

inline float cross_simd_sum(
    float val,
    uint simd_id,
    uint lid,
    threadgroup float* scratch)
{
    val = simd_sum(val);
    if (lid % 32 == 0) scratch[simd_id] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0] + scratch[1] + scratch[2] + scratch[3];
}

} // namespace ops
} // namespace superkittens

#endif // SUPERKITTENS_OPS_GLOBAL_H
