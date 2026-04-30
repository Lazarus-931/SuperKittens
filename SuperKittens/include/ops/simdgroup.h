//
//  simdgroup.h
//  SuperKittens
//
//  By Alazar Manakelew
//
//  Cross-SIMD reductions and barrier helpers.
//

#ifndef MEOW_OPS_SIMDGROUP_H
#define MEOW_OPS_SIMDGROUP_H

#include <metal_stdlib>
using namespace metal;

namespace meow {
namespace ops {
namespace simdgroup {

inline float broadcast_first(float val) {
    return simd_broadcast_first(val);
}

inline float cross_simd_reduce_sum(
    float val,
    uint simd_id,
    uint lane_id,
    uint simd_count,
    threadgroup float* scratch)
{
    val = simd_sum(val);
    if (lane_id == 0) scratch[simd_id] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float out = 0.0f;
    if (simd_id == 0 && lane_id < simd_count) out = scratch[lane_id];
    out = simd_sum(out);
    if (simd_id == 0 && lane_id == 0) scratch[0] = out;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}

inline float cross_simd_reduce_max(
    float val,
    uint simd_id,
    uint lane_id,
    uint simd_count,
    threadgroup float* scratch)
{
    val = simd_max(val);
    if (lane_id == 0) scratch[simd_id] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float out = -INFINITY;
    if (simd_id == 0 && lane_id < simd_count) out = scratch[lane_id];
    out = simd_max(out);
    if (simd_id == 0 && lane_id == 0) scratch[0] = out;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}

inline float cross_simd_reduce_min(
    float val,
    uint simd_id,
    uint lane_id,
    uint simd_count,
    threadgroup float* scratch)
{
    val = simd_min(val);
    if (lane_id == 0) scratch[simd_id] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float out = INFINITY;
    if (simd_id == 0 && lane_id < simd_count) out = scratch[lane_id];
    out = simd_min(out);
    if (simd_id == 0 && lane_id == 0) scratch[0] = out;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}

inline float cross_simd_max(
    float val,
    uint simd_id,
    uint lid,
    threadgroup float* scratch)
{
    return cross_simd_reduce_max(val, simd_id, lid % 32, 4, scratch);
}

inline float cross_simd_sum(
    float val,
    uint simd_id,
    uint lid,
    threadgroup float* scratch)
{
    return cross_simd_reduce_sum(val, simd_id, lid % 32, 4, scratch);
}

inline float cross_simd_min(
    float val,
    uint simd_id,
    uint lid,
    threadgroup float* scratch)
{
    return cross_simd_reduce_min(val, simd_id, lid % 32, 4, scratch);
}

} // namespace simdgroup
} // namespace ops
} // namespace meow

#endif // MEOW_OPS_SIMDGROUP_H
