//
//  utils.metal
//  SuperKittens
//
//  Created by Alazar Manakelew on 4/7/26.
//

#ifndef SUPERKITTENS_UTILS_H
#define SUPERKITTENS_UTILS_H

#include <metal_stdlib>
using namespace metal;

/////////////////////////////////////////////////////////////////////////////////////////////
/// Helper macros
/////////////////////////////////////////////////////////////////////////////////////////////

// Thread/SIMD identity — saves repeating these casts everywhere
#define SK_TID_SETUP \
    uint lid      = thread_index_in_threadgroup; \
    uint simd_id  = simdgroup_index_in_threadgroup; \
    uint lane_id  = thread_index_in_simdgroup;

// Cooperative load: all threads in a threadgroup load N elements from src → dst
// Handles arbitrary counts with stride = total threads in TG
#define SK_COOP_LOAD(dst, dst_stride, src, src_ld, rows, cols, lid, n_threads) \
    for (uint _i = lid; _i < (rows) * (cols); _i += (n_threads)) { \
        uint _r = _i / (cols), _c = _i % (cols); \
        (dst)[_r * (dst_stride) + _c] = (src)[_r * (src_ld) + _c]; \
    }

// Cooperative store: all threads write from threadgroup → device
#define SK_COOP_STORE(dst, dst_ld, src, src_stride, rows, cols, lid, n_threads) \
    for (uint _i = lid; _i < (rows) * (cols); _i += (n_threads)) { \
        uint _r = _i / (cols), _c = _i % (cols); \
        (dst)[_r * (dst_ld) + _c] = (src)[_r * (src_stride) + _c]; \
    }

/////////////////////////////////////////////////////////////////////////////////////////////
/// Next:
/// - SK_COOP_LOAD_TRANSPOSED: cooperative transposed load (for K^T in attention, etc.)
/// - SK_BARRIER: threadgroup_barrier(mem_flags::mem_threadgroup) shorthand
/// - SK_SIMD_REDUCE_MAX / SK_SIMD_REDUCE_SUM: cross-SIMD reductions via scratch
/// - SK_BOUNDS_CHECK: safe load with bounds (for edge tiles at seq/dim boundaries)
/////////////////////////////////////////////////////////////////////////////////////////////

#define EXP(x) metal::fast::exp(x)

#endif // SUPERKITTENS_UTILS_H
