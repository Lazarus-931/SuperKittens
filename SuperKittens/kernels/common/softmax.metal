//
//  softmax.metal
//  SuperKittens
//
//  Created by Alazar Manakelew on 4/3/26.
//

#include <metal_stdlib>
using namespace metal;

/////////////////////////////////////////////////////////////////////////////////////////////
/// Softmax
/////////////////////////////////////////////////////////////////////////////////////////////

METAL_FUNC softmax_threadgroup(
                               device const float *src [[buffer(0)]],
                               device float *dst [[buffer(1)]],
                               constant uint rows [[buffer(2)]],
                               constant uint cols [[buffer(3)]]
                               uint row [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
    
}
