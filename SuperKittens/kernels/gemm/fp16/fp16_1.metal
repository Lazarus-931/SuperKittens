//
//  fp16_1.metal
//  SuperKittens
//
//  Created by Alazar Manakelew on 4/1/26.
//

#include <metal_stdlib>
#include "types.h"

using namespace metal;

// Uses fp16_1_config from types.metal: BM=32, BN=32, BK=16, WM=2, WN=2
// 128 threads (4 simdgroups), threadgroup memory: 2 KB

kernel void fp16_1_gemm(
    device const half*  A       [[buffer(0)]],
    device const half*  B       [[buffer(1)]],
    device half*        C       [[buffer(2)]],
    constant uint&      M       [[buffer(3)]],
    constant uint&      N       [[buffer(4)]],
    constant uint&      K       [[buffer(5)]],
    uint2 tid                   [[thread_position_in_threadgroup]],
    uint2 tgid                  [[threadgroup_position_in_grid]],
    uint  simd_id               [[simdgroup_index_in_threadgroup]],
    uint  simd_lane             [[thread_index_in_simdgroup]]
) {

    constexpr uint BM = fp16_1_config::BM;
    constexpr uint BN = fp16_1_config::BN;
    constexpr uint BK = fp16_1_config::BK;
    constexpr uint THREADS = fp16_1_config::WM * fp16_1_config::WN * 32;


    threadgroup half As[BM * BK]; // 32x16
    threadgroup half Bs[BK * BN]; // 16x32
    
    
    
    
    // allocates space in unifed memory
    
     

}
