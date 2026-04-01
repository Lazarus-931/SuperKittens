//
//  fp16_m2_gemm.metal
//  SuperKittens
//
//  Created by Alazar Manakelew on 4/1/26.
//

#include <metal_stdlib>

using namespace metal;

int const BATCH_SIZE = 32;

// Tile configuration for FP16 GEMM on M2
// MB: tile rows, NB: tile cols, KB: k-dimension tile
template<int MB, int NB, int KB>
struct fp16_gemm_config {
    static_assert(MB == 256, "MB needs to be 256");
    static_assert(NB >= 16 && NB <= 256 && NB % 16 == 0, "NB needs to be 16, 32...256");
    static_assert(KB >= 16 && KB % 16 == 0, "KB needs to be 16,...128,...512...");
};

kernel void fp16_gemm(
    device const half*  A       [[buffer(0)]],
    device const half*  B       [[buffer(1)]],
    device half*        C       [[buffer(2)]],
    constant uint&      M       [[buffer(3)]],
    constant uint&      N       [[buffer(4)]],
    constant uint&      K       [[buffer(5)]],
    uint2 gid                   [[thread_position_in_grid]],
    uint2 tid                   [[thread_position_in_threadgroup]]
) {

    if (M > N) {
        
    }

}


<template <typename C>>
kernel void fp16_gemm1(
    device const half*  A
    device const half*  B
    device half*        C
    constant uint&      M
    constant unit&      N
    constant unit&      K
)



int main() {
    int N;


}

