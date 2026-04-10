//
//  bwd.metal
//  SuperKittens
//
//  Created by Alazar Manakelew on 4/6/26.
//

#include "mamba_impl.h"

using namespace superkittens::mamba;

template<typename T, int N_THREADS, int N_ITEMS, int N_ROWS,
         bool kIsEvenLen, bool kIsVariableB, bool kIsVariableC,
         bool kHasZ, bool kIsComplex>
[[kernel, max_total_threads_per_threadgroup(N_THREADS)]]
void ss_bwd_kernel(
    device const T* x       [[buffer(0)]],
    device const T* A       [[buffer(1)]],
    device const T* B       [[buffer(2)]],
    device const T* C       [[buffer(3)]],
    device const T* delta   [[buffer(4)]],
    device const T* z       [[buffer(5)]],
    device const T* dout    [[buffer(6)]],
    device T* dx            [[buffer(7)]],
    device T* dA            [[buffer(8)]],
    device T* dB            [[buffer(9)]],
    device T* dC            [[buffer(10)]],
    device T* ddelta        [[buffer(11)]],
    device T* dz            [[buffer(12)]],
    constant BwdParams& params [[buffer(13)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
}
