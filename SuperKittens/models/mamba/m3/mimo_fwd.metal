//
//  mimo_fwd.metal
//  SuperKittens
//
//  Created by Alazar Manakelew on 4/13/26.
//


#include <metal_stdlib>
#include <metal/metal_math>




template<int HEAD_DIM_QK, int CHUNK_SIZE, int HEAD_DIM_V>
[[kernel]] void Mamba3MIMO_Kernel(
                               device const half* q [[buffer(0)]],
                               device const half* k [[buffer(1)]],
                               device const half* v [[buffer(2)]],
                               device const half* a [[buffer(3)]],
                               device const half* b [[buffer(4)]],
                               device const half* angle [[buffer(5)]],
                               const int seqlen,
                               const int d_head,
                               const int n_heads)

{
    int batch_id
    int seqlen, d_head, n_heads = q.
    
    int offset = batch_id * (seqlen * d_head * n_heads)
        +
    
    threadgroup As[];
    threadgroup Bs[];
    
}
