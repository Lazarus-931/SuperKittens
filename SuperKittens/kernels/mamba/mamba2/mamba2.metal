//
//  fwd.metal
//  SuperKittens
//
//  Mamba-2 SSD forward kernel (chunked matmul approach).
//  Follows ThunderKittens' algorithm: cumsum → decay → intra-chunk attn → inter-chunk state.
//

#include "../../../meow.h"
#include "../mamba_impl.h"



using namespace meow::mamba;

METAL_FUNC float a_floor() { return 1e-5f; }


namespace meow::mamba::mamba2 {

template<bool bi_directional>
[[kernel, max_total_threads_per_threadgroup(N_THREADS)]]
void mamba2_ssm(
    device const half* Q       [[buffer(0)]],   // [B, 1, N, D]
    device const half* K       [[buffer(1)]],   // [B, 1, N, D]
    device const half* V       [[buffer(2)]],   // [B, H, N, D]
    device const float* A      [[buffer(3)]],   // [B, H, N]
    device half* O             [[buffer(4)]],   // [B, H, N, D]
    constant Mamba2Params& p   [[buffer(5)]],
    uint3 gid    [[threadgroup_position_in_grid]],
    uint  lid    [[thread_index_in_threadgroup]],
    uint  simd_id [[simdgroup_index_in_threadgroup]],
    uint  lane_id [[thread_index_in_simdgroup]])
{
    
    const int batch = gid.x;
    const int head  = gid.y;
    const int chunk = gid.z;
    const int seq_offset = chunk * CHUNK_SIZE;

    
    threadgroup float a_cumsum[CHUNK_SIZE];            // 256 b
    threadgroup float cumsum_scratch[CHUNK_SIZE / 32];
    threadgroup half  Qs[CHUNK_SIZE * HEAD_DIM];       // 8 kb
    threadgroup half  Ks[CHUNK_SIZE * HEAD_DIM];
    threadgroup half  Vs[CHUNK_SIZE * HEAD_DIM];
    threadgroup half  kv_state[HEAD_DIM * HEAD_DIM];   // 8 kb (was register Tile; float regs still used for accum)
    
    

    device const float* A_chunk = A + batch * p.a_batch
                                    + head  * p.a_head
                                    + seq_offset;

    if (lid < CHUNK_SIZE) a_cumsum[lid] = (seq_offset + lid < p.seq) ? A_chunk[lid] : 0.0f;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
                                                    
    meow::tools::threadgroup_cumsum<float, CHUNK_SIZE>(
        a_cumsum, cumsum_scratch, lid, lane_id, simd_id);
    
    threadgroup_barrier(mem_flags::mem_threadgroup);

   
    uint row_base = simd_id * 16;  // each SIMD with 16 rows

    
    meow::mma::Tile<2, 8> attn_block;
    meow::mma::Tile<2, 8> o_reg;

    // Clear kv_state cooperatively (threadgroup half, 4096 halfs / 128 threads = 32 per thread)
    for (uint i = lid; i < HEAD_DIM * HEAD_DIM; i += N_THREADS) {
        kv_state[i] = half(0);
    }

    // load up q into registers
    meow::mma::Tile<2, 8> q_reg;
    q_reg.load(Qs + row_base * HEAD_DIM, HEAD_DIM);

   
    attn_block.clear();
    meow::mma::mm_ABt<HEAD_DIM, 2, 8>(attn_block, Qs + row_base * HEAD_DIM, HEAD_DIM, Ks, HEAD_DIM);
    meow::tools::apply_causal_decay(attn_block, a_cumsum, row_base, lane_id);

    uint qid = lane_id / 4;
    uint local_row = (qid & 4) + (lane_id / 2) % 4;
    
    for (int r = 0; r < 2; r++) {
        uint abs_row = row_base + r * 8 + local_row;
        float decay = exp(a_cumsum[abs_row]);
        for (int c = 0; c < 8; c++) {
            auto d = reinterpret_cast<thread float2&>(q_reg.data[r][c].thread_elements());
            d *= decay;
        }
    }
    
    // Store scaled q back to threadgroup; kv_state already lives in threadgroup memory
    q_reg.copy_to_half(Qs + row_base * HEAD_DIM, HEAD_DIM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    meow::mma::mm_AB<HEAD_DIM, 2, 8>(o_reg, Qs + row_base * HEAD_DIM, HEAD_DIM, kv_state, HEAD_DIM);

    attn_block.copy_to_half(Ks + row_base * CHUNK_SIZE, CHUNK_SIZE);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    meow::mma::mm_AB<CHUNK_SIZE, 2, 8>(o_reg, Ks + row_base * CHUNK_SIZE, CHUNK_SIZE, Vs, HEAD_DIM);

    o_reg.store(O + (batch * p.o_batch + head * p.o_head + (seq_offset + row_base) * HEAD_DIM), HEAD_DIM);
    
    
    
}



inline bool dispatch_ssm(
    int B, int H, int N, int D,
    thread const int* q_strides,
    thread const int* k_strides,
    thread const int* v_strides,
    thread const int* o_strides,
    thread const int* a_strides,
    thread const int* b_strides)
{
    // Q, K, V, O are [B, H, N, D]; A, B are [B, H, N]
    int qkvo_shape[4] = { B, H, N, D };
    int ab_shape[3]   = { B, H, N };

    if (!meow::tools::is_contiguous<4>(qkvo_shape, q_strides)) return false;
    if (!meow::tools::is_contiguous<4>(qkvo_shape, k_strides)) return false;
    if (!meow::tools::is_contiguous<4>(qkvo_shape, v_strides)) return false;
    if (!meow::tools::is_contiguous<4>(qkvo_shape, o_strides)) return false;
    if (!meow::tools::is_contiguous<3>(ab_shape,   a_strides)) return false;
    if (!meow::tools::is_contiguous<3>(ab_shape,   b_strides)) return false;

    return true;
}

} // namespace meow::mamba::mamba2
