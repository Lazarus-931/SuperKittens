//
//  mamba3.metal
//  SuperKittens
//
//  Created by Alazar Manakelew on 4/13/26.
//

#include <metal_stdlib>


using namespace meow::mamba;

namespace meow::models::m3 {

template<int CHUNK, int HD_QK, int HD_V, int RANK>
struct mimo_fwd {

    static constexpr int CHUNK_SIZE = CHUNK;
    static constexpr int HEAD_DIM_QK = HD_QK;
    static constexpr int HEAD_DIM_V = HD_V;
    static constexpr int MIMO_RANK = RANK;
    static constexpr int N_THREADS = 256;
    static constexpr int TILE_M = 16;
    static constexpr int TILE_N = HD_V;

    static_assert(CHUNK_SIZE % TILE_M == 0);
    static_assert(HEAD_DIM_QK % 16 == 0);
    static_assert(MIMO_RANK >= 1);
    static_assert(N_THREADS % 128 == 0);

    struct layout {
        static constexpr int Q_SIZE = CHUNK_SIZE * HEAD_DIM_QK * MIMO_RANK;
        static constexpr int K_SIZE = CHUNK_SIZE * HEAD_DIM_QK * MIMO_RANK;
        static constexpr int V_SIZE = CHUNK_SIZE * HEAD_DIM_V;
        static constexpr int A_SIZE = CHUNK_SIZE;
        static constexpr int B_SIZE = CHUNK_SIZE;
        static constexpr int ANGLE_SIZE = CHUNK_SIZE * (HEAD_DIM_QK / 2);
    };


    struct load {
        
        static void tiles(threadgroup half* Qs, threadgroup half* Ks,
                          threadgroup half* Vs, threadgroup float* As,
                          threadgroup float* Bs, threadgroup half* angles,
                          device const half* Q, device const half* K,
                          device const half* V, device const float* A,
                          device const float* B, device const half* angle,
                          const int batch, const int head, const int chunk,
                          uint lid) {
            
            for (uint i = lid; i < K_SIZE; i += N_THREADS) {
                Ks[i] = K[k_offset + i];
                Qs[i] = Q[q_offset + i];
            }
            
            for (uint i = lid; i < A_SIZE; i += N_THREADS) {
                As[i] = A[a_offset + i];
                Bs[i] = B[b_offset + i];
            }
                                                                                                                                    
            for (uint i = lid; i < V_SIZE; i += N_THREADS)
                Vs[i] = V[v_offset + i];
                                                                  
            for (uint i = lid; i < ANGLE_SIZE; i += N_THREADS)
                angles[i] = angle[angle_offset + i];
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        
        
        
    };

    struct compute {
        static void intra_chunk(threadgroup half* Qs, threadgroup half* Ks,
                                threadgroup half* Vs, threadgroup float* As,
                                threadgroup float* Bs, threadgroup half* angles,
                                uint simd_id, uint lane_id) {
            
        }

        static void inter_chunk(threadgroup half* Qs, threadgroup half* Ks,
                                threadgroup half* Vs, threadgroup float* state,
                                threadgroup float* As, threadgroup float* Bs,
                                threadgroup half* angles,
                                uint simd_id, uint lane_id) {
            
        }
    };

   
    struct store {
        static void output(device half* O, threadgroup half* out,
                           uint lid) {
            // TODO: write results to device memory
        }
    };
};

} // namespace meow::models::m3
