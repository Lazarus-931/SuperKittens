//
//  mamba3.metal
//  SuperKittens
//
//  Created by Alazar Manakelew on 4/13/26.
//

#include "../../../meow.h"

using namespace meow::mamba;

namespace meow::mamba::mamba3 {

template<int CHUNK, int HD_QK, int HD_V>
struct mamba3_fwd {
    
    // metal is very annoying, no struct, enums instead
    enum : int {
        CHUNK_SIZE = CHUNK,
        HEAD_DIM_QK = HD_QK,
        HEAD_DIM_V = HD_V,
        N_THREADS = 256,
        TILE_M = 16
    };
    
    
    static_assert(CHUNK_SIZE % TILE_M == 0, "CHUNK_SIZE must be multiple of TILE_M");
    static_assert(HEAD_DIM_QK % 16 == 0, "HEAD_DIM_QK must be multiple of 16");
    static_assert(N_THREADS % 128 == 0, "N_THREADS must be multiple of 128");

    struct shared_layout {
        enum : int {
            A_SIZE = CHUNK_SIZE,
            B_SIZE = CHUNK_SIZE,
            ANGLE_SIZE = CHUNK_SIZE * (HEAD_DIM_QK / 2),
            CUMSUM_SCRATCH_SIZE = CHUNK_SIZE / 32,
            DECAY_SIZE = CHUNK_SIZE
        };
    };

    struct common_state {
        int batch, head, chunk;
        int a_offset, b_offset, angle_offset;
    };
    
    ///  ----- mamba3 specific --------
    static void setup_common(thread common_state& cs, uint3 gid,
                             int seq_len, int n_heads) {
        cs.batch = gid.x;
        cs.head = gid.y;
        cs.chunk = gid.z;

        const int seq_offset = cs.chunk * CHUNK_SIZE;
        cs.a_offset = (cs.batch * n_heads + cs.head) * seq_len + seq_offset;
        cs.b_offset = cs.a_offset;
        cs.angle_offset = (cs.batch * n_heads + cs.head) * seq_len * (HEAD_DIM_QK / 2)
                        + seq_offset * (HEAD_DIM_QK / 2);
    }

    static void load_common(device const float* A, device const float* B_trap,
                            device const half* angle,
                            const thread common_state& cs,
                            threadgroup float* As, threadgroup float* Bs,
                            threadgroup half* angles, uint lid) {
        for (uint i = lid; i < shared_layout::A_SIZE; i += N_THREADS) {
            As[i] = A[cs.a_offset + i];
            Bs[i] = B_trap[cs.b_offset + i];
        }

        for (uint i = lid; i < shared_layout::ANGLE_SIZE; i += N_THREADS) {
            angles[i] = angle[cs.angle_offset + i];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /// 1. helper funcs related to m3 specific trap discredization
    static float trap_scale(float a, float b) {
        return 1.0f + b * metal::fast::exp(-a);
    }
    
    static void build_trap_scale(threadgroup float* A,
                                 threadgroup float* B,
                                 threadgroup float* trap_scale,
                                 uint lid) {
        for (uint i = lid; i < CHUNK_SIZE; i += N_THREADS) {
            trap_scale[i] = mamba3_fwd::trap_scale(A[i], B[i]);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
    }
    
    static void apply_trapezoidal_scale(threadgroup float* A,
                                        threadgroup float* B_scale,
                                        threadgroup float* local_decay,
                                        uint lid) {
        for (uint i = lid; i < CHUNK_SIZE; i += N_THREADS) {
            local_decay[i] = fast::exp(A[i]) * B_scale[i];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    /// 2. helper funcs related to m3 specific rotary embedding
    static float2 rotary_trig(float angle) {
        return float2(metal::fast::cos(angle), metal::fast::sin(angle));
    }

    
    static void def_rotary_angle(threadgroup half* angles,
                                 threadgroup float* As,
                                 threadgroup float* angle_state,
                                 uint lid) {
        constexpr int HALF_DIM = HEAD_DIM_QK / 2;
        for (uint i = lid; i < shared_layout::ANGLE_SIZE; i += N_THREADS) {
            uint t = i / HALF_DIM;
            uint p = i % HALF_DIM;
            float raw = float(angles[i]);
            angles[i] = half(angle_state[p] + As[t] * raw * 3.141592653589793f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }


    static void apply_rotary_qk(threadgroup half* Q,
                                threadgroup half* K,
                                threadgroup half* angles,
                                uint lid) {
        constexpr int HALF_DIM = HEAD_DIM_QK / 2;
        constexpr int TOTAL_PAIRS = CHUNK_SIZE * HALF_DIM;

        for (uint i = lid; i < TOTAL_PAIRS; i += N_THREADS) {
            uint t = i / HALF_DIM;
            uint p = i % HALF_DIM;

            float2 cs = rotary_trig(float(angles[i]));
            float c = cs.x;
            float s = cs.y;

            uint d0 = t * HEAD_DIM_QK + p;
            uint d1 = d0 + HALF_DIM;

            float q0 = float(Q[d0]);
            float q1 = float(Q[d1]);
            Q[d0] = half(q0 * c - q1 * s);
            Q[d1] = half(q0 * s + q1 * c);

            float k0 = float(K[d0]);
            float k1 = float(K[d1]);
            K[d0] = half(k0 * c - k1 * s);
            K[d1] = half(k0 * s + k1 * c);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    
    struct siso {
        struct layout {
            enum : int {
                Q_SIZE = CHUNK_SIZE * HEAD_DIM_QK,
                K_SIZE = CHUNK_SIZE * HEAD_DIM_QK,
                V_SIZE = CHUNK_SIZE * HEAD_DIM_V
            };
        };

        struct state {
            int q_offset, k_offset, v_offset;
            int o_offset;
        };

        static void setup(thread state& st, const thread common_state& cs,
                          int seq_len, int n_heads) {
            const int seq_offset = cs.chunk * CHUNK_SIZE;
            st.q_offset = (cs.batch * n_heads + cs.head) * seq_len * HEAD_DIM_QK
                        + seq_offset * HEAD_DIM_QK;
            st.k_offset = st.q_offset;
            st.v_offset = (cs.batch * n_heads + cs.head) * seq_len * HEAD_DIM_V
                        + seq_offset * HEAD_DIM_V;
            st.o_offset = st.v_offset;
        }

        static void load(device const half* Q, device const half* K,
                         device const half* V, const thread state& st,
                         threadgroup half* Qs, threadgroup half* Ks,
                         threadgroup half* Vs, uint lid) {
            for (uint i = lid; i < layout::Q_SIZE; i += N_THREADS)
                Qs[i] = Q[st.q_offset + i];

            for (uint i = lid; i < layout::K_SIZE; i += N_THREADS)
                Ks[i] = K[st.k_offset + i];

            for (uint i = lid; i < layout::V_SIZE; i += N_THREADS)
                Vs[i] = V[st.v_offset + i];

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        static void intra_chunk(threadgroup half* Qs, threadgroup half* Ks,
                                threadgroup half* Vs, threadgroup float* As,
                                threadgroup half* scratch,
                                uint simd_id, uint lane_id) {
            enum : int {
                TILE_ROWS = TILE_M / 8,
                QK_COLS = CHUNK_SIZE / 8,
                V_COLS = HEAD_DIM_V / 8
            };

            uint row_base = simd_id * TILE_M;

            meow::mma::Tile<TILE_ROWS, QK_COLS> attn_block;
            attn_block.clear();
            meow::mma::mm_ABt<HEAD_DIM_QK, TILE_ROWS, QK_COLS>(
                attn_block, Qs + row_base * HEAD_DIM_QK, HEAD_DIM_QK,
                Ks, HEAD_DIM_QK);

            meow::tools::apply_causal_decay(attn_block, As, row_base, lane_id);

            attn_block.copy_to_half(scratch + row_base * CHUNK_SIZE, CHUNK_SIZE);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            meow::mma::Tile<TILE_ROWS, V_COLS> o_reg;
            o_reg.clear();
            meow::mma::mm_AB<CHUNK_SIZE, TILE_ROWS, V_COLS>(
                o_reg, scratch + row_base * CHUNK_SIZE, CHUNK_SIZE,
                Vs, HEAD_DIM_V);

            o_reg.store(Vs + row_base * HEAD_DIM_V, HEAD_DIM_V);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        static void inter_chunk(threadgroup half* Qs, threadgroup half* Ks,
                                threadgroup half* Vs, threadgroup float* local_decay,
                                threadgroup half* kv_state,
                                uint lid, uint simd_id, uint lane_id) {
            enum : int {
                TILE_ROWS = TILE_M / 8,
                V_COLS = HEAD_DIM_V / 8,
                QK_COLS = HEAD_DIM_QK / 8,
                STATE_SIZE = HEAD_DIM_QK * HEAD_DIM_V
            };

            uint row_base = simd_id * TILE_M;
            uint qid = lane_id / 4;
            uint local_row = (qid & 4) + (lane_id / 2) % 4;

            meow::mma::Tile<TILE_ROWS, QK_COLS> q_reg;
            q_reg.load(Qs + row_base * HEAD_DIM_QK, HEAD_DIM_QK);

            for (int r = 0; r < TILE_ROWS; r++) {
                uint abs_row = row_base + r * 8 + local_row;
                float decay = local_decay[abs_row];
                for (int c = 0; c < QK_COLS; c++) {
                    auto d = reinterpret_cast<thread float2&>(
                        q_reg.data[r][c].thread_elements());
                    d *= decay;
                }
            }

            q_reg.copy_to_half(Qs + row_base * HEAD_DIM_QK, HEAD_DIM_QK);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            meow::mma::Tile<TILE_ROWS, V_COLS> inter_out;
            inter_out.clear();
            meow::mma::mm_AB<HEAD_DIM_QK, TILE_ROWS, V_COLS>(
                inter_out, Qs + row_base * HEAD_DIM_QK, HEAD_DIM_QK,
                kv_state, HEAD_DIM_V);

            meow::mma::Tile<TILE_ROWS, V_COLS> intra_out;
            intra_out.load(Vs + row_base * HEAD_DIM_V, HEAD_DIM_V);
            for (int r = 0; r < TILE_ROWS; r++) {
                for (int c = 0; c < V_COLS; c++) {
                    auto d = reinterpret_cast<thread float2&>(
                        inter_out.data[r][c].thread_elements());
                    auto s = reinterpret_cast<thread float2&>(
                        intra_out.data[r][c].thread_elements());
                    d += s;
                }
            }

            inter_out.store(Vs + row_base * HEAD_DIM_V, HEAD_DIM_V);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (simd_id == 0) {
                float chunk_decay = local_decay[CHUNK_SIZE - 1];
                for (uint i = lid; i < STATE_SIZE; i += 32) {
                    kv_state[i] = half(float(kv_state[i]) * chunk_decay);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                meow::mma::Tile<QK_COLS, V_COLS> state_update;
                state_update.clear();
                meow::mma::mm_AtB<CHUNK_SIZE, QK_COLS, V_COLS>(
                    state_update, Ks, HEAD_DIM_QK, Vs, HEAD_DIM_V);
                meow::mma::Tile<QK_COLS, V_COLS> existing;
                existing.load(kv_state, HEAD_DIM_V);
                for (int r = 0; r < QK_COLS; r++) {
                    for (int c = 0; c < V_COLS; c++) {
                        auto d = reinterpret_cast<thread float2&>(
                            state_update.data[r][c].thread_elements());
                        auto e = reinterpret_cast<thread float2&>(
                            existing.data[r][c].thread_elements());
                        d += e;
                    }
                }
                state_update.store(kv_state, HEAD_DIM_V);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        static void compute(threadgroup half* Qs, threadgroup half* Ks,
                            threadgroup half* Vs, threadgroup float* As,
                            threadgroup float* Bs,
                            threadgroup half* angles,
                            threadgroup float* angle_state,
                            threadgroup float* cumsum_scratch,
                            threadgroup float* local_decay,
                            threadgroup half* scratch,
                            threadgroup half* kv_state,
                            uint lid, uint simd_id, uint lane_id) {
            meow::tools::threadgroup_cumsum<float, CHUNK_SIZE>(
                As, cumsum_scratch, lid, lane_id, simd_id);

            build_trap_scale(As, Bs, Bs, lid);
            apply_trapezoidal_scale(As, Bs, local_decay, lid);
            def_rotary_angle(angles, As, angle_state, lid);
            apply_rotary_qk(Qs, Ks, angles, lid);

            intra_chunk(Qs, Ks, Vs, As, scratch, simd_id, lane_id);
            inter_chunk(Qs, Ks, Vs, local_decay, kv_state, lid, simd_id, lane_id);
        }

        // writeback output
        static void finish(device half* O, const thread state& st,
                           threadgroup half* Vs, uint lid) {
            for (uint i = lid; i < layout::V_SIZE; i += N_THREADS)
                O[st.o_offset + i] = Vs[i];
        }
    };

    template<int RANK, int DIM>
    struct mimo {
        enum : int {
            MIMO_DIM = DIM,
            MIMO_RANK = RANK
        };

        static_assert(MIMO_RANK >= 1, "MIMO_RANK must be >= 1");

        struct layout {
            enum : int {
                Q_SIZE = CHUNK_SIZE * HEAD_DIM_QK * MIMO_RANK,
                K_SIZE = CHUNK_SIZE * HEAD_DIM_QK * MIMO_RANK,
                V_SIZE = CHUNK_SIZE * HEAD_DIM_V
            };
        };

        struct state {
            int q_offset, k_offset, v_offset;
            int o_offset;
        };

        static void setup(thread state& st, const thread common_state& cs,
                          int seq_len, int n_heads) {
            const int seq_offset = cs.chunk * CHUNK_SIZE;
            st.q_offset = (cs.batch * n_heads + cs.head) * seq_len * HEAD_DIM_QK * MIMO_RANK
                        + seq_offset * HEAD_DIM_QK * MIMO_RANK;
            st.k_offset = st.q_offset;
            st.v_offset = (cs.batch * n_heads + cs.head) * seq_len * HEAD_DIM_V
                        + seq_offset * HEAD_DIM_V;
            st.o_offset = st.v_offset;
        }

        static void load(device const half* Q, device const half* K,
                         device const half* V, const thread state& st,
                         threadgroup half* Qs, threadgroup half* Ks,
                         threadgroup half* Vs, uint lid) {
            for (uint i = lid; i < layout::Q_SIZE; i += N_THREADS)
                Qs[i] = Q[st.q_offset + i];

            for (uint i = lid; i < layout::K_SIZE; i += N_THREADS)
                Ks[i] = K[st.k_offset + i];

            for (uint i = lid; i < layout::V_SIZE; i += N_THREADS)
                Vs[i] = V[st.v_offset + i];

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        
        
        static void intra_chunk(threadgroup half* Qs, threadgroup half* Ks,
                                threadgroup half* Vs, threadgroup float* As,
                                threadgroup half* scratch,
                                uint simd_id, uint lane_id) {
            enum : int {
                TILE_ROWS = TILE_M / 8,
                QK_COLS = CHUNK_SIZE / 8,
                V_COLS = HEAD_DIM_V / 8,
                FLAT_QK_DIM = HEAD_DIM_QK * MIMO_RANK
            };

            uint row_base = simd_id * TILE_M;

            meow::mma::Tile<TILE_ROWS, QK_COLS> attn_block;
            attn_block.clear();
            meow::mma::mm_ABt<FLAT_QK_DIM, TILE_ROWS, QK_COLS>(
                attn_block, Qs + row_base * FLAT_QK_DIM, FLAT_QK_DIM,
                Ks, FLAT_QK_DIM);

            meow::tools::apply_causal_decay(attn_block, As, row_base, lane_id);

            attn_block.copy_to_half(scratch + row_base * CHUNK_SIZE, CHUNK_SIZE);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            meow::mma::Tile<TILE_ROWS, V_COLS> o_reg;
            o_reg.clear();
            meow::mma::mm_AB<CHUNK_SIZE, TILE_ROWS, V_COLS>(
                o_reg, scratch + row_base * CHUNK_SIZE, CHUNK_SIZE,
                Vs, HEAD_DIM_V);

            o_reg.store(Vs + row_base * HEAD_DIM_V, HEAD_DIM_V);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        
        static void inter_chunk() {
            
        }

        static void compute(threadgroup float* As, threadgroup float* Bs,
                            threadgroup float* cumsum_scratch,
                            threadgroup float* local_decay,
                            uint lid, uint simd_id, uint lane_id) {
            // cumsum (hillis-steele)
            meow::tools::threadgroup_cumsum<float, CHUNK_SIZE>(
                As, cumsum_scratch, lid, lane_id, simd_id);
            // decay
            for (uint i = lid; i < shared_layout::DECAY_SIZE; i += N_THREADS) {
                local_decay[i] = fast::exp(As[i]) * Bs[i];
            }
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        
        
        
        
        
    };
};

} // namespace meow::mamba::mamba3
