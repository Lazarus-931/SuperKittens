//
//  mamba3.metal
//  SuperKittens
//
//  Created by Alazar Manakelew on 4/13/26.
//

#include "../../../meow.h"

namespace meow::mamba::mamba3 {

template<int CHUNK, int HD_QK, int HD_V>
struct mamba3_fwd {
    
    // metal is very annoying, no struct, enums instead
    enum : int {
        CHUNK_SIZE = CHUNK,
        HEAD_DIM_QK = HD_QK,
        HEAD_DIM_V = HD_V,
        TILE_M = 8,
        N_SIMDS = CHUNK_SIZE / TILE_M,
        N_THREADS = N_SIMDS * 32,
    };
    
    
    static_assert(CHUNK_SIZE % TILE_M == 0, "CHUNK_SIZE must be multiple of TILE_M");
    static_assert(HEAD_DIM_QK % 16 == 0, "HEAD_DIM_QK must be multiple of 16");
    static_assert(N_THREADS % 128 == 0, "N_THREADS must be multiple of 128");

    struct shared_layout {
        enum : int {
            A_SIZE = CHUNK_SIZE,
            B_SIZE = CHUNK_SIZE,
            ANGLE_SIZE = CHUNK_SIZE * (HEAD_DIM_QK / 2),
            CUMSUM_SCRATCH_SIZE = CHUNK_SIZE / 32
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

        for (uint i = lid; i < shared_layout::ANGLE_SIZE; i += N_THREADS)
            angles[i] = angle[cs.angle_offset + i];


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


    template<int RANK_BLOCKS>
    static void apply_rotary_qk_impl(threadgroup half* Q,
                                     threadgroup half* K,
                                     threadgroup half* angles,
                                     uint lid) {
        constexpr int HALF_DIM = HEAD_DIM_QK / 2;
        constexpr int TOTAL_PAIRS = CHUNK_SIZE * HALF_DIM * RANK_BLOCKS;
        constexpr int ROW_STRIDE = HEAD_DIM_QK * RANK_BLOCKS;

        for (uint i = lid; i < TOTAL_PAIRS; i += N_THREADS) {
            uint t = i / (HALF_DIM * RANK_BLOCKS);
            uint rem = i % (HALF_DIM * RANK_BLOCKS);
            uint rank = rem / HALF_DIM;
            uint p = rem % HALF_DIM;

            float2 cs = rotary_trig(float(angles[t * HALF_DIM + p]));
            float c = cs.x;
            float s = cs.y;

            uint d0 = t * ROW_STRIDE + rank * HEAD_DIM_QK + p;
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

    static void apply_rotary_qk(threadgroup half* Q,
                                threadgroup half* K,
                                threadgroup half* angles,
                                uint lid) {
        apply_rotary_qk_impl<1>(Q, K, angles, lid);
    }

    template<int QK_DIM>
    static void intra_chunk_impl(threadgroup half* Qs, threadgroup half* Ks,
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
        meow::mma::mm_ABt<QK_DIM, TILE_ROWS, QK_COLS>(
            attn_block, Qs + row_base * QK_DIM, QK_DIM,
            Ks, QK_DIM);

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

    template<int QK_DIM>
    static void state_update_impl(threadgroup half* Ks,
                                  threadgroup half* Vs, threadgroup float* As,
                                  threadgroup float* Bs,
                                  threadgroup half* kv_state,
                                  uint lid, uint simd_id) {
        enum : int {
            QK_COLS = QK_DIM / 8,
            V_COLS = HEAD_DIM_V / 8,
            STATE_SIZE = QK_DIM * HEAD_DIM_V
        };

        if (simd_id == 0) {
            float chunk_decay = metal::fast::exp(As[CHUNK_SIZE - 1]) * Bs[CHUNK_SIZE - 1];
            for (uint i = lid; i < STATE_SIZE; i += 32) {
                kv_state[i] = half(float(kv_state[i]) * chunk_decay);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_id == 0) {
            meow::mma::Tile<QK_COLS, V_COLS> state_update;
            state_update.clear();
            meow::mma::mm_AtB<CHUNK_SIZE, QK_COLS, V_COLS>(
                state_update, Ks, QK_DIM, Vs, HEAD_DIM_V);
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

    template<int QK_DIM>
    static void state_update_scalar_impl(threadgroup half* Ks,
                                         threadgroup half* Vs,
                                         threadgroup float* As,
                                         threadgroup float* Bs,
                                         threadgroup half* kv_state,
                                         uint lid) {
        const float chunk_decay = metal::fast::exp(As[CHUNK_SIZE - 1]) * Bs[CHUNK_SIZE - 1];
        const uint state_size = QK_DIM * HEAD_DIM_V;
        for (uint idx = lid; idx < state_size; idx += N_THREADS) {
            const uint i = idx / HEAD_DIM_V;
            const uint j = idx % HEAD_DIM_V;
            float acc = float(kv_state[idx]) * chunk_decay;
            for (uint t = 0; t < CHUNK_SIZE; ++t) {
                acc += float(Ks[t * QK_DIM + i]) * float(Vs[t * HEAD_DIM_V + j]);
            }
            kv_state[idx] = half(acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    template<int QK_DIM>
    static void output_scalar_impl(device const half* V,
                                   const thread int& v_offset,
                                   threadgroup half* Qs,
                                   threadgroup half* Ks,
                                   threadgroup half* Vs,
                                   threadgroup float* As,
                                   threadgroup float* Bs,
                                   threadgroup half* kv_state,
                                   uint lid) {
        const uint out_size = CHUNK_SIZE * HEAD_DIM_V;
        for (uint idx = lid; idx < out_size; idx += N_THREADS) {
            const uint row = idx / HEAD_DIM_V;
            const uint j = idx % HEAD_DIM_V;

            float intra = 0.0f;
            for (uint cc = 0; cc <= row; ++cc) {
                float score = 0.0f;
                for (uint i = 0; i < QK_DIM; ++i) {
                    score += float(Qs[row * QK_DIM + i]) * float(Ks[cc * QK_DIM + i]);
                }
                intra += score * metal::fast::exp(As[row] - As[cc]) * float(V[v_offset + cc * HEAD_DIM_V + j]);
            }

            float inter = 0.0f;
            for (uint i = 0; i < QK_DIM; ++i) {
                inter += float(Qs[row * QK_DIM + i]) * float(kv_state[i * HEAD_DIM_V + j]);
            }

            const float q_decay = metal::fast::exp(As[row]) * Bs[row];
            Vs[idx] = half(intra + q_decay * inter);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    template<int QK_DIM>
    static void inter_chunk_scalar_add_impl(threadgroup half* Qs,
                                            threadgroup half* Vs,
                                            threadgroup float* As,
                                            threadgroup float* Bs,
                                            threadgroup half* kv_state,
                                            uint lid) {
        const uint out_size = CHUNK_SIZE * HEAD_DIM_V;
        for (uint idx = lid; idx < out_size; idx += N_THREADS) {
            const uint row = idx / HEAD_DIM_V;
            const uint j = idx % HEAD_DIM_V;
            float inter = 0.0f;
            for (uint i = 0; i < QK_DIM; ++i) {
                inter += float(Qs[row * QK_DIM + i]) * float(kv_state[i * HEAD_DIM_V + j]);
            }

            const float q_decay = metal::fast::exp(As[row]) * Bs[row];
            Vs[idx] = half(float(Vs[idx]) + q_decay * inter);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    template<int QK_DIM>
    static void inter_chunk_impl(threadgroup half* Qs,
                                 threadgroup half* Vs, threadgroup float* As,
                                 threadgroup float* Bs,
                                 threadgroup half* kv_state,
                                 uint simd_id, uint lane_id) {
        enum : int {
            TILE_ROWS = TILE_M / 8,
            V_COLS = HEAD_DIM_V / 8,
            QK_COLS = QK_DIM / 8
        };

        uint row_base = simd_id * TILE_M;
        uint qid = lane_id / 4;
        uint local_row = (qid & 4) + (lane_id / 2) % 4;

        meow::mma::Tile<TILE_ROWS, QK_COLS> q_reg;
        q_reg.load(Qs + row_base * QK_DIM, QK_DIM);

        for (int r = 0; r < TILE_ROWS; r++) {
            uint abs_row = row_base + r * 8 + local_row;
            float decay = metal::fast::exp(As[abs_row]) * Bs[abs_row];
            for (int c = 0; c < QK_COLS; c++) {
                auto d = reinterpret_cast<thread float2&>(
                    q_reg.data[r][c].thread_elements());
                d *= decay;
            }
        }

        q_reg.copy_to_half(Qs + row_base * QK_DIM, QK_DIM);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        meow::mma::Tile<TILE_ROWS, V_COLS> inter_out;
        inter_out.clear();
        meow::mma::mm_AB<QK_DIM, TILE_ROWS, V_COLS>(
            inter_out, Qs + row_base * QK_DIM, QK_DIM,
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
            mamba3_fwd::template intra_chunk_impl<HEAD_DIM_QK>(
                Qs, Ks, Vs, As, scratch, simd_id, lane_id);
        }

        static void state_update(threadgroup half* Ks,
                                 threadgroup half* Vs, threadgroup float* As,
                                 threadgroup float* Bs,
                                 threadgroup half* kv_state,
                                 uint lid, uint simd_id) {
            mamba3_fwd::template state_update_impl<HEAD_DIM_QK>(
                Ks, Vs, As, Bs, kv_state, lid, simd_id);
        }

        static void inter_chunk(threadgroup half* Qs,
                                threadgroup half* Vs, threadgroup float* As,
                                threadgroup float* Bs,
                                threadgroup half* kv_state,
                                uint simd_id, uint lane_id) {
            mamba3_fwd::template inter_chunk_impl<HEAD_DIM_QK>(
                Qs, Vs, As, Bs, kv_state, simd_id, lane_id);
        }

        static void compute(device const half* V,
                            const thread state& st,
                            threadgroup half* Qs, threadgroup half* Ks,
                            threadgroup half* Vs, threadgroup float* As,
                            threadgroup float* Bs,
                            threadgroup half* angles,
                            threadgroup float* angle_state,
                            threadgroup float* cumsum_scratch,
                            threadgroup half* scratch,
                            threadgroup half* kv_state,
                            uint lid, uint simd_id, uint lane_id) {
            meow::tools::threadgroup_cumsum<float, CHUNK_SIZE>(
                As, cumsum_scratch, lid, lane_id, simd_id);

            build_trap_scale(As, Bs, Bs, lid);
            def_rotary_angle(angles, As, angle_state, lid);
            apply_rotary_qk(Qs, Ks, angles, lid);

            mamba3_fwd::template state_update_scalar_impl<HEAD_DIM_QK>(
                Ks, Vs, As, Bs, kv_state, lid);
            mamba3_fwd::template output_scalar_impl<HEAD_DIM_QK>(
                V, st.v_offset, Qs, Ks, Vs, As, Bs, kv_state, lid);
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
            mamba3_fwd::template intra_chunk_impl<HEAD_DIM_QK * MIMO_RANK>(
                Qs, Ks, Vs, As, scratch, simd_id, lane_id);
        }
        
        
        static void state_update(threadgroup half* Ks,
                                 threadgroup half* Vs, threadgroup float* As,
                                 threadgroup float* Bs,
                                 threadgroup half* kv_state,
                                 uint lid, uint simd_id) {
            mamba3_fwd::template state_update_impl<HEAD_DIM_QK * MIMO_RANK>(
                Ks, Vs, As, Bs, kv_state, lid, simd_id);
        }

        static void inter_chunk(threadgroup half* Qs,
                                threadgroup half* Vs, threadgroup float* As,
                                threadgroup float* Bs,
                                threadgroup half* kv_state,
                                uint simd_id, uint lane_id) {
            mamba3_fwd::template inter_chunk_impl<HEAD_DIM_QK * MIMO_RANK>(
                Qs, Vs, As, Bs, kv_state, simd_id, lane_id);
        }

        static void compute(threadgroup half* Qs, threadgroup half* Ks,
                            threadgroup half* Vs, threadgroup float* As,
                            threadgroup float* Bs,
                            threadgroup half* angles,
                            threadgroup float* angle_state,
                            threadgroup float* cumsum_scratch,
                            threadgroup half* scratch,
                            threadgroup half* kv_state,
                            uint lid, uint simd_id, uint lane_id) {
            meow::tools::threadgroup_cumsum<float, CHUNK_SIZE>(
                As, cumsum_scratch, lid, lane_id, simd_id);

            build_trap_scale(As, Bs, Bs, lid);
            def_rotary_angle(angles, As, angle_state, lid);
            mamba3_fwd::template apply_rotary_qk_impl<MIMO_RANK>(
                Qs, Ks, angles, lid);

            state_update(Ks, Vs, As, Bs, kv_state, lid, simd_id);
            intra_chunk(Qs, Ks, Vs, As, scratch, simd_id, lane_id);
            inter_chunk(Qs, Vs, As, Bs, kv_state, simd_id, lane_id);
        }

        static void finish(device half* O, const thread state& st,
                           threadgroup half* Vs, uint lid) {
            for (uint i = lid; i < layout::V_SIZE; i += N_THREADS)
                O[st.o_offset + i] = Vs[i];
        }
        
        
        
        
        
        
    };
};

struct Mamba3FwdArgs {
    uint batch;
    uint nheads;
    uint seq_len;
    uint n_chunks;
};

template<int CHUNK, int HD_QK, int HD_V>
[[kernel, max_total_threads_per_threadgroup(256)]]
void mamba3_siso_fwd_kernel(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device const float* A [[buffer(3)]],
    device const float* B_trap [[buffer(4)]],
    device const half* angle [[buffer(5)]],
    device half* O [[buffer(6)]],
    constant Mamba3FwdArgs& args [[buffer(7)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]]
) {
    if (gid.x >= args.batch || gid.y >= args.nheads) return;

    using op = mamba3_fwd<CHUNK, HD_QK, HD_V>;

    threadgroup float As[op::shared_layout::A_SIZE];
    threadgroup float Bs[op::shared_layout::B_SIZE];
    threadgroup half angles[op::shared_layout::ANGLE_SIZE];
    threadgroup float cumsum_scratch[op::shared_layout::CUMSUM_SCRATCH_SIZE];
    threadgroup float angle_state[HD_QK / 2];
    threadgroup half Qs[op::siso::layout::Q_SIZE];
    threadgroup half Ks[op::siso::layout::K_SIZE];
    threadgroup half Vs[op::siso::layout::V_SIZE];
    threadgroup half scratch[CHUNK * CHUNK];
    threadgroup half kv_state[HD_QK * HD_V];

    for (uint i = lid; i < HD_QK / 2; i += op::N_THREADS) angle_state[i] = 0.0f;
    for (uint i = lid; i < HD_QK * HD_V; i += op::N_THREADS) kv_state[i] = half(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Loop over chunks sequentially so kv_state accumulates across the sequence
    for (uint chunk_idx = 0; chunk_idx < args.n_chunks; chunk_idx++) {
        uint3 chunk_gid = uint3(gid.x, gid.y, chunk_idx);

        typename op::common_state cs;
        typename op::siso::state st;
        op::setup_common(cs, chunk_gid, args.seq_len, args.nheads);
        op::siso::setup(st, cs, args.seq_len, args.nheads);
        op::load_common(A, B_trap, angle, cs, As, Bs, angles, lid);
        op::siso::load(Q, K, V, st, Qs, Ks, Vs, lid);
        op::siso::compute(V, st, Qs, Ks, Vs, As, Bs, angles, angle_state,
                          cumsum_scratch, scratch, kv_state,
                          lid, simd_id, lane_id);
        op::siso::finish(O, st, Vs, lid);
    }
}

template<int CHUNK, int HD_QK, int HD_V, int RANK, int DIM>
[[kernel, max_total_threads_per_threadgroup(256)]]
void mamba3_mimo_fwd_kernel(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device const float* A [[buffer(3)]],
    device const float* B_trap [[buffer(4)]],
    device const half* angle [[buffer(5)]],
    device half* O [[buffer(6)]],
    constant Mamba3FwdArgs& args [[buffer(7)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]]
) {
    if (gid.x >= args.batch || gid.y >= args.nheads) return;

    using op = mamba3_fwd<CHUNK, HD_QK, HD_V>;
    using variant = typename op::template mimo<RANK, DIM>;

    threadgroup float As[op::shared_layout::A_SIZE];
    threadgroup float Bs[op::shared_layout::B_SIZE];
    threadgroup half angles[op::shared_layout::ANGLE_SIZE];
    threadgroup float cumsum_scratch[op::shared_layout::CUMSUM_SCRATCH_SIZE];
    threadgroup float angle_state[HD_QK / 2];
    threadgroup half Qs[variant::layout::Q_SIZE];
    threadgroup half Ks[variant::layout::K_SIZE];
    threadgroup half Vs[variant::layout::V_SIZE];
    threadgroup half scratch[CHUNK * CHUNK];
    threadgroup half kv_state[HD_QK * RANK * HD_V];

    for (uint i = lid; i < HD_QK / 2; i += op::N_THREADS) angle_state[i] = 0.0f;
    for (uint i = lid; i < HD_QK * RANK * HD_V; i += op::N_THREADS) kv_state[i] = half(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint chunk_idx = 0; chunk_idx < args.n_chunks; chunk_idx++) {
        uint3 chunk_gid = uint3(gid.x, gid.y, chunk_idx);

        typename op::common_state cs;
        typename variant::state st;
        op::setup_common(cs, chunk_gid, args.seq_len, args.nheads);
        variant::setup(st, cs, args.seq_len, args.nheads);
        op::load_common(A, B_trap, angle, cs, As, Bs, angles, lid);
        variant::load(Q, K, V, st, Qs, Ks, Vs, lid);
        variant::compute(Qs, Ks, Vs, As, Bs, angles, angle_state,
                         cumsum_scratch, scratch, kv_state,
                         lid, simd_id, lane_id);
        variant::finish(O, st, Vs, lid);
    }
}

template [[host_name("mamba3_siso_fwd_64_64_64")]]
[[kernel]]
void mamba3_siso_fwd_kernel<64, 64, 64>(
    device const half*, device const half*, device const half*,
    device const float*, device const float*, device const half*,
    device half*, constant Mamba3FwdArgs&,
    uint3, uint, uint, uint);

template [[host_name("mamba3_siso_fwd_32_64_64")]]
[[kernel]]
void mamba3_siso_fwd_kernel<32, 64, 64>(
    device const half*, device const half*, device const half*,
    device const float*, device const float*, device const half*,
    device half*, constant Mamba3FwdArgs&,
    uint3, uint, uint, uint);

template [[host_name("mamba3_mimo_fwd_32_64_64_r2")]]
[[kernel]]
void mamba3_mimo_fwd_kernel<32, 64, 64, 2, 64>(
    device const half*, device const half*, device const half*,
    device const float*, device const float*, device const half*,
    device half*, constant Mamba3FwdArgs&,
    uint3, uint, uint, uint);

} // namespace meow::mamba::mamba3
