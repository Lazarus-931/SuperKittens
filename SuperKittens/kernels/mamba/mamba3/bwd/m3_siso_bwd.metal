//
//  m3_siso_bwd.metal
//  SuperKittens
//


#include "../../../../meow.h"

namespace meow::mamba::mamba3 {

struct Mamba3SisoBwdArgs {
    uint batch;
    uint nheads;
    uint nheads_qk;
    uint seq_len;
    uint n_chunks;
};

struct Mamba3SisoBwdMainArgs {
    uint batch;
    uint nheads;
    uint seq_len;
    uint n_chunks;
};

    
// This does dZ = dO * O * sigmoid(Z) * (1 + Z * (1 - sigmoid(Z))) & dO_scaled = dO * sigmoid(Z) * Z  
template<int CHUNK, int HD_QK, int HD_V>
struct mamba3_siso_bwd_dzdo {
    enum : int {
        CHUNK_SIZE = CHUNK,
        HEAD_DIM_QK = HD_QK,
        HEAD_DIM_V = HD_V,
        N_THREADS = 256
    };

    static_assert(HEAD_DIM_V % 16 == 0, "HEAD_DIM_V must be multiple of 16");

    struct state {
        int batch;
        int head;
        int chunk;
    };

    static void setup(thread state& st, uint3 gid) {
        st.batch = gid.z;
        st.head = gid.y;
        st.chunk = gid.x;
    }

    static void compute(
        device const half* dO,
        device const half* Z,
        device const half* O,
        device half* dZ,
        device half* dO_scaled,
        const constant Mamba3SisoBwdArgs& args,
        const thread state& st,
        uint lid)
    {
        const uint chunk_start = static_cast<uint>(st.chunk) * CHUNK_SIZE;
        const uint VEC = 4;
        const uint HEAD_DIM_V_VECS = HEAD_DIM_V / VEC;
        const uint total_vecs = CHUNK_SIZE * HEAD_DIM_V_VECS;
        const uint head_stride = args.nheads * HEAD_DIM_V;
        const uint batch_offset = static_cast<uint>(st.batch) * args.seq_len * head_stride;
        const uint head_offset = static_cast<uint>(st.head) * HEAD_DIM_V;

        for (uint idx = lid; idx < total_vecs; idx += N_THREADS) {
            const uint t = idx / HEAD_DIM_V_VECS;
            const uint v = (idx - t * HEAD_DIM_V_VECS) * VEC;
            const uint seq = chunk_start + t;
            if (seq >= args.seq_len) continue;

            const uint offset = batch_offset + seq * head_stride + head_offset + v;

            const float4 do_val = float4(
                static_cast<float>(dO[offset + 0]),
                static_cast<float>(dO[offset + 1]),
                static_cast<float>(dO[offset + 2]),
                static_cast<float>(dO[offset + 3]));
            const float4 z_val = float4(
                static_cast<float>(Z[offset + 0]),
                static_cast<float>(Z[offset + 1]),
                static_cast<float>(Z[offset + 2]),
                static_cast<float>(Z[offset + 3]));
            const float4 o_val = float4(
                static_cast<float>(O[offset + 0]),
                static_cast<float>(O[offset + 1]),
                static_cast<float>(O[offset + 2]),
                static_cast<float>(O[offset + 3]));
            const float4 sigmoid_z = 1.0f / (1.0f + metal::fast::exp(-z_val));
            const float4 do_sigmoid = do_val * sigmoid_z;
            const float4 dz_val = do_sigmoid * o_val * (1.0f + z_val * (1.0f - sigmoid_z));
            const float4 do_scaled_val = do_sigmoid * z_val;

            dZ[offset + 0] = half(dz_val[0]);
            dZ[offset + 1] = half(dz_val[1]);
            dZ[offset + 2] = half(dz_val[2]);
            dZ[offset + 3] = half(dz_val[3]);
            dO_scaled[offset + 0] = half(do_scaled_val[0]);
            dO_scaled[offset + 1] = half(do_scaled_val[1]);
            dO_scaled[offset + 2] = half(do_scaled_val[2]);
            dO_scaled[offset + 3] = half(do_scaled_val[3]);
        }
    }
};

template<int CHUNK, int HD_QK, int HD_V>
struct mamba3_siso_bwd_dqkv {
    enum : int {
        CHUNK_SIZE = CHUNK,
        HEAD_DIM_QK = HD_QK,
        HEAD_DIM_V = HD_V,
        N_THREADS = 256
    };

    static_assert(HEAD_DIM_QK % 16 == 0, "HEAD_DIM_QK must be multiple of 16");
    static_assert(HEAD_DIM_V % 16 == 0, "HEAD_DIM_V must be multiple of 16");

    struct state {
        int batch;
        int head;
    };

    static void setup(thread state& st, uint3 gid) {
        st.batch = gid.y;
        st.head = gid.x;
    }

    static void compute(
        device const half* Q,
        device const half* K,
        device const half* V,
        device const float* DA_CS,
        device const float* DA_CS_SUM,
        device const half* QK_Dot,
        device const float* D,
        device const half* SSM_States,
        device const half* dO_scaled,
        device const half* d_OSSM_State,
        device const int* Cu_Seqlens,
        device half* dQ,
        device half* dK,
        device half* dV,
        device float* dAdt,
        device half* dQK_Dot,
        device float* dD,
        device half* d_ISSM_State,
        const constant Mamba3SisoBwdArgs& args,
        const thread state& st,
        uint lid,
        uint simd_id,
        uint lane_id)
    {
        (void)Q;
        (void)K;
        (void)V;
        (void)DA_CS;
        (void)DA_CS_SUM;
        (void)QK_Dot;
        (void)D;
        (void)SSM_States;
        (void)dO_scaled;
        (void)d_OSSM_State;
        (void)Cu_Seqlens;
        (void)dQ;
        (void)dK;
        (void)dV;
        (void)dAdt;
        (void)dQK_Dot;
        (void)dD;
        (void)d_ISSM_State;
        (void)args;
        (void)st;
        (void)lid;
        (void)simd_id;
        (void)lane_id;
        
        
    }
};

template<int CHUNK, int HD_QK, int HD_V>
[[kernel, max_total_threads_per_threadgroup(256)]]
void mamba3_siso_bwd_dzdo_kernel(
    device const half* dO [[buffer(0)]],
    device const half* Z [[buffer(1)]],
    device const half* O [[buffer(2)]],
    device half* dZ [[buffer(3)]],
    device half* dO_scaled [[buffer(4)]],
    constant Mamba3SisoBwdArgs& args [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    if (gid.z >= args.batch || gid.y >= args.nheads || gid.x >= args.n_chunks) return;

    using op = mamba3_siso_bwd_dzdo<CHUNK, HD_QK, HD_V>;
    typename op::state st;
    op::setup(st, gid);
    op::compute(dO, Z, O, dZ, dO_scaled, args, st, lid);
}

template<int CHUNK, int HD_QK, int HD_V>
[[kernel, max_total_threads_per_threadgroup(256)]]
void mamba3_siso_bwd_dqkv_kernel(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device const float* DA_CS [[buffer(3)]],
    device const float* DA_CS_SUM [[buffer(4)]],
    device const half* QK_Dot [[buffer(5)]],
    device const float* D [[buffer(6)]],
    device const half* SSM_States [[buffer(7)]],
    device const half* dO_scaled [[buffer(8)]],
    device const half* d_OSSM_State [[buffer(9)]],
    device const int* Cu_Seqlens [[buffer(10)]],
    device half* dQ [[buffer(11)]],
    device half* dK [[buffer(12)]],
    device half* dV [[buffer(13)]],
    device float* dAdt [[buffer(14)]],
    device half* dQK_Dot [[buffer(15)]],
    device float* dD [[buffer(16)]],
    device half* d_ISSM_State [[buffer(17)]],
    constant Mamba3SisoBwdArgs& args [[buffer(18)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]])
{
    if (gid.y >= args.batch || gid.x >= args.nheads) return;

    using op = mamba3_siso_bwd_dqkv<CHUNK, HD_QK, HD_V>;
    typename op::state st;
    op::setup(st, gid);
    op::compute(
        Q, K, V, DA_CS, DA_CS_SUM, QK_Dot, D, SSM_States,
        dO_scaled, d_OSSM_State, Cu_Seqlens,
        dQ, dK, dV, dAdt, dQK_Dot, dD, d_ISSM_State,
        args, st, lid, simd_id, lane_id);
}

template [[host_name("mamba3_siso_bwd_dzdo_32_64_64")]]
[[kernel]]
void mamba3_siso_bwd_dzdo_kernel<32, 64, 64>(
    device const half*, device const half*, device const half*,
    device half*, device half*, constant Mamba3SisoBwdArgs&,
    uint3, uint);

template [[host_name("mamba3_siso_bwd_dqkv_32_64_64")]]
[[kernel]]
void mamba3_siso_bwd_dqkv_kernel<32, 64, 64>(
    device const half*, device const half*, device const half*,
    device const float*, device const float*, device const half*,
    device const float*, device const half*, device const half*,
    device const half*, device const int*, device half*, device half*,
    device half*, device float*, device half*, device float*, device half*,
    constant Mamba3SisoBwdArgs&, uint3, uint, uint, uint);

template [[host_name("mamba3_siso_bwd_dzdo_64_64_64")]]
[[kernel]]
void mamba3_siso_bwd_dzdo_kernel<64, 64, 64>(
    device const half*, device const half*, device const half*,
    device half*, device half*, constant Mamba3SisoBwdArgs&,
    uint3, uint);

template [[host_name("mamba3_siso_bwd_dqkv_64_64_64")]]
[[kernel]]
void mamba3_siso_bwd_dqkv_kernel<64, 64, 64>(
    device const half*, device const half*, device const half*,
    device const float*, device const float*, device const half*,
    device const float*, device const half*, device const half*,
    device const half*, device const int*, device half*, device half*,
    device half*, device float*, device half*, device float*, device half*,
    constant Mamba3SisoBwdArgs&, uint3, uint, uint, uint);

template<int CHUNK, int HD_QK, int HD_V>
struct mamba3_siso_bwd_main {
    enum : int {
        CHUNK_SIZE = CHUNK,
        HEAD_DIM_QK = HD_QK,
        HEAD_DIM_V = HD_V,
        N_THREADS = HEAD_DIM_V,
    };

    static inline uint qk_offset(uint batch, uint head, uint seq, uint dim,
                                 const constant Mamba3SisoBwdMainArgs& args) {
        return (((batch * args.nheads + head) * args.seq_len + seq) * HEAD_DIM_QK + dim);
    }

    static inline uint v_offset(uint batch, uint head, uint seq, uint dim,
                                const constant Mamba3SisoBwdMainArgs& args) {
        return (((batch * args.nheads + head) * args.seq_len + seq) * HEAD_DIM_V + dim);
    }

    static inline uint scalar_offset(uint batch, uint head, uint seq,
                                     const constant Mamba3SisoBwdMainArgs& args) {
        return ((batch * args.nheads + head) * args.seq_len + seq);
    }

    static inline uint angle_offset(uint batch, uint head, uint seq, uint dim,
                                    const constant Mamba3SisoBwdMainArgs& args) {
        return ((((batch * args.nheads + head) * args.seq_len) + seq) * (HEAD_DIM_QK / 2) + dim);
    }

    static inline uint state_offset(uint batch, uint head, uint chunk, uint qk, uint v,
                                    const constant Mamba3SisoBwdMainArgs& args) {
        return (((((batch * args.nheads + head) * args.n_chunks + chunk) * HEAD_DIM_QK) + qk) * HEAD_DIM_V + v);
    }

    static inline float trap_scale(float a_cs, float b_raw) {
        return 1.0f + b_raw * metal::fast::exp(-a_cs);
    }

    static inline float reverse_cumsum_sum(const thread float* vals, uint start, uint len) {
        float acc = 0.0f;
        for (uint i = len; i > start; --i) acc += vals[i - 1];
        return acc;
    }

    static inline float reverse_cumsum_sum(const threadgroup float* vals, uint start, uint len) {
        float acc = 0.0f;
        for (uint i = len; i > start; --i) acc += vals[i - 1];
        return acc;
    }

    static inline void rotate_forward(
        float q0, float q1, float k0, float k1, float theta,
        thread float& q0_rot, thread float& q1_rot,
        thread float& k0_rot, thread float& k1_rot)
    {
        const float c = metal::fast::cos(theta);
        const float s = metal::fast::sin(theta);
        q0_rot = q0 * c - q1 * s;
        q1_rot = q0 * s + q1 * c;
        k0_rot = k0 * c - k1 * s;
        k1_rot = k0 * s + k1 * c;
    }

    static inline void rotate_backward(
        float q0, float q1, float k0, float k1, float theta,
        float dq0_rot, float dq1_rot, float dk0_rot, float dk1_rot,
        thread float& dq0, thread float& dq1,
        thread float& dk0, thread float& dk1,
        thread float& dtheta)
    {
        const float c = metal::fast::cos(theta);
        const float s = metal::fast::sin(theta);

        dq0 = dq0_rot * c + dq1_rot * s;
        dq1 = -dq0_rot * s + dq1_rot * c;
        dk0 = dk0_rot * c + dk1_rot * s;
        dk1 = -dk0_rot * s + dk1_rot * c;

        dtheta =
            dq0_rot * (-q0 * s - q1 * c) +
            dq1_rot * ( q0 * c - q1 * s) +
            dk0_rot * (-k0 * s - k1 * c) +
            dk1_rot * ( k0 * c - k1 * s);
    }

    static void compute(
        device const half* Q,
        device const half* K,
        device const half* V,
        device const float* A,
        device const float* B,
        device const half* angle,
        device const half* dO,
        device const float* saved_states,
        device float* dQ,
        device float* dK,
        device float* dV,
        device float* dA,
        device float* dB,
        device float* dAngle,
        const constant Mamba3SisoBwdMainArgs& args,
        uint3 gid,
        uint lid,
        uint simd_id,
        uint lane_id,
        threadgroup float* d_state_acc,
        threadgroup float* q_rot_row,
        threadgroup float* k_rot_row,
        threadgroup float* a_cs_shared,
        threadgroup float* b_scale_shared,
        threadgroup float* scalar_scratch,
        threadgroup float* partial_sums,
        threadgroup float* d_a_cs_shared,
        threadgroup float* d_b_scale_shared,
        threadgroup float* d_q_decay_shared,
        threadgroup float* dk_chunk_shared,
        threadgroup float* dq_row_shared)
    {
        const float pi = 3.14159265358979323846f;
        const uint batch = gid.y;
        const uint head = gid.x;
        const uint half_dim = HEAD_DIM_QK / 2;
        for (uint idx = lid; idx < HEAD_DIM_QK * HEAD_DIM_V; idx += N_THREADS) {
            d_state_acc[idx] = 0.0f;
        }
        for (uint seq = 0; seq < args.seq_len; ++seq) {
            if (lid < HEAD_DIM_QK) {
                dQ[qk_offset(batch, head, seq, lid, args)] = 0.0f;
                dK[qk_offset(batch, head, seq, lid, args)] = 0.0f;
            }
            if (lid < HEAD_DIM_V) {
                dV[v_offset(batch, head, seq, lid, args)] = 0.0f;
            }
            if (lid < HEAD_DIM_QK / 2) {
                dAngle[angle_offset(batch, head, seq, lid, args)] = 0.0f;
            }
            if (lid == 0) {
                dA[scalar_offset(batch, head, seq, args)] = 0.0f;
                dB[scalar_offset(batch, head, seq, args)] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int chunk = int(args.n_chunks) - 1; chunk >= 0; --chunk) {
            const uint chunk_start = uint(chunk) * CHUNK_SIZE;
            const uint chunk_len = min(CHUNK_SIZE, int(args.seq_len - chunk_start));

            float dv_acc[CHUNK_SIZE];

            if (lid == 0) {
                float running = 0.0f;
                for (uint t = 0; t < chunk_len; ++t) {
                    const uint seq = chunk_start + t;
                    running += A[scalar_offset(batch, head, seq, args)];
                    a_cs_shared[t] = running;
                    b_scale_shared[t] = trap_scale(
                        a_cs_shared[t],
                        B[scalar_offset(batch, head, seq, args)]);
                    d_a_cs_shared[t] = 0.0f;
                    d_b_scale_shared[t] = 0.0f;
                    d_q_decay_shared[t] = 0.0f;
                }
                for (uint i = 0; i < chunk_len * HEAD_DIM_QK; ++i) {
                    dk_chunk_shared[i] = 0.0f;
                }
            }
            for (uint t = 0; t < chunk_len; ++t) {
                dv_acc[t] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint state_chunk_base = uint(chunk);
            device const float* s_cur = saved_states + state_offset(batch, head, state_chunk_base, 0, 0, args);
            device const float* s_prev = (chunk > 0)
                ? (saved_states + state_offset(batch, head, state_chunk_base - 1, 0, 0, args))
                : nullptr;

            for (uint r = 0; r < chunk_len; ++r) {
                const uint seq_r = chunk_start + r;
                const uint half_dim = HEAD_DIM_QK / 2;
                const float q_decay = metal::fast::exp(a_cs_shared[r]) * b_scale_shared[r];

                if (lid == 0) {
                    for (uint i = 0; i < half_dim; ++i) {
                        const float q0 = float(Q[qk_offset(batch, head, seq_r, i, args)]);
                        const float q1 = float(Q[qk_offset(batch, head, seq_r, i + half_dim, args)]);
                        const float theta = a_cs_shared[r] * float(angle[angle_offset(batch, head, seq_r, i, args)]) * pi;
                        float q0_rot, q1_rot, k0_dummy, k1_dummy;
                        rotate_forward(q0, q1, 0.0f, 0.0f, theta, q0_rot, q1_rot, k0_dummy, k1_dummy);
                        q_rot_row[i] = q0_rot;
                        q_rot_row[i + half_dim] = q1_rot;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (lid == 0) {
                    for (uint i = 0; i < HEAD_DIM_QK; ++i) dq_row_shared[i] = 0.0f;
                }
                float inter = 0.0f;
                for (uint i = 0; i < HEAD_DIM_QK; ++i) {
                    inter += q_rot_row[i] * s_cur[i * HEAD_DIM_V + lid];
                }
                partial_sums[lid] = float(dO[v_offset(batch, head, seq_r, lid, args)]) * inter;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lid == 0) {
                    float d_q_decay = 0.0f;
                    for (uint j = 0; j < HEAD_DIM_V; ++j) d_q_decay += partial_sums[j];
                    d_q_decay_shared[r] += d_q_decay;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const float do_val_rj_init = float(dO[v_offset(batch, head, seq_r, lid, args)]);
                for (uint i = 0; i < HEAD_DIM_QK; ++i) {
                    d_state_acc[i * HEAD_DIM_V + lid] += q_decay * q_rot_row[i] * do_val_rj_init;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float s_grad = 0.0f;
                for (uint j = 0; j < HEAD_DIM_V; ++j) {
                    s_grad += float(dO[v_offset(batch, head, seq_r, j, args)]) * s_cur[lid * HEAD_DIM_V + j];
                }
                dq_row_shared[lid] += q_decay * s_grad;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint c = 0; c <= r; ++c) {
                    const uint seq_c = chunk_start + c;
                    if (lid == 0) {
                        float score = 0.0f;
                        for (uint i = 0; i < half_dim; ++i) {
                            const float k0 = float(K[qk_offset(batch, head, seq_c, i, args)]);
                            const float k1 = float(K[qk_offset(batch, head, seq_c, i + half_dim, args)]);
                            const float theta = a_cs_shared[c] * float(angle[angle_offset(batch, head, seq_c, i, args)]) * pi;
                            float q0_dummy, q1_dummy, k0_rot, k1_rot;
                            rotate_forward(0.0f, 0.0f, k0, k1, theta, q0_dummy, q1_dummy, k0_rot, k1_rot);
                            k_rot_row[i] = k0_rot;
                            k_rot_row[i + half_dim] = k1_rot;
                            score += q_rot_row[i] * k0_rot + q_rot_row[i + half_dim] * k1_rot;
                        }
                        scalar_scratch[0] = score;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    const float do_val_rj = float(dO[v_offset(batch, head, seq_r, lid, args)]);
                    const float decay = metal::fast::exp(a_cs_shared[r] - a_cs_shared[c]);
                    dv_acc[c] += do_val_rj * (decay * scalar_scratch[0]);
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    partial_sums[lid] =
                        float(dO[v_offset(batch, head, seq_r, lid, args)]) *
                        float(V[v_offset(batch, head, seq_c, lid, args)]);
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    if (lid == 0) {
                        float alpha = 0.0f;
                        for (uint j = 0; j < HEAD_DIM_V; ++j) alpha += partial_sums[j];
                        alpha *= decay;
                        scalar_scratch[1] = alpha;
                        const float pair_term = alpha * scalar_scratch[0];
                        d_a_cs_shared[r] += pair_term;
                        d_a_cs_shared[c] -= pair_term;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    dq_row_shared[lid] += scalar_scratch[1] * k_rot_row[lid];
                    dk_chunk_shared[c * HEAD_DIM_QK + lid] += scalar_scratch[1] * q_rot_row[lid];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                if (lid == 0) {
                    for (uint i = 0; i < half_dim; ++i) {
                        const float q0 = float(Q[qk_offset(batch, head, seq_r, i, args)]);
                        const float q1 = float(Q[qk_offset(batch, head, seq_r, i + half_dim, args)]);
                        const float k0 = float(K[qk_offset(batch, head, seq_r, i, args)]);
                        const float k1 = float(K[qk_offset(batch, head, seq_r, i + half_dim, args)]);
                        const float raw_angle = float(angle[angle_offset(batch, head, seq_r, i, args)]);
                        const float theta = a_cs_shared[r] * raw_angle * pi;
                        float dq0, dq1, dk0_dummy, dk1_dummy, dtheta;
                        rotate_backward(
                            q0, q1, k0, k1, theta,
                            dq_row_shared[i],
                            dq_row_shared[i + half_dim],
                            0.0f,
                            0.0f,
                            dq0, dq1, dk0_dummy, dk1_dummy, dtheta);
                        dQ[qk_offset(batch, head, seq_r, i, args)] += dq0;
                        dQ[qk_offset(batch, head, seq_r, i + half_dim, args)] += dq1;
                        dAngle[angle_offset(batch, head, seq_r, i, args)] += dtheta * a_cs_shared[r] * pi;
                        d_a_cs_shared[r] += dtheta * raw_angle * pi;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            float d_decay_partial = 0.0f;
            for (uint i = 0; i < HEAD_DIM_QK; ++i) {
                const float prev_val = (s_prev != nullptr) ? s_prev[i * HEAD_DIM_V + lid] : 0.0f;
                d_decay_partial += d_state_acc[i * HEAD_DIM_V + lid] * prev_val;
            }
            partial_sums[lid] = d_decay_partial;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lid == 0) {
                float d_decay = 0.0f;
                for (uint j = 0; j < HEAD_DIM_V; ++j) d_decay += partial_sums[j];
                scalar_scratch[0] = d_decay;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float d_decay = scalar_scratch[0];

            if (lid == 0) {
                scalar_scratch[1] = metal::fast::exp(a_cs_shared[chunk_len - 1]) * b_scale_shared[chunk_len - 1];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float chunk_decay = scalar_scratch[1];

            for (uint t = 0; t < chunk_len; ++t) {
                const uint seq = chunk_start + t;
                const uint half_dim = HEAD_DIM_QK / 2;
                if (lid == 0) {
                    for (uint i = 0; i < half_dim; ++i) {
                        const float k0 = float(K[qk_offset(batch, head, seq, i, args)]);
                        const float k1 = float(K[qk_offset(batch, head, seq, i + half_dim, args)]);
                        const float theta = a_cs_shared[t] * float(angle[angle_offset(batch, head, seq, i, args)]) * pi;
                        float q0_dummy, q1_dummy, k0_rot, k1_rot;
                        rotate_forward(0.0f, 0.0f, k0, k1, theta, q0_dummy, q1_dummy, k0_rot, k1_rot);
                        k_rot_row[i] = k0_rot;
                        k_rot_row[i + half_dim] = k1_rot;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint i = 0; i < HEAD_DIM_QK; ++i) {
                    dv_acc[t] += k_rot_row[i] * d_state_acc[i * HEAD_DIM_V + lid];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float acc = 0.0f;
                for (uint j = 0; j < HEAD_DIM_V; ++j) {
                    acc += d_state_acc[lid * HEAD_DIM_V + j] * float(V[v_offset(batch, head, seq, j, args)]);
                }
                dk_chunk_shared[t * HEAD_DIM_QK + lid] += acc;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (lid == 0) {
                d_a_cs_shared[chunk_len - 1] += d_decay * chunk_decay;
                d_b_scale_shared[chunk_len - 1] += d_decay * metal::fast::exp(a_cs_shared[chunk_len - 1]);

                for (uint t = 0; t < chunk_len; ++t) {
                    const float q_decay = metal::fast::exp(a_cs_shared[t]) * b_scale_shared[t];
                    d_a_cs_shared[t] += d_q_decay_shared[t] * q_decay;
                    d_b_scale_shared[t] += d_q_decay_shared[t] * metal::fast::exp(a_cs_shared[t]);
                }

                for (uint t = 0; t < chunk_len; ++t) {
                    const uint seq = chunk_start + t;
                    const float b_raw = B[scalar_offset(batch, head, seq, args)];
                    const float exp_neg = metal::fast::exp(-a_cs_shared[t]);
                    dB[scalar_offset(batch, head, seq, args)] += d_b_scale_shared[t] * exp_neg;
                    d_a_cs_shared[t] += d_b_scale_shared[t] * (-b_raw * exp_neg);
                }

                for (uint t = 0; t < chunk_len; ++t) {
                    const uint seq = chunk_start + t;
                    for (uint i = 0; i < half_dim; ++i) {
                        const float q0 = float(Q[qk_offset(batch, head, seq, i, args)]);
                        const float q1 = float(Q[qk_offset(batch, head, seq, i + half_dim, args)]);
                        const float k0 = float(K[qk_offset(batch, head, seq, i, args)]);
                        const float k1 = float(K[qk_offset(batch, head, seq, i + half_dim, args)]);
                        const float raw_angle = float(angle[angle_offset(batch, head, seq, i, args)]);
                        const float theta = a_cs_shared[t] * raw_angle * pi;
                        float dq0_dummy, dq1_dummy, dk0, dk1, dtheta;
                        rotate_backward(
                            q0, q1, k0, k1, theta,
                            0.0f,
                            0.0f,
                            dk_chunk_shared[t * HEAD_DIM_QK + i],
                            dk_chunk_shared[t * HEAD_DIM_QK + i + half_dim],
                            dq0_dummy, dq1_dummy, dk0, dk1, dtheta);
                        dK[qk_offset(batch, head, seq, i, args)] += dk0;
                        dK[qk_offset(batch, head, seq, i + half_dim, args)] += dk1;
                        dAngle[angle_offset(batch, head, seq, i, args)] += dtheta * a_cs_shared[t] * pi;
                        d_a_cs_shared[t] += dtheta * raw_angle * pi;
                    }
                }
                float corrected_a[CHUNK_SIZE];
                float exact_db[CHUNK_SIZE];
                for (uint t = 0; t < chunk_len; ++t) {
                    const float a_val = a_cs_shared[t];
                    const float b_raw = B[scalar_offset(batch, head, chunk_start + t, args)];
                    const float exp_pos_fast = metal::fast::exp(a_val);
                    const float exp_neg_fast = metal::fast::exp(-a_val);
                    const float exp_pos = exp(a_val);
                    const float exp_neg = exp(-a_val);

                    float fast_db = d_q_decay_shared[t] * exp_pos_fast;
                    float exact_db_t = d_q_decay_shared[t] * exp_pos;
                    float fast_da = d_q_decay_shared[t] * (exp_pos_fast * b_scale_shared[t]);
                    float exact_da = d_q_decay_shared[t] * (exp_pos * b_scale_shared[t]);
                    if (t + 1 == chunk_len) {
                        fast_db += d_decay * exp_pos_fast;
                        exact_db_t += d_decay * exp_pos;
                        fast_da += d_decay * (exp_pos_fast * b_scale_shared[t]);
                        exact_da += d_decay * (exp_pos * b_scale_shared[t]);
                    }
                    fast_da += fast_db * (-b_raw * exp_neg_fast);
                    exact_da += exact_db_t * (-b_raw * exp_neg);
                    corrected_a[t] = d_a_cs_shared[t] - fast_da + exact_da;
                    exact_db[t] = exact_db_t * exp_neg;
                }

                float suffix = 0.0f;
                for (int t = int(chunk_len) - 1; t >= 0; --t) {
                    suffix += corrected_a[t];
                    dA[scalar_offset(batch, head, chunk_start + uint(t), args)] = suffix;
                    dB[scalar_offset(batch, head, chunk_start + uint(t), args)] = exact_db[t];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            dV[v_offset(batch, head, chunk_start, lid, args)] += dv_acc[0];
            for (uint t = 1; t < chunk_len; ++t) {
                dV[v_offset(batch, head, chunk_start + t, lid, args)] += dv_acc[t];
            }
            for (uint idx = lid; idx < HEAD_DIM_QK * HEAD_DIM_V; idx += N_THREADS) {
                d_state_acc[idx] *= chunk_decay;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
};

template<int CHUNK, int HD_QK, int HD_V>
[[kernel, max_total_threads_per_threadgroup(64)]]
void mamba3_siso_bwd_main_kernel(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device const float* A [[buffer(3)]],
    device const float* B [[buffer(4)]],
    device const half* angle [[buffer(5)]],
    device const half* dO [[buffer(6)]],
    device const float* saved_states [[buffer(7)]],
    device float* dQ [[buffer(8)]],
    device float* dK [[buffer(9)]],
    device float* dV [[buffer(10)]],
    device float* dA [[buffer(11)]],
    device float* dB [[buffer(12)]],
    device float* dAngle [[buffer(13)]],
    constant Mamba3SisoBwdMainArgs& args [[buffer(14)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]])
{
    if (gid.y >= args.batch || gid.x >= args.nheads) return;

    using op = mamba3_siso_bwd_main<CHUNK, HD_QK, HD_V>;
    threadgroup float d_state_acc[op::HEAD_DIM_QK * op::HEAD_DIM_V];
    threadgroup float q_rot_row[op::HEAD_DIM_QK];
    threadgroup float k_rot_row[op::HEAD_DIM_QK];
    threadgroup float a_cs_shared[op::CHUNK_SIZE];
    threadgroup float b_scale_shared[op::CHUNK_SIZE];
    threadgroup float scalar_scratch[2];
    threadgroup float partial_sums[op::N_THREADS];
    threadgroup float d_a_cs_shared[op::CHUNK_SIZE];
    threadgroup float d_b_scale_shared[op::CHUNK_SIZE];
    threadgroup float d_q_decay_shared[op::CHUNK_SIZE];
    threadgroup float dk_chunk_shared[op::CHUNK_SIZE * op::HEAD_DIM_QK];
    threadgroup float dq_row_shared[op::HEAD_DIM_QK];
    op::compute(Q, K, V, A, B, angle, dO, saved_states,
                dQ, dK, dV, dA, dB, dAngle, args, gid,
                lid, simd_id, lane_id,
                d_state_acc, q_rot_row, k_rot_row,
                a_cs_shared, b_scale_shared, scalar_scratch, partial_sums,
                d_a_cs_shared, d_b_scale_shared, d_q_decay_shared,
                dk_chunk_shared, dq_row_shared);
}

template [[host_name("mamba3_siso_bwd_main_32_64_64")]]
[[kernel]]
void mamba3_siso_bwd_main_kernel<32, 64, 64>(
    device const half*, device const half*, device const half*,
    device const float*, device const float*, device const half*,
    device const half*, device const float*,
    device float*, device float*, device float*,
    device float*, device float*, device float*,
    constant Mamba3SisoBwdMainArgs&, uint3, uint, uint, uint);

template [[host_name("mamba3_siso_bwd_main_64_64_64")]]
[[kernel]]
void mamba3_siso_bwd_main_kernel<64, 64, 64>(
    device const half*, device const half*, device const half*,
    device const float*, device const float*, device const half*,
    device const half*, device const float*,
    device float*, device float*, device float*,
    device float*, device float*, device float*,
    constant Mamba3SisoBwdMainArgs&, uint3, uint, uint, uint);

} // namespace meow::mamba::mamba3
