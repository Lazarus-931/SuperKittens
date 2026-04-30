#include <metal_stdlib>
using namespace metal;

namespace meow { namespace mamba { namespace mamba2 {

enum : uint {
    CHUNK_SIZE = 32,
    HEAD_DIM_QK = 64,
    HEAD_DIM_V = 64,
    STATE_SIZE = HEAD_DIM_QK * HEAD_DIM_V,
    N_THREADS = 128
};

struct Mamba2BwdArgs {
    uint batch;
    uint nheads;
    uint seq_len;
    uint n_chunks;
};

METAL_FUNC size_t qk_index(uint b, uint h, uint t, uint d, constant Mamba2BwdArgs& args) {
    return ((((size_t)b * args.nheads + h) * args.seq_len) + t) * HEAD_DIM_QK + d;
}

METAL_FUNC size_t v_index(uint b, uint h, uint t, uint d, constant Mamba2BwdArgs& args) {
    return ((((size_t)b * args.nheads + h) * args.seq_len) + t) * HEAD_DIM_V + d;
}

METAL_FUNC size_t scalar_index(uint b, uint h, uint t, constant Mamba2BwdArgs& args) {
    return (((size_t)b * args.nheads + h) * args.seq_len) + t;
}

METAL_FUNC size_t state_index(uint b, uint h, uint t, uint i, uint j, constant Mamba2BwdArgs& args) {
    return ((((((size_t)b * args.nheads + h) * args.seq_len) + t) * HEAD_DIM_QK) + i) * HEAD_DIM_V + j;
}

METAL_FUNC size_t token_qk_offset(uint b, uint h, uint t, constant Mamba2BwdArgs& args) {
    return ((((size_t)b * args.nheads + h) * args.seq_len) + t) * HEAD_DIM_QK;
}

METAL_FUNC size_t token_v_offset(uint b, uint h, uint t, constant Mamba2BwdArgs& args) {
    return ((((size_t)b * args.nheads + h) * args.seq_len) + t) * HEAD_DIM_V;
}

METAL_FUNC size_t chunk_scalar_offset(uint b, uint h, uint c, constant Mamba2BwdArgs& args) {
    return (((size_t)b * args.nheads + h) * args.n_chunks) + c;
}

METAL_FUNC size_t chunk_state_offset(uint b, uint h, uint c, constant Mamba2BwdArgs& args) {
    return ((((size_t)b * args.nheads + h) * args.n_chunks) + c) * STATE_SIZE;
}

[[host_name("mamba2_siso_bwd_precompute_32_64_64")]]
[[kernel, max_total_threads_per_threadgroup(N_THREADS)]]
void mamba2_siso_bwd_precompute_32_64_64(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device const float* A [[buffer(3)]],
    device const float* states [[buffer(4)]],
    device const float* dO [[buffer(5)]],
    device float* dQ [[buffer(6)]],
    device float* local_dK [[buffer(7)]],
    device float* local_dV [[buffer(8)]],
    device float* local_dA [[buffer(9)]],
    device float* suffix [[buffer(10)]],
    device float* chunk_decay [[buffer(11)]],
    device float* chunk_u [[buffer(12)]],
    constant Mamba2BwdArgs& args [[buffer(13)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {
    if (gid.x >= args.batch || gid.y >= args.nheads || gid.z >= args.n_chunks) return;

    const uint b = gid.x;
    const uint h = gid.y;
    const uint chunk_idx = gid.z;
    const uint chunk_start = chunk_idx * CHUNK_SIZE;
    const uint chunk_len = min(uint(CHUNK_SIZE), args.seq_len - chunk_start);
    const size_t u_base = chunk_state_offset(b, h, chunk_idx, args);

    threadgroup float h_zero[STATE_SIZE];
    threadgroup float reduce[N_THREADS];

    for (uint idx = lid; idx < STATE_SIZE; idx += N_THREADS) h_zero[idx] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float carry_scale = 1.0f;
    for (int local_t = int(chunk_len) - 1; local_t >= 0; --local_t) {
        const uint t = chunk_start + uint(local_t);
        const size_t qk_base = token_qk_offset(b, h, t, args);
        const size_t v_base = token_v_offset(b, h, t, args);
        const size_t s_base = state_index(b, h, t, 0, 0, args);

        if (lid < HEAD_DIM_QK) {
            const uint i = lid;
            float dq_acc = 0.0f;
            const float q_val = float(Q[qk_base + i]);
            for (uint j = 0; j < HEAD_DIM_V; ++j) {
                const float dout = dO[v_base + j];
                const size_t idx = (size_t)i * HEAD_DIM_V + j;
                dq_acc += dout * states[s_base + idx];
                h_zero[idx] += q_val * dout;
            }
            dQ[qk_base + i] = dq_acc;
        }
        if (lid == 0) suffix[scalar_index(b, h, t, args)] = carry_scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid < HEAD_DIM_QK) {
            const uint i = lid;
            float dk_acc = 0.0f;
            for (uint j = 0; j < HEAD_DIM_V; ++j)
                dk_acc += h_zero[(size_t)i * HEAD_DIM_V + j] * float(V[v_base + j]);
            local_dK[qk_base + i] = dk_acc;
        } else if (lid < HEAD_DIM_QK + HEAD_DIM_V) {
            const uint j = lid - HEAD_DIM_QK;
            float dv_acc = 0.0f;
            for (uint i = 0; i < HEAD_DIM_QK; ++i)
                dv_acc += float(K[qk_base + i]) * h_zero[(size_t)i * HEAD_DIM_V + j];
            local_dV[v_base + j] = dv_acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float decay = exp(A[scalar_index(b, h, t, args)]);
        float partial = 0.0f;
        if (t > 0) {
            const size_t s_prev = state_index(b, h, t - 1, 0, 0, args);
            for (uint idx = lid; idx < STATE_SIZE; idx += N_THREADS)
                partial += h_zero[idx] * states[s_prev + idx];
        }
        reduce[lid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = N_THREADS / 2; offset > 0; offset >>= 1) {
            if (lid < offset) reduce[lid] += reduce[lid + offset];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lid == 0) local_dA[scalar_index(b, h, t, args)] = (t > 0) ? (decay * reduce[0]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint idx = lid; idx < STATE_SIZE; idx += N_THREADS) h_zero[idx] *= decay;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        carry_scale *= decay;
    }

    if (lid == 0) chunk_decay[chunk_scalar_offset(b, h, chunk_idx, args)] = carry_scale;
    for (uint idx = lid; idx < STATE_SIZE; idx += N_THREADS) chunk_u[u_base + idx] = h_zero[idx];
}

[[host_name("mamba2_siso_bwd_scan_32_64_64")]]
[[kernel, max_total_threads_per_threadgroup(N_THREADS)]]
void mamba2_siso_bwd_scan_32_64_64(
    device const float* chunk_decay [[buffer(0)]],
    device const float* chunk_u [[buffer(1)]],
    device float* chunk_carry [[buffer(2)]],
    constant Mamba2BwdArgs& args [[buffer(3)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {
    if (gid.x >= args.batch || gid.y >= args.nheads) return;
    const uint b = gid.x;
    const uint h = gid.y;

    threadgroup float carry[STATE_SIZE];
    for (uint idx = lid; idx < STATE_SIZE; idx += N_THREADS) carry[idx] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int c = int(args.n_chunks) - 1; c >= 0; --c) {
        const size_t base = chunk_state_offset(b, h, uint(c), args);
        for (uint idx = lid; idx < STATE_SIZE; idx += N_THREADS)
            chunk_carry[base + idx] = carry[idx];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float decay = chunk_decay[chunk_scalar_offset(b, h, uint(c), args)];
        for (uint idx = lid; idx < STATE_SIZE; idx += N_THREADS)
            carry[idx] = decay * carry[idx] + chunk_u[base + idx];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

[[host_name("mamba2_siso_bwd_finalize_32_64_64")]]
[[kernel, max_total_threads_per_threadgroup(N_THREADS)]]
void mamba2_siso_bwd_finalize_32_64_64(
    device const half* K [[buffer(0)]],
    device const half* V [[buffer(1)]],
    device const float* A [[buffer(2)]],
    device const float* states [[buffer(3)]],
    device const float* local_dK [[buffer(4)]],
    device const float* local_dV [[buffer(5)]],
    device const float* local_dA [[buffer(6)]],
    device const float* suffix [[buffer(7)]],
    device const float* chunk_carry [[buffer(8)]],
    device float* dK [[buffer(9)]],
    device float* dV [[buffer(10)]],
    device float* dA [[buffer(11)]],
    constant Mamba2BwdArgs& args [[buffer(12)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {
    if (gid.x >= args.batch || gid.y >= args.nheads || gid.z >= args.n_chunks) return;

    const uint b = gid.x;
    const uint h = gid.y;
    const uint chunk_idx = gid.z;
    const uint chunk_start = chunk_idx * CHUNK_SIZE;
    const uint chunk_len = min(uint(CHUNK_SIZE), args.seq_len - chunk_start);
    const size_t carry_base = chunk_state_offset(b, h, chunk_idx, args);

    threadgroup float carry[STATE_SIZE];
    threadgroup float reduce[N_THREADS];
    for (uint idx = lid; idx < STATE_SIZE; idx += N_THREADS) carry[idx] = chunk_carry[carry_base + idx];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint local_t = 0; local_t < chunk_len; ++local_t) {
        const uint t = chunk_start + local_t;
        const float scale = suffix[scalar_index(b, h, t, args)];
        const size_t qk_base = token_qk_offset(b, h, t, args);
        const size_t v_base = token_v_offset(b, h, t, args);

        if (lid < HEAD_DIM_QK) {
            const uint i = lid;
            float dk_acc = local_dK[qk_base + i];
            for (uint j = 0; j < HEAD_DIM_V; ++j)
                dk_acc += scale * carry[(size_t)i * HEAD_DIM_V + j] * float(V[v_base + j]);
            dK[qk_base + i] = dk_acc;
        } else if (lid < HEAD_DIM_QK + HEAD_DIM_V) {
            const uint j = lid - HEAD_DIM_QK;
            float dv_acc = local_dV[v_base + j];
            for (uint i = 0; i < HEAD_DIM_QK; ++i)
                dv_acc += scale * float(K[qk_base + i]) * carry[(size_t)i * HEAD_DIM_V + j];
            dV[v_base + j] = dv_acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float partial = 0.0f;
        if (t > 0) {
            const size_t s_prev = state_index(b, h, t - 1, 0, 0, args);
            for (uint idx = lid; idx < STATE_SIZE; idx += N_THREADS)
                partial += carry[idx] * states[s_prev + idx];
        }
        reduce[lid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = N_THREADS / 2; offset > 0; offset >>= 1) {
            if (lid < offset) reduce[lid] += reduce[lid + offset];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lid == 0) {
            const float decay = exp(A[scalar_index(b, h, t, args)]);
            const float extra = (t > 0) ? (decay * scale * reduce[0]) : 0.0f;
            dA[scalar_index(b, h, t, args)] = local_dA[scalar_index(b, h, t, args)] + extra;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

}}} // namespace meow::mamba::mamba2
