//
//  mamba2.metal
//  SuperKittens
//
//

#include "../../../meow.h"

namespace meow::mamba::mamba2_parallel {

enum : int {
    CHUNK_SIZE = 32,
    HEAD_DIM_QK = 64,
    HEAD_DIM_V = 64,
    STATE_SIZE = HEAD_DIM_QK * HEAD_DIM_V,
    TILE_M = 8,
    N_SIMDS = CHUNK_SIZE / TILE_M,
    PRECOMP_THREADS = N_SIMDS * 32,
    SCAN_THREADS = 128
};

struct Mamba2ParallelArgs {
    uint batch;
    uint nheads;
    uint seq_len;
    uint n_chunks;
};

METAL_FUNC uint state_offset(uint batch, uint head, uint nheads, uint n_chunks, uint chunk_idx) {
    return ((((batch * nheads) + head) * n_chunks + chunk_idx) * STATE_SIZE);
}

METAL_FUNC uint chunk_scalar_offset(uint batch, uint head, uint nheads, uint n_chunks, uint chunk_idx) {
    return (((batch * nheads) + head) * n_chunks + chunk_idx);
}

METAL_FUNC uint local_out_offset(uint batch, uint head, uint nheads, uint seq_len, uint token_idx) {
    return ((((batch * nheads) + head) * seq_len + token_idx) * HEAD_DIM_V);
}

METAL_FUNC uint q_scaled_offset(uint batch, uint head, uint nheads, uint seq_len, uint token_idx) {
    return ((((batch * nheads) + head) * seq_len + token_idx) * HEAD_DIM_QK);
}

[[host_name("mamba2_precompute_32_64_64")]]
[[kernel, max_total_threads_per_threadgroup(PRECOMP_THREADS)]]
void mamba2_precompute_32_64_64(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device const float* A [[buffer(3)]],
    device half* local_out [[buffer(4)]],
    device half* q_scaled [[buffer(5)]],
    device float* chunk_decay [[buffer(6)]],
    device float* chunk_update [[buffer(7)]],
    constant Mamba2ParallelArgs& args [[buffer(8)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]])
{
    if (gid.x >= args.batch || gid.y >= args.nheads || gid.z >= args.n_chunks) return;

    const uint batch = gid.x;
    const uint head = gid.y;
    const uint chunk_idx = gid.z;
    const uint token_start = chunk_idx * CHUNK_SIZE;
    const uint chunk_len = min(uint(CHUNK_SIZE), args.seq_len - token_start);
    const uint chunk_base_qk = (((batch * args.nheads + head) * args.seq_len + token_start) * HEAD_DIM_QK);
    const uint chunk_base_v = (((batch * args.nheads + head) * args.seq_len + token_start) * HEAD_DIM_V);
    const uint out_base = (((batch * args.nheads + head) * args.seq_len + token_start) * HEAD_DIM_V);
    const uint q_scaled_base = (((batch * args.nheads + head) * args.seq_len + token_start) * HEAD_DIM_QK);

    threadgroup float As[CHUNK_SIZE];
    threadgroup float cumsum_scratch[CHUNK_SIZE / 32];
    threadgroup half Qs[CHUNK_SIZE * HEAD_DIM_QK];
    threadgroup half Ks[CHUNK_SIZE * HEAD_DIM_QK];
    threadgroup half Vs[CHUNK_SIZE * HEAD_DIM_V];
    threadgroup half scratch[CHUNK_SIZE * CHUNK_SIZE];
    threadgroup float update_chunk[STATE_SIZE];

    for (uint i = lid; i < CHUNK_SIZE; i += PRECOMP_THREADS)
        As[i] = (i < chunk_len) ? A[(batch * args.nheads + head) * args.seq_len + token_start + i] : 0.0f;
    for (uint i = lid; i < CHUNK_SIZE * HEAD_DIM_QK; i += PRECOMP_THREADS) {
        Qs[i] = (i < chunk_len * HEAD_DIM_QK) ? Q[chunk_base_qk + i] : half(0.0f);
        Ks[i] = (i < chunk_len * HEAD_DIM_QK) ? K[chunk_base_qk + i] : half(0.0f);
    }
    for (uint i = lid; i < CHUNK_SIZE * HEAD_DIM_V; i += PRECOMP_THREADS)
        Vs[i] = (i < chunk_len * HEAD_DIM_V) ? V[chunk_base_v + i] : half(0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    meow::tools::threadgroup_cumsum<float, CHUNK_SIZE>(As, cumsum_scratch, lid, lane_id, simd_id);

    for (uint idx = lid; idx < CHUNK_SIZE * HEAD_DIM_QK; idx += PRECOMP_THREADS) {
        const uint row = idx / HEAD_DIM_QK;
        const float decay = (row < chunk_len) ? metal::fast::exp(As[row]) : 0.0f;
        q_scaled[q_scaled_base + idx] = half(float(Qs[idx]) * decay);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        enum : int {
            TILE_ROWS = TILE_M / 8,
            QK_COLS = CHUNK_SIZE / 8,
            V_COLS = HEAD_DIM_V / 8
        };
        const uint row_base = simd_id * TILE_M;

        meow::mma::Tile<TILE_ROWS, QK_COLS> attn_block;
        attn_block.clear();
        meow::mma::mm_ABt<HEAD_DIM_QK, TILE_ROWS, QK_COLS>(
            attn_block, Qs + row_base * HEAD_DIM_QK, HEAD_DIM_QK, Ks, HEAD_DIM_QK);
        attn_block.copy_to_half(scratch + row_base * CHUNK_SIZE, CHUNK_SIZE);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint idx = lid; idx < CHUNK_SIZE * CHUNK_SIZE; idx += PRECOMP_THREADS) {
            const uint r = idx / CHUNK_SIZE;
            const uint c = idx % CHUNK_SIZE;
            const float mask = (r < chunk_len && c <= r)
                ? metal::fast::exp(As[r] - As[c])
                : 0.0f;
            scratch[idx] = half(float(scratch[idx]) * mask);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        meow::mma::Tile<TILE_ROWS, V_COLS> o_reg;
        o_reg.clear();
        meow::mma::mm_AB<CHUNK_SIZE, TILE_ROWS, V_COLS>(
            o_reg, scratch + row_base * CHUNK_SIZE, CHUNK_SIZE, Vs, HEAD_DIM_V);
        o_reg.store(Qs + row_base * HEAD_DIM_V, HEAD_DIM_V);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint idx = lid; idx < chunk_len * HEAD_DIM_V; idx += PRECOMP_THREADS)
        local_out[out_base + idx] = Qs[idx];
    for (uint idx = lid + chunk_len * HEAD_DIM_V; idx < CHUNK_SIZE * HEAD_DIM_V; idx += PRECOMP_THREADS)
        local_out[out_base + idx] = half(0.0f);

    for (uint idx = lid; idx < STATE_SIZE; idx += PRECOMP_THREADS)
        update_chunk[idx] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = 0; t < chunk_len; ++t) {
        const float prev_cs = (t > 0) ? As[t - 1] : 0.0f;
        const float decay_t = metal::fast::exp(As[t] - prev_cs);
        const uint k_base = t * HEAD_DIM_QK;
        const uint v_base = t * HEAD_DIM_V;
        for (uint idx = lid; idx < STATE_SIZE; idx += PRECOMP_THREADS) {
            const uint i = idx / HEAD_DIM_V;
            const uint j = idx % HEAD_DIM_V;
            update_chunk[idx] = decay_t * update_chunk[idx] +
                                float(Ks[k_base + i]) * float(Vs[v_base + j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint upd_base = state_offset(batch, head, args.nheads, args.n_chunks, chunk_idx);
    for (uint idx = lid; idx < STATE_SIZE; idx += PRECOMP_THREADS)
        chunk_update[upd_base + idx] = update_chunk[idx];

    if (lid == 0) {
        const float decay_last = metal::fast::exp(As[chunk_len - 1]);
        const uint scalar_idx = chunk_scalar_offset(batch, head, args.nheads, args.n_chunks, chunk_idx);
        chunk_decay[scalar_idx] = decay_last;
    }
}

[[host_name("mamba2_scan")]]
[[kernel, max_total_threads_per_threadgroup(SCAN_THREADS)]]
void mamba2_scan(
    device const float* chunk_decay [[buffer(0)]],
    device const float* chunk_update [[buffer(1)]],
    device const half* local_out [[buffer(2)]],
    device const half* q_scaled [[buffer(3)]],
    device half* O [[buffer(4)]],
    constant Mamba2ParallelArgs& args [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    if (gid.x >= args.batch || gid.y >= args.nheads) return;

    const uint batch = gid.x;
    const uint head = gid.y;
    threadgroup float state[STATE_SIZE];

    for (uint idx = lid; idx < STATE_SIZE; idx += SCAN_THREADS)
        state[idx] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint chunk_idx = 0; chunk_idx < args.n_chunks; ++chunk_idx) {
        const uint state_base = state_offset(batch, head, args.nheads, args.n_chunks, chunk_idx);
        const uint token_start = chunk_idx * CHUNK_SIZE;
        const uint chunk_len = min(uint(CHUNK_SIZE), args.seq_len - token_start);
        for (uint idx = lid; idx < chunk_len * HEAD_DIM_V; idx += SCAN_THREADS) {
            const uint t = idx / HEAD_DIM_V;
            const uint j = idx % HEAD_DIM_V;
            const uint token_idx = token_start + t;
            const uint out_base = local_out_offset(batch, head, args.nheads, args.seq_len, token_idx);
            const uint qs_base = q_scaled_offset(batch, head, args.nheads, args.seq_len, token_idx);
            float acc = float(local_out[out_base + j]);
            for (uint i = 0; i < HEAD_DIM_QK; ++i)
                acc += float(q_scaled[qs_base + i]) * state[i * HEAD_DIM_V + j];
            O[out_base + j] = half(acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float decay = chunk_decay[chunk_scalar_offset(batch, head, args.nheads, args.n_chunks, chunk_idx)];
        for (uint idx = lid; idx < STATE_SIZE; idx += SCAN_THREADS)
            state[idx] = decay * state[idx] + chunk_update[state_base + idx];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

} // namespace meow::mamba::mamba2_parallel
