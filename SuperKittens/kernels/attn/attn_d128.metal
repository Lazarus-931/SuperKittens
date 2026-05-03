//
//  attn_d128.metal
//  SuperKittens — MHA for d=128 with half4 SIMD fast path (causal + noncausal)
//
//  128 threads, row-per-SIMD online softmax, cooperative K/V tile loading.
//  Falls back to generic scalar path for other head dims.
//

#include <metal_stdlib>
using namespace metal;

#include "../../include/ops/ops.metal"
#include "ops.h"

namespace meow::attn {

enum : uint {
    MHA_THREADS      = 128,
    MHA_ROWS_PER_GRP = 4,
    MHA_KEY_TILE     = 32
};

// ── d=128 fast path: half4 + simd_sum(dot(float4, float4)) ────────

// causal
static inline void mha_causal_128(
    device const half* Q, device const half* K, device const half* V,
    device half* O, uint seq, uint row, uint lane_id, uint lid,
    threadgroup half4* k_tile, threadgroup half4* v_tile)
{
    const float scale = meow::ops::fast_rsqrt(128.0f);
    float row_max = -INFINITY, row_sum = 0.0f;
    const size_t q_base = (size_t)row * 128;
    const device half4* q_row = reinterpret_cast<const device half4*>(Q + q_base);
    const float4 qv = float4(q_row[lane_id]);
    float4 out_vec = float4(0.0f);

    for (uint tile_col = 0; tile_col <= row; tile_col += MHA_KEY_TILE) {
        for (uint i = lid; i < MHA_KEY_TILE * 32; i += MHA_THREADS) {
            uint local_row = i / 32, local_col = i % 32;
            uint global_col = tile_col + local_row;
            if (global_col < seq && global_col <= row) {
                const device half4* k_row = reinterpret_cast<const device half4*>(K + (size_t)global_col * 128);
                const device half4* v_row = reinterpret_cast<const device half4*>(V + (size_t)global_col * 128);
                k_tile[i] = k_row[local_col]; v_tile[i] = v_row[local_col];
            } else { k_tile[i] = half4(0.0h); v_tile[i] = half4(0.0h); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tile_limit = min(MHA_KEY_TILE, row + 1 - tile_col);
        for (uint local_col = 0; local_col < tile_limit; local_col++) {
            uint idx = local_col * 32 + lane_id;
            float score = simd_sum(dot(qv, float4(k_tile[idx]))) * scale;
            float new_max = max(row_max, score);
            float alpha = meow::ops::fast_exp(row_max - new_max);
            float beta  = meow::ops::fast_exp(score - new_max);
            row_sum = row_sum * alpha + beta;
            out_vec *= alpha; out_vec += beta * float4(v_tile[idx]);
            row_max = new_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device half4* o_row = reinterpret_cast<device half4*>(O + q_base);
    o_row[lane_id] = half4(out_vec / row_sum);
}

// noncausal
static inline void mha_noncausal_128(
    device const half* Q, device const half* K, device const half* V,
    device half* O, uint seq, uint row, uint lane_id, uint lid,
    threadgroup half4* k_tile, threadgroup half4* v_tile)
{
    const float scale = meow::ops::fast_rsqrt(128.0f);
    float row_max = -INFINITY, row_sum = 0.0f;
    const size_t q_base = (size_t)row * 128;
    const device half4* q_row = reinterpret_cast<const device half4*>(Q + q_base);
    const float4 qv = float4(q_row[lane_id]);
    float4 out_vec = float4(0.0f);

    for (uint tile_col = 0; tile_col < seq; tile_col += MHA_KEY_TILE) {
        for (uint i = lid; i < MHA_KEY_TILE * 32; i += MHA_THREADS) {
            uint local_row = i / 32, local_col = i % 32;
            uint global_col = tile_col + local_row;
            if (global_col < seq) {
                const device half4* k_row = reinterpret_cast<const device half4*>(K + (size_t)global_col * 128);
                const device half4* v_row = reinterpret_cast<const device half4*>(V + (size_t)global_col * 128);
                k_tile[i] = k_row[local_col]; v_tile[i] = v_row[local_col];
            } else { k_tile[i] = half4(0.0h); v_tile[i] = half4(0.0h); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tile_limit = min(MHA_KEY_TILE, seq - tile_col);
        for (uint local_col = 0; local_col < tile_limit; local_col++) {
            uint idx = local_col * 32 + lane_id;
            float score = simd_sum(dot(qv, float4(k_tile[idx]))) * scale;
            float new_max = max(row_max, score);
            float alpha = meow::ops::fast_exp(row_max - new_max);
            float beta  = meow::ops::fast_exp(score - new_max);
            row_sum = row_sum * alpha + beta;
            out_vec *= alpha; out_vec += beta * float4(v_tile[idx]);
            row_max = new_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device half4* o_row = reinterpret_cast<device half4*>(O + q_base);
    o_row[lane_id] = half4(out_vec / row_sum);
}

// ── generic fallback (arbitrary head_dim) ──────────────────────────

template<bool Causal>
static inline void mha_generic(
    device const half* Q, device const half* K, device const half* V,
    device half* O, uint seq, uint head_dim, uint row, uint lane_id)
{
    const float scale = meow::ops::fast_rsqrt(float(head_dim));
    meow::attn::SoftmaxState state = { -INFINITY, 0.0f };
    float4 out_vec = float4(0.0f);
    uint c0 = lane_id, c1 = lane_id + 32, c2 = lane_id + 64, c3 = lane_id + 96;
    const size_t q_base = (size_t)row * head_dim;

    uint limit = Causal ? row + 1 : seq;
    for (uint col = 0; col < limit; ++col) {
        const size_t k_base = (size_t)col * head_dim;
        float partial_dot = 0.0f;
        for (uint kk = lane_id; kk < head_dim; kk += 32)
            partial_dot += float(Q[q_base + kk]) * float(K[k_base + kk]);
        float score = simd_sum(partial_dot) * scale;
        float new_max = max(state.row_max, score);
        float alpha = meow::ops::fast_exp(state.row_max - new_max);
        float beta  = meow::ops::fast_exp(score - new_max);
        state.row_sum = state.row_sum * alpha + beta;
        out_vec *= alpha;
        if (c0 < head_dim) out_vec.x += beta * float(V[k_base + c0]);
        if (c1 < head_dim) out_vec.y += beta * float(V[k_base + c1]);
        if (c2 < head_dim) out_vec.z += beta * float(V[k_base + c2]);
        if (c3 < head_dim) out_vec.w += beta * float(V[k_base + c3]);
        state.row_max = new_max;
    }
    float inv_sum = 1.0f / state.row_sum;
    if (c0 < head_dim) O[q_base + c0] = half(out_vec.x * inv_sum);
    if (c1 < head_dim) O[q_base + c1] = half(out_vec.y * inv_sum);
    if (c2 < head_dim) O[q_base + c2] = half(out_vec.z * inv_sum);
    if (c3 < head_dim) O[q_base + c3] = half(out_vec.w * inv_sum);
}

// ── kernel entry points ────────────────────────────────────────────

template<bool Causal>
[[kernel, max_total_threads_per_threadgroup(MHA_THREADS)]]
void mha_kernel(
    device const half* Q      [[buffer(0)]],
    device const half* K      [[buffer(1)]],
    device const half* V      [[buffer(2)]],
    device half* O            [[buffer(3)]],
    constant uint& seq        [[buffer(4)]],
    constant uint& head_dim   [[buffer(5)]],
    constant uint& num_heads  [[buffer(6)]],
    uint2 gid     [[threadgroup_position_in_grid]],
    uint  lid     [[thread_index_in_threadgroup]],
    uint  lane_id [[thread_index_in_simdgroup]],
    uint  simd_id [[simdgroup_index_in_threadgroup]])
{
    const uint head = gid.x, row = gid.y * MHA_ROWS_PER_GRP + simd_id;
    if (head >= num_heads || row >= seq) return;
    const size_t off = (size_t)head * seq * head_dim;

    threadgroup half4 k_tile[MHA_KEY_TILE * 32];
    threadgroup half4 v_tile[MHA_KEY_TILE * 32];

    if (head_dim == 128) {
        if (Causal) mha_causal_128(Q + off, K + off, V + off, O + off, seq, row, lane_id, lid, k_tile, v_tile);
        else        mha_noncausal_128(Q + off, K + off, V + off, O + off, seq, row, lane_id, lid, k_tile, v_tile);
        return;
    }
    mha_generic<Causal>(Q + off, K + off, V + off, O + off, seq, head_dim, row, lane_id);
}

template [[host_name("mha_causal")]]
[[kernel]] void mha_kernel<true>(
    device const half*, device const half*, device const half*,
    device half*, constant uint&, constant uint&, constant uint&,
    uint2, uint, uint, uint);

template [[host_name("mha_noncausal")]]
[[kernel]] void mha_kernel<false>(
    device const half*, device const half*, device const half*,
    device half*, constant uint&, constant uint&, constant uint&,
    uint2, uint, uint, uint);

} // namespace meow::attn
