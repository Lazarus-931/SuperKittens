//
//  attn_d64.metal
//  SuperKittens — Flash Attention for d=64 (causal + noncausal)
//
//  1024 threads = 32 SIMD groups × 32 lanes.
//  Each SIMD owns one Q row.  Lanes cooperate via simd_sum.
//  K tiles in threadgroup memory; V from device; softmax in registers.
//

#include <metal_stdlib>
using namespace metal;

namespace meow::attn {

template<int Br, int Bc, int D>
struct fa_config {
    enum : int {
        BR = Br, BC = Bc, HEAD_DIM = D,
        N_THREADS   = BR * 32,
        QK_PER_LANE = D / 32,
        V_PER_LANE  = D / 32,
        D4          = D / 4,
    };
};

template<int N_THREADS, int Bc, int D>
static void load_k_tile(
    device const half* src, threadgroup half4* dst,
    uint col_start, uint seq, uint lid)
{
    constexpr uint D4 = D / 4;
    const uint total = Bc * D4;
    for (uint i = lid; i < total; i += N_THREADS) {
        uint c = i / D4, d4 = i % D4, k_col = col_start + c;
        if (k_col < seq) {
            dst[i] = reinterpret_cast<const device half4*>(src + (size_t)k_col * D)[d4];
        } else {
            dst[i] = half4(0.0h);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

template<int Br, int Bc, int D, bool Causal>
[[kernel, max_total_threads_per_threadgroup(1024)]]
void fa_kernel(
    device const half* Q     [[buffer(0)]],
    device const half* K     [[buffer(1)]],
    device const half* V     [[buffer(2)]],
    device half* O           [[buffer(3)]],
    constant uint& seq       [[buffer(4)]],
    constant uint& n_heads   [[buffer(5)]],
    uint3 gid   [[threadgroup_position_in_grid]],
    uint  lid   [[thread_index_in_threadgroup]],
    uint  simd  [[simdgroup_index_in_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]])
{
    using Cfg = fa_config<Br, Bc, D>;

    const uint head = gid.x, tile_idx = gid.y, batch = gid.z;
    if (head >= n_heads) return;
    const uint q_row = tile_idx * Br + simd;
    if (q_row >= seq) return;

    const size_t off = (size_t)(batch * n_heads + head) * seq * D;

    const float scale = 1.0f / sqrt(float(D));
    float q_reg[Cfg::QK_PER_LANE];
    const device half* q_src = Q + off + (size_t)q_row * D + lane * Cfg::QK_PER_LANE;
    for (int i = 0; i < Cfg::QK_PER_LANE; i++) q_reg[i] = float(q_src[i]) * scale;

    float max_score = -INFINITY, sum_exp = 0.0f;
    float o_acc[Cfg::V_PER_LANE];
    for (int i = 0; i < Cfg::V_PER_LANE; i++) o_acc[i] = 0.0f;

    threadgroup half4 K_tile[Bc * Cfg::D4];

    const uint n_kv_tiles = Causal
        ? (min(seq, q_row + 1) + Bc - 1) / Bc
        : (seq + Bc - 1) / Bc;

    for (uint kv_tile = 0; kv_tile < n_kv_tiles; kv_tile++) {
        uint col_start = kv_tile * Bc;
        load_k_tile<Cfg::N_THREADS, Bc, D>(K + off, K_tile, col_start, seq, lid);

        for (int k_idx = 0; k_idx < Bc; k_idx++) {
            uint k_col = col_start + k_idx;
            if (k_col >= seq) continue;
            if (Causal && k_col > q_row) continue;

            const threadgroup half4* k4 = K_tile + k_idx * Cfg::D4;
            float dp = 0.0f;
            for (int i = 0; i < Cfg::QK_PER_LANE; i++) {
                uint half_idx = lane * Cfg::QK_PER_LANE + i;
                dp += q_reg[i] * float(
                    reinterpret_cast<const threadgroup half*>(k4 + half_idx / 4)[half_idx % 4]);
            }
            float score = simd_sum(dp);

            float new_max   = max(max_score, score);
            float factor    = metal::fast::exp(max_score - new_max);
            float exp_score = metal::fast::exp(score - new_max);
            max_score = new_max;
            sum_exp   = sum_exp * factor + exp_score;

            for (int i = 0; i < Cfg::V_PER_LANE; i++) {
                o_acc[i] = fma(exp_score,
                    float(reinterpret_cast<const device half*>(
                        V + off + (size_t)k_col * D)[lane * Cfg::V_PER_LANE + i]),
                    o_acc[i] * factor);
            }
        }
    }

    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    device half* o_row = O + off + (size_t)q_row * D + lane * Cfg::V_PER_LANE;
    for (int i = 0; i < Cfg::V_PER_LANE; i++) o_row[i] = half(o_acc[i] * inv_sum);
}

// d=64 instantiations
template [[host_name("fa_causal_64")]]
[[kernel]] void fa_kernel<32, 32, 64, true>(
    device const half*, device const half*, device const half*,
    device half*, constant uint&, constant uint&, uint3, uint, uint, uint);

template [[host_name("fa_noncausal_64")]]
[[kernel]] void fa_kernel<32, 32, 64, false>(
    device const half*, device const half*, device const half*,
    device half*, constant uint&, constant uint&, uint3, uint, uint, uint);

} // namespace meow::attn
