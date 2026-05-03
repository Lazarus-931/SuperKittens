//
//  attn.metal — Unified attention: d=64 FA + d=128 MHA
//  Single MHA_Params buffer at slot 4.  p.causal flag controls causal/noncausal.
//
//  d=64:  Br=32, Bc=128, K+V in threadgroup, 1024 threads.
//  d=128: Br=4,  Bc=128, K in threadgroup, V from device, 128 threads.
//

#include <metal_stdlib>
using namespace metal;

struct MHA_Params { uint seq, head_dim, num_heads, causal; };

// ═══════════════════ d=128, Bc=128, Br=4, 128 threads ═══════════════════

[[host_name("mha_causal")]]
[[kernel, max_total_threads_per_threadgroup(128)]]
void mha_causal(
    device const half* Q     [[buffer(0)]],
    device const half* K     [[buffer(1)]],
    device const half* V     [[buffer(2)]],
    device half* O           [[buffer(3)]],
    constant MHA_Params& p   [[buffer(4)]],
    uint3 gid  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  simd [[simdgroup_index_in_threadgroup]])
{
    const uint head = gid.x, batch = gid.z;
    if (head >= p.num_heads) return;
    const uint row = gid.y * 4 + simd;
    if (row >= p.seq) return;
    const size_t off = (size_t)(batch * p.num_heads + head) * p.seq * 128;

    const float scale = 1.0f / sqrt(128.0f);
    float row_max = -INFINITY, row_sum = 0.0f;

    const device half4* q_row = reinterpret_cast<const device half4*>(Q + off + (size_t)row * 128);
    const float4 qv = float4(q_row[lane]);
    float4 out_vec = float4(0.0f);

    threadgroup half4 k_tile[128 * 32];  // 4096 half4 = 32KB

    const uint limit = p.causal ? (row + 1) : p.seq;

    for (uint tc = 0; tc < limit; tc += 128) {
        for (uint i = lid; i < 128 * 32; i += 128) {
            uint lr = i / 32, lc = i % 32, gc = tc + lr;
            k_tile[i] = (gc < limit)
                ? reinterpret_cast<const device half4*>(K + off + (size_t)gc * 128)[lc]
                : half4(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tl = min(128u, limit - tc);
        for (uint lc = 0; lc < tl; lc++) {
            uint idx = lc * 32 + lane;
            float score = simd_sum(dot(qv, float4(k_tile[idx]))) * scale;
            float new_max = max(row_max, score);
            float alpha = metal::fast::exp(row_max - new_max);
            float beta  = metal::fast::exp(score - new_max);
            row_sum = row_sum * alpha + beta;
            out_vec *= alpha;
            uint gc = tc + lc;
            out_vec += beta * float4(reinterpret_cast<const device half4*>(V + off + (size_t)gc * 128)[lane]);
            row_max = new_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    reinterpret_cast<device half4*>(O + off + (size_t)row * 128)[lane] = half4(out_vec / row_sum);
}


// ═══════════════════ d=64, Bc=128, Br=32, 1024 threads ═══════════════════

[[host_name("fa_causal_64")]]
[[kernel, max_total_threads_per_threadgroup(1024)]]
void fa_causal_64(
    device const half* Q     [[buffer(0)]],
    device const half* K     [[buffer(1)]],
    device const half* V     [[buffer(2)]],
    device half* O           [[buffer(3)]],
    constant MHA_Params& p   [[buffer(4)]],
    uint3 gid  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const uint head = gid.x, batch = gid.z;
    if (head >= p.num_heads) return;
    const uint q_row = gid.y * 32 + simd;
    if (q_row >= p.seq) return;
    const size_t off = (size_t)(batch * p.num_heads + head) * p.seq * 64;

    const float scale = 1.0f / sqrt(64.0f);
    float q_reg[2];
    const device half* q_src = Q + off + (size_t)q_row * 64 + lane * 2;
    q_reg[0] = float(q_src[0]) * scale;
    q_reg[1] = float(q_src[1]) * scale;

    float max_score = -INFINITY, sum_exp = 0.0f;
    float o_acc[2] = {0.0f, 0.0f};

    threadgroup half4 K_tile[128 * 16];  // 128×16 = 2048 half4 = 16KB
    threadgroup half4 V_tile[128 * 16];  // 16KB → 32KB total

    const uint limit = p.causal ? min(p.seq, q_row + 1) : p.seq;
    const uint n_tiles = (limit + 127) / 128;

    for (uint tile = 0; tile < n_tiles; tile++) {
        uint cs = tile * 128;

        for (uint i = lid; i < 128 * 16; i += 1024) {
            uint c = i / 16, d4 = i % 16, kc = cs + c;
            K_tile[i] = (kc < limit)
                ? reinterpret_cast<const device half4*>(K + off + (size_t)kc * 64)[d4]
                : half4(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = lid; i < 128 * 16; i += 1024) {
            uint c = i / 16, d4 = i % 16, kc = cs + c;
            V_tile[i] = (kc < limit)
                ? reinterpret_cast<const device half4*>(V + off + (size_t)kc * 64)[d4]
                : half4(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint ki = 0; ki < 128; ki++) {
            uint kc = cs + ki;
            if (kc >= limit) continue;
            if (p.causal && kc > q_row) continue;

            float dp = 0.0f;
            const threadgroup half4* k4 = K_tile + ki * 16;
            for (int i = 0; i < 2; i++) {
                uint hi = lane * 2 + i;
                dp += q_reg[i] * float(reinterpret_cast<const threadgroup half*>(k4 + hi/4)[hi%4]);
            }
            float score = simd_sum(dp);

            float new_max = max(max_score, score);
            float factor  = metal::fast::exp(max_score - new_max);
            float exp_s   = metal::fast::exp(score - new_max);
            max_score = new_max;
            sum_exp   = sum_exp * factor + exp_s;

            const threadgroup half4* v4 = V_tile + ki * 16;
            for (int i = 0; i < 2; i++) {
                uint hi = lane * 2 + i;
                o_acc[i] = fma(exp_s,
                    float(reinterpret_cast<const threadgroup half*>(v4 + hi/4)[hi%4]),
                    o_acc[i] * factor);
            }
        }
    }

    float inv = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    device half* orow = O + off + (size_t)q_row * 64 + lane * 2;
    orow[0] = half(o_acc[0] * inv);
    orow[1] = half(o_acc[1] * inv);
}

