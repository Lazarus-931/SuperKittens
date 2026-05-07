//
//  paged_attn.metal — Paged attention, single head per threadgroup
//  Merged K+V load, correct block-table indexing. Bc=32.
//

#include <metal_stdlib>
using namespace metal;

enum : uint { Bc = 32, THREADS = 128, D4_MAX = 32 };

[[host_name("paged_attn")]]
[[kernel, max_total_threads_per_threadgroup(THREADS)]]
void paged_attn(
    device const half* Q,
    device const half* K_cache,
    device const half* V_cache,
    device const int*  block_table,
    device const int*  seq_lens,
    device half* O,
    constant uint& num_seqs, constant uint& num_heads, constant uint& head_dim,
    constant uint& num_kv_heads, constant uint& block_size,
    constant uint& max_blocks, constant uint& seq_stride, constant uint& block_stride,
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  simd [[simdgroup_index_in_threadgroup]])
{
    const uint seq = gid.x, head = gid.y;
    if (seq >= num_seqs || head >= num_heads) return;

    const uint kvh = (head * num_kv_heads) / num_heads;
    const uint d4 = head_dim / 4;

    const size_t q_off = (size_t)seq * seq_stride + (size_t)head * head_dim;
    float4 q_reg = float4(reinterpret_cast<const device half4*>(Q + q_off)[lane]);

    float row_max = -INFINITY, row_sum = 0.0f;
    float4 out_vec = float4(0.0f);
    const float scale = 1.0f / sqrt(float(head_dim));

    threadgroup half4 k_tile[Bc * D4_MAX];
    threadgroup half4 v_tile[Bc * D4_MAX];

    const int seq_len = seq_lens[seq];
    const int num_blocks = (seq_len + block_size - 1) / block_size;
    const device int* bt = block_table + (size_t)seq * max_blocks;
    const uint head_stride = num_kv_heads * head_dim;  // stride per token within a block

    for (int bi = 0; bi < num_blocks; bi++) {
        int phys_block = bt[bi];
        if (phys_block < 0) continue;

        int tokens_in_block = min((int)block_size, seq_len - bi * (int)block_size);
        const size_t block_base = (size_t)phys_block * block_stride;

        for (uint tc = 0; tc < (uint)tokens_in_block; tc += Bc) {
            uint cur_rows = min(Bc, (uint)tokens_in_block - tc);
            uint load_count = cur_rows * d4;

            // Merged K+V load
            for (uint i = lid; i < load_count; i += THREADS) {
                uint r = i / d4, c = i % d4;
                size_t off = block_base + (size_t)(tc + r) * head_stride + (size_t)kvh * head_dim + (size_t)c * 4;
                k_tile[i] = reinterpret_cast<const device half4*>(K_cache + off)[0];
                v_tile[i] = reinterpret_cast<const device half4*>(V_cache + off)[0];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint r = 0; r < cur_rows; r++) {
                uint idx = r * d4 + lane;
                float score = simd_sum(dot(q_reg, float4(k_tile[idx]))) * scale;
                float new_max = max(row_max, score);
                float alpha = metal::fast::exp(row_max - new_max);
                float beta  = metal::fast::exp(score - new_max);
                row_sum = row_sum * alpha + beta;
                out_vec *= alpha;
                out_vec += beta * float4(v_tile[idx]);
                row_max = new_max;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const size_t o_off = (size_t)seq * seq_stride + (size_t)head * head_dim;
    reinterpret_cast<device half4*>(O + o_off)[lane] = half4(out_vec / row_sum);
}
