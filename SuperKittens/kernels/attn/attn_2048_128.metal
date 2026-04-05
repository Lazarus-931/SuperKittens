//
//  attn_2048_128.metal
//  SuperKittens
//
//  Specialized fused attention for seq=2048, d=128
//  2×2 SIMD layout: 2 row groups × 2 col groups
//  Q reloaded per k-block to shrink shared memory → better occupancy

#include <metal_stdlib>
#include "params.h"
#include "tools/tile.h"
#include "ops.h"
#include "types.h"

using namespace superkittens;
using namespace metal;


kernel void attn_2048_128(
    device const half* Q   [[buffer(0)]],
    device const half* K   [[buffer(1)]],
    device const half* V   [[buffer(2)]],
    device half* O         [[buffer(3)]],
    constant superkittens::attn::Params& param [[buffer(4)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]])
{
    constexpr uint D = 128;
    constexpr uint SEQ = 2048;
    constexpr uint KEY_TILES = SEQ / 128;
    constexpr float SCALE = 1.0f / 11.3137f;
    constexpr uint Q_STRIDE = 17;
    constexpr uint KV_STRIDE = 129;
    constexpr uint V_STRIDE = 65;      

    // Shared memory layout (~13.7 KB total):
    //   Qs:      16 × 17  = 1.1 KB   (Q reloaded per k-block)
    //   KVs:     16 × 129 = 8.3 KB   (K^T, then softmax scores)
    //   Vs:   2× 16 × 65  = 4.2 KB   (V: one block per col group)
    //   scratch:  32 floats = 0.1 KB
    threadgroup half Qs[16 * Q_STRIDE];
    threadgroup half KVs[16 * KV_STRIDE];
    threadgroup half Vs[2 * 16 * V_STRIDE];
    threadgroup half simd_scratch[4 * 8];

    uint tileRow = gid.y * 16;
    uint sr = (simd_id / 2) * 8;
    uint sc = (simd_id % 2) * 64;
    uint partner = simd_id ^ 1;
    uint col_group = simd_id % 2;      

    float rmax = -INFINITY;
    float rsum = 0.0f;
    simdgroup_float8x8 output_acc[8] = {};

    using namespace superkittens::tools;
    short my_row = Frag::get_coord(lane_id).y;

    for (uint t = 0; t < KEY_TILES; t++) {
        simdgroup_float8x8 scores[8] = {};

        for (uint kb = 0; kb < D; kb += 16) {
            // Load Q tile: 16 rows × 16 cols
            for (uint i = lid; i < 16 * 16; i += 128) {
                uint r = i / 16, c = i % 16;
                Qs[r * Q_STRIDE + c] = half(Q[(tileRow + r) * D + kb + c]) * SCALE;
            }
            // Load K^T tile: 16 rows × 128 cols
            for (uint i = lid; i < 16 * 128; i += 128) {
                uint r = i / 128, c = i % 128;
                KVs[r * KV_STRIDE + c] = half(K[(t * 128 + c) * D + kb + r]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_half8x8 a0, b0, b1, b2, b3, b4, b5, b6, b7;
            
            // First 8 rows of K^T × Q rows at sr
            simdgroup_load(a0, Qs + sr * Q_STRIDE, Q_STRIDE);
            simdgroup_load(b0, KVs + sc,      KV_STRIDE);
            simdgroup_load(b1, KVs + sc + 8,  KV_STRIDE);
            simdgroup_load(b2, KVs + sc + 16, KV_STRIDE);
            simdgroup_load(b3, KVs + sc + 24, KV_STRIDE);
            simdgroup_load(b4, KVs + sc + 32, KV_STRIDE);
            simdgroup_load(b5, KVs + sc + 40, KV_STRIDE);
            simdgroup_load(b6, KVs + sc + 48, KV_STRIDE);
            simdgroup_load(b7, KVs + sc + 56, KV_STRIDE);
            simdgroup_multiply_accumulate(scores[0], a0, b0, scores[0]);
            simdgroup_multiply_accumulate(scores[1], a0, b1, scores[1]);
            simdgroup_multiply_accumulate(scores[2], a0, b2, scores[2]);
            simdgroup_multiply_accumulate(scores[3], a0, b3, scores[3]);
            simdgroup_multiply_accumulate(scores[4], a0, b4, scores[4]);
            simdgroup_multiply_accumulate(scores[5], a0, b5, scores[5]);
            simdgroup_multiply_accumulate(scores[6], a0, b6, scores[6]);
            simdgroup_multiply_accumulate(scores[7], a0, b7, scores[7]);

            // Second 8 rows of K^T
            simdgroup_load(a0, Qs + sr * Q_STRIDE + 8, Q_STRIDE);
            simdgroup_load(b0, KVs + 8 * KV_STRIDE + sc,      KV_STRIDE);
            simdgroup_load(b1, KVs + 8 * KV_STRIDE + sc + 8,  KV_STRIDE);
            simdgroup_load(b2, KVs + 8 * KV_STRIDE + sc + 16, KV_STRIDE);
            simdgroup_load(b3, KVs + 8 * KV_STRIDE + sc + 24, KV_STRIDE);
            simdgroup_load(b4, KVs + 8 * KV_STRIDE + sc + 32, KV_STRIDE);
            simdgroup_load(b5, KVs + 8 * KV_STRIDE + sc + 40, KV_STRIDE);
            simdgroup_load(b6, KVs + 8 * KV_STRIDE + sc + 48, KV_STRIDE);
            simdgroup_load(b7, KVs + 8 * KV_STRIDE + sc + 56, KV_STRIDE);
            simdgroup_multiply_accumulate(scores[0], a0, b0, scores[0]);
            simdgroup_multiply_accumulate(scores[1], a0, b1, scores[1]);
            simdgroup_multiply_accumulate(scores[2], a0, b2, scores[2]);
            simdgroup_multiply_accumulate(scores[3], a0, b3, scores[3]);
            simdgroup_multiply_accumulate(scores[4], a0, b4, scores[4]);
            simdgroup_multiply_accumulate(scores[5], a0, b5, scores[5]);
            simdgroup_multiply_accumulate(scores[6], a0, b6, scores[6]);
            simdgroup_multiply_accumulate(scores[7], a0, b7, scores[7]);

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ── Step 2: online softmax in registers ──
        Tile<1, 8> S;
        S.set_coord(lane_id);
        S.from_simd(scores);

        // Row max — local then cross-SIMD (per-row, between partners)
        half tile_max[1];
        S.row_max(tile_max);
        simd_scratch[simd_id * 8 + my_row] = tile_max[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_max[0] = max(tile_max[0], simd_scratch[partner * 8 + my_row]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax state update
        float old_max = rmax;
        rmax = max(rmax, float(tile_max[0]));
        float rescale = metal::fast::exp(old_max - rmax);
        rsum *= rescale;

        // Rescale previous output in registers
        Tile<1, 8> Out;
        Out.set_coord(lane_id);
        Out.from_simd(output_acc);
        float rescale_arr[1] = {rescale};
        Out.row_scale(rescale_arr);

        // exp(score - max) + row sum, then cross-SIMD sum (per-row)
        S.row_softmax_exp(tile_max);
        half tile_sum[1];
        S.row_sum(tile_sum);
        simd_scratch[simd_id * 8 + my_row] = tile_sum[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_sum[0] += simd_scratch[partner * 8 + my_row];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        rsum += tile_sum[0];

        // ── Step 3: scores × V ──
        for (int i = 0; i < 8; i++) Out.at(0, i).to_simd(output_acc[i]);

        // Store softmax scores to KVs (full 16×128)
        for (int i = 0; i < 8; i++)
            S.at(0, i).store(KVs + sr * KV_STRIDE + sc + i * 8, KV_STRIDE);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup half* my_Vs = Vs + col_group * 16 * V_STRIDE;

        for (uint vk = 0; vk < 128; vk += 16) {
            uint pair_lid = (simd_id / 2) * 32 + lane_id;
            for (uint i = pair_lid; i < 16 * 64; i += 64) {
                uint r = i / 64, c = i % 64;
                my_Vs[r * V_STRIDE + c] = half(V[(t * 128 + vk + r) * D + sc + c]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_half8x8 s0, s1, v0, v1, v2, v3, v4, v5, v6, v7;
            simdgroup_load(s0, KVs + sr * KV_STRIDE + vk,     KV_STRIDE);
            simdgroup_load(s1, KVs + sr * KV_STRIDE + vk + 8, KV_STRIDE);

            // First 8 rows of V
            simdgroup_load(v0, my_Vs,      V_STRIDE);
            simdgroup_load(v1, my_Vs + 8,  V_STRIDE);
            simdgroup_load(v2, my_Vs + 16, V_STRIDE);
            simdgroup_load(v3, my_Vs + 24, V_STRIDE);
            simdgroup_load(v4, my_Vs + 32, V_STRIDE);
            simdgroup_load(v5, my_Vs + 40, V_STRIDE);
            simdgroup_load(v6, my_Vs + 48, V_STRIDE);
            simdgroup_load(v7, my_Vs + 56, V_STRIDE);
            simdgroup_multiply_accumulate(output_acc[0], s0, v0, output_acc[0]);
            simdgroup_multiply_accumulate(output_acc[1], s0, v1, output_acc[1]);
            simdgroup_multiply_accumulate(output_acc[2], s0, v2, output_acc[2]);
            simdgroup_multiply_accumulate(output_acc[3], s0, v3, output_acc[3]);
            simdgroup_multiply_accumulate(output_acc[4], s0, v4, output_acc[4]);
            simdgroup_multiply_accumulate(output_acc[5], s0, v5, output_acc[5]);
            simdgroup_multiply_accumulate(output_acc[6], s0, v6, output_acc[6]);
            simdgroup_multiply_accumulate(output_acc[7], s0, v7, output_acc[7]);

            // Second 8 rows of V
            simdgroup_load(v0, my_Vs + 8 * V_STRIDE,      V_STRIDE);
            simdgroup_load(v1, my_Vs + 8 * V_STRIDE + 8,  V_STRIDE);
            simdgroup_load(v2, my_Vs + 8 * V_STRIDE + 16, V_STRIDE);
            simdgroup_load(v3, my_Vs + 8 * V_STRIDE + 24, V_STRIDE);
            simdgroup_load(v4, my_Vs + 8 * V_STRIDE + 32, V_STRIDE);
            simdgroup_load(v5, my_Vs + 8 * V_STRIDE + 40, V_STRIDE);
            simdgroup_load(v6, my_Vs + 8 * V_STRIDE + 48, V_STRIDE);
            simdgroup_load(v7, my_Vs + 8 * V_STRIDE + 56, V_STRIDE);
            simdgroup_multiply_accumulate(output_acc[0], s1, v0, output_acc[0]);
            simdgroup_multiply_accumulate(output_acc[1], s1, v1, output_acc[1]);
            simdgroup_multiply_accumulate(output_acc[2], s1, v2, output_acc[2]);
            simdgroup_multiply_accumulate(output_acc[3], s1, v3, output_acc[3]);
            simdgroup_multiply_accumulate(output_acc[4], s1, v4, output_acc[4]);
            simdgroup_multiply_accumulate(output_acc[5], s1, v5, output_acc[5]);
            simdgroup_multiply_accumulate(output_acc[6], s1, v6, output_acc[6]);
            simdgroup_multiply_accumulate(output_acc[7], s1, v7, output_acc[7]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // ── Normalize + write to device ──
    Tile<1, 8> O_final;
    O_final.set_coord(lane_id);
    O_final.from_simd(output_acc);

    float inv_sum[1] = {1.0f / rsum};
    O_final.row_scale(inv_sum);

    for (int j = 0; j < 8; j++)
        O_final.at(0, j).store(O + (tileRow + sr) * D + sc + j * 8, D);
}
