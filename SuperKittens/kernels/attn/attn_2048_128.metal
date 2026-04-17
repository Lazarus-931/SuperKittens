//
//  attn_2048_128.metal
//  SuperKittens
//
//  Fused attention for seq=2048, d=128
//  2×2 SIMD layout: 2 row groups × 2 col groups
//  Q persistent; K/V share one buffer; BlockMMA for tiled GEMM

#include "../../meow.h"
#include "params.h"
#include "types.h"

using namespace meow;
using namespace metal;


kernel void attn_2048_128(
    device const half* Q   [[buffer(0)]],
    device const half* K   [[buffer(1)]],
    device const half* V   [[buffer(2)]],
    device half* O         [[buffer(3)]],
    constant meow::attn::Params& param [[buffer(4)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]])
{
    constexpr uint D = 128;
    constexpr uint SEQ = 2048;
    constexpr uint KEY_TILES = SEQ / 128;
    constexpr float SCALE = (1.0f / 11.3137f) * 1.4426950408889634f;
    constexpr uint KV_STRIDE = 134;
    constexpr uint Q_FULL_STRIDE = 136;

    threadgroup half Qs[16 * Q_FULL_STRIDE];
    threadgroup half KVs[16 * KV_STRIDE];
    threadgroup half simd_scratch[4 * 8];

    uint tileRow = gid.y * 16;
    uint sr = (simd_id / 2) * 8;
    uint sc = (simd_id % 2) * 64;
    uint partner = simd_id ^ 1;
    uint col_group = simd_id % 2;

    using namespace meow::tools;
    short my_row = Frag::get_coord(lane_id).y;


    BlockMMA<64> output;
    output.clear();
    float rmax = -INFINITY;
    float rsum = 0.0f;


    for (uint i = lid; i < 16 * D; i += 128) {
        uint r = i / D, c = i % D;
        Qs[r * Q_FULL_STRIDE + c] = half(Q[(tileRow + r) * D + c]) * SCALE;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = 0; t < KEY_TILES; t++) {

        // ── QK^T ──
        BlockMMA<64> scores;
        scores.clear();

        for (uint kb = 0; kb < D; kb += 16) {
            for (uint i = lid; i < 16 * 128; i += 128) {
                uint r = i % 16, c = i / 16;
                KVs[r * KV_STRIDE + c] = K[(t * 128 + c) * D + kb + r];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            scores.mma<16>(Qs + sr * Q_FULL_STRIDE + kb, Q_FULL_STRIDE,
                           KVs + sc, KV_STRIDE);

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        
        Tile<1, 8> S;
        S.set_coord(lane_id);
        S.from_simd(scores.acc);

        half tile_max[1];
        S.row_max(tile_max);
        simd_scratch[simd_id * 8 + my_row] = tile_max[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_max[0] = max(tile_max[0], simd_scratch[partner * 8 + my_row]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float old_max = rmax;
        rmax = max(rmax, float(tile_max[0]));
        float rescale = metal::fast::exp2(old_max - rmax);
        rsum *= rescale;

        
        Tile<1, 8> Out;
        Out.set_coord(lane_id);
        Out.from_simd(output.acc);
        float rescale_arr[1] = {rescale};
        Out.row_scale(rescale_arr);
        for (int i = 0; i < 8; i++) Out.at(0, i).to_simd(output.acc[i]);

        S.row_softmax_exp2(tile_max);
        half tile_sum[1];
        S.row_sum(tile_sum);
        simd_scratch[simd_id * 8 + my_row] = tile_sum[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_sum[0] += simd_scratch[partner * 8 + my_row];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        rsum += tile_sum[0];

        
        for (int i = 0; i < 8; i++)
            S.at(0, i).store(KVs + sr * KV_STRIDE + sc + i * 8, KV_STRIDE);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        
        uint v_col_off = col_group * 68;
        threadgroup half* my_Vs = Qs + v_col_off;

        for (uint vk = 0; vk < 128; vk += 16) {
            uint pair_lid = (simd_id / 2) * 32 + lane_id;
            for (uint i = pair_lid; i < 16 * 64; i += 64) {
                uint r = i / 64, c = i % 64;
                my_Vs[r * Q_FULL_STRIDE + c] = V[(t * 128 + vk + r) * D + sc + c];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            output.mma<16>(KVs + sr * KV_STRIDE + vk, KV_STRIDE,
                           my_Vs, Q_FULL_STRIDE);

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        
        if (t + 1 < KEY_TILES) {
            for (uint i = lid; i < 16 * D; i += 128) {
                uint r = i / D, c = i % D;
                Qs[r * Q_FULL_STRIDE + c] = half(Q[(tileRow + r) * D + c]) * SCALE;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    
    Tile<1, 8> O_final;
    O_final.set_coord(lane_id);
    O_final.from_simd(output.acc);

    float inv_sum[1] = {1.0f / rsum};
    O_final.row_scale(inv_sum);

    for (int j = 0; j < 8; j++)
        O_final.at(0, j).store(KVs + sr * KV_STRIDE + sc + j * 8, KV_STRIDE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < 16 * D; i += 128) {
        uint r = i / D, c = i % D;
        O[(tileRow + r) * D + c] = KVs[r * KV_STRIDE + c];
    }
}
