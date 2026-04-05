//
//  attn.metal
//  SuperKittens
//
//  Created by Alazar Manakelew on 4/1/26.
//
//  Fused attention: Q×K^T → online softmax → scores×V in one kernel
//  No intermediate DRAM writes.
//
//  Changes from v1:
//    - Removed TILED_MMA macro, inlined everything for clarity
//    - V loop loads 16×32 per SIMD with half→float promotion
//    - Removed redundant score reload (scores stay in Bs throughout softmax+V multiply)
//    - Kept Cs for output rescale only (unavoidable on Metal)

#include <metal_stdlib>
using namespace metal;

kernel void fused_attention(
    device const half* Q   [[buffer(0)]],
    device const half* K   [[buffer(1)]],
    device const half* V   [[buffer(2)]],
    device half* O         [[buffer(3)]],
    constant uint& seq     [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float As[16 * 17];
    threadgroup float Bs[16 * 129];
    threadgroup float Cs[16 * 129];

    uint tileRow = gid.y * 16;
    uint sr = (simd_id / 4) * 8;
    uint sc = (simd_id % 4) * 32;

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    simdgroup_float8x8 output[4] = {};
    float rsqrt_d = 1.0f / sqrt(float(head_dim));

    for (uint tileCol = 0; tileCol < seq; tileCol += 128) {

        // ── Step 1: QK^T ──
        simdgroup_float8x8 scores[4] = {};
        for (uint kb = 0; kb < head_dim; kb += 16) {
            for (uint i = lid; i < 16 * 16; i += 128) {
                uint r = i / 16, c = i % 16;
                uint gr = tileRow + r, gc = kb + c;
                As[r * 17 + c] = (gr < seq && gc < head_dim) ? float(Q[gr * head_dim + gc]) : 0.0f;
            }
            for (uint i = lid; i < 16 * 128; i += 128) {
                uint r = i / 128, c = i % 128;
                uint gr = kb + r, gc = tileCol + c;
                Bs[r * 129 + c] = (gc < seq && gr < head_dim) ? float(K[gc * head_dim + gr]) : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_float8x8 a0, b0, b1, b2, b3;
            simdgroup_load(a0, As + sr * 17, 17);
            simdgroup_load(b0, Bs + sc, 129);
            simdgroup_load(b1, Bs + sc + 8, 129);
            simdgroup_load(b2, Bs + sc + 16, 129);
            simdgroup_load(b3, Bs + sc + 24, 129);
            simdgroup_multiply_accumulate(scores[0], a0, b0, scores[0]);
            simdgroup_multiply_accumulate(scores[1], a0, b1, scores[1]);
            simdgroup_multiply_accumulate(scores[2], a0, b2, scores[2]);
            simdgroup_multiply_accumulate(scores[3], a0, b3, scores[3]);

            simdgroup_load(a0, As + sr * 17 + 8, 17);
            simdgroup_load(b0, Bs + 8 * 129 + sc, 129);
            simdgroup_load(b1, Bs + 8 * 129 + sc + 8, 129);
            simdgroup_load(b2, Bs + 8 * 129 + sc + 16, 129);
            simdgroup_load(b3, Bs + 8 * 129 + sc + 24, 129);
            simdgroup_multiply_accumulate(scores[0], a0, b0, scores[0]);
            simdgroup_multiply_accumulate(scores[1], a0, b1, scores[1]);
            simdgroup_multiply_accumulate(scores[2], a0, b2, scores[2]);
            simdgroup_multiply_accumulate(scores[3], a0, b3, scores[3]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ── Step 2: Online softmax ──
        // dump scores to Bs
        simdgroup_store(scores[0], Bs + sr * 129 + sc, 129);
        simdgroup_store(scores[1], Bs + sr * 129 + sc + 8, 129);
        simdgroup_store(scores[2], Bs + sr * 129 + sc + 16, 129);
        simdgroup_store(scores[3], Bs + sr * 129 + sc + 24, 129);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // scale + max in one pass
        float tile_max = -INFINITY;
        for (uint i = lid; i < 16 * 128; i += 128) {
            uint idx = (i / 128) * 129 + (i % 128);
            Bs[idx] *= rsqrt_d;
            tile_max = max(tile_max, Bs[idx]);
        }
        tile_max = simd_max(tile_max);
        threadgroup float simd_maxes[4];
        if (lid % 32 == 0) simd_maxes[simd_id] = tile_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_max = max(max(simd_maxes[0], simd_maxes[1]),
                       max(simd_maxes[2], simd_maxes[3]));

        float old_max = row_max;
        row_max = max(row_max, tile_max);
        float scale = exp(old_max - row_max);
        row_sum *= scale;

        // rescale previous output via Cs (only when max changed)
        if (tile_max > old_max && tileCol > 0) {
            simdgroup_store(output[0], Cs + sr * 129 + sc, 129);
            simdgroup_store(output[1], Cs + sr * 129 + sc + 8, 129);
            simdgroup_store(output[2], Cs + sr * 129 + sc + 16, 129);
            simdgroup_store(output[3], Cs + sr * 129 + sc + 24, 129);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = lid; i < 16 * 128; i += 128)
                Cs[(i / 128) * 129 + (i % 128)] *= scale;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_load(output[0], Cs + sr * 129 + sc, 129);
            simdgroup_load(output[1], Cs + sr * 129 + sc + 8, 129);
            simdgroup_load(output[2], Cs + sr * 129 + sc + 16, 129);
            simdgroup_load(output[3], Cs + sr * 129 + sc + 24, 129);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // exp + sum in one pass (Bs still holds scaled scores)
        float tile_sum = 0.0f;
        for (uint i = lid; i < 16 * 128; i += 128) {
            uint idx = (i / 128) * 129 + (i % 128);
            float val = exp(Bs[idx] - row_max);
            Bs[idx] = val;
            tile_sum += val;
        }
        tile_sum = simd_sum(tile_sum);
        threadgroup float simd_sums[4];
        if (lid % 32 == 0) simd_sums[simd_id] = tile_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        row_sum += simd_sums[0] + simd_sums[1] + simd_sums[2] + simd_sums[3];

        // ── Step 3: scores × V ──
        // Bs holds softmax probs (16×128). V is (seq, head_dim) in device memory.
        // Each SIMD owns 32 output columns. Load V in 16-row strips, MMA against score strips.
        // No reload of scores needed — Bs is untouched, we read score strips directly.
        for (uint vk = 0; vk < 128; vk += 16) {
            for (uint i = lid; i < 16 * 32; i += 128) {
                uint r = i / 32, c = i % 32;
                uint gr = tileCol + vk + r;
                uint gc = sc + c;
                As[r * 17 + c] = (gr < seq && gc < head_dim) ? float(V[gr * head_dim + gc]) : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_float8x8 s0, v0, v1, v2, v3;
            simdgroup_load(s0, Bs + sr * 129 + vk, 129);
            simdgroup_load(v0, As, 17);
            simdgroup_load(v1, As + 8, 17);
            simdgroup_load(v2, As + 16, 17);
            simdgroup_load(v3, As + 24, 17);
            simdgroup_multiply_accumulate(output[0], s0, v0, output[0]);
            simdgroup_multiply_accumulate(output[1], s0, v1, output[1]);
            simdgroup_multiply_accumulate(output[2], s0, v2, output[2]);
            simdgroup_multiply_accumulate(output[3], s0, v3, output[3]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // ── Final: normalize + write ──
    float inv_sum = 1.0f / row_sum;
    simdgroup_store(output[0], Bs + sr * 129 + sc, 129);
    simdgroup_store(output[1], Bs + sr * 129 + sc + 8, 129);
    simdgroup_store(output[2], Bs + sr * 129 + sc + 16, 129);
    simdgroup_store(output[3], Bs + sr * 129 + sc + 24, 129);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < 16 * 128; i += 128)
        Bs[(i / 128) * 129 + (i % 128)] *= inv_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < 16 * head_dim; i += 128) {
        uint r = i / head_dim, c = i % head_dim;
        uint gr = tileRow + r;
        if (gr < seq)
            O[gr * head_dim + c] = half(Bs[r * 129 + c]);
    }
}



