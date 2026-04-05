//
//  fp16_2.metal
//  SuperKittens
//
//  Created by Alazar Manakelew on 4/2/26.
//

#include <metal_stdlib>
#include "types.h"

using namespace metal;


/////////
/// GEMM <64, 64, 16> — tiled with vectorized loads
/// 128 threads (4 simdgroups), 4 KB threadgroup memory + padding
/// Each simdgroup owns 32x32 output via 4x4 grid of 8x8 fragments
////////

template<uint BM, uint BN, uint BK>
kernel void fp16_2_gemm(
    device const half*  A       [[buffer(0)]],
    device const half*  B       [[buffer(1)]],
    device half*        C       [[buffer(2)]],
    constant uint&      M       [[buffer(3)]],
    constant uint&      N       [[buffer(4)]],
    constant uint&      K       [[buffer(5)]],
    uint2 tid                   [[thread_position_in_threadgroup]],
    uint2 tgid                  [[threadgroup_position_in_grid]],
    uint  simd_id               [[simdgroup_index_in_threadgroup]],
    uint  simd_lane             [[thread_index_in_simdgroup]]
) {
    constexpr uint WM = BM / 32;
    constexpr uint WN = BN / 32;
    constexpr uint THREADS = WM * WN * 32;
    constexpr uint PAD = 8; // padding to avoid bank conflicts

    static_assert(BM == 64, "BM needs to be 64");
    static_assert(BN == 64, "BN needs to be 64");
    static_assert(BK == 16, "BK needs to be 16");
    static_assert(BK % 4 == 0, "BK must be divisible by 4 for half4 loads");
    static_assert(BN % 4 == 0, "BN must be divisible by 4 for half4 loads");

    // Padded threadgroup memory to avoid bank conflicts
    constexpr uint LDA_S = BK + PAD;  // stride for As: 16 + 8 = 24
    constexpr uint LDB_S = BN + PAD;  // stride for Bs: 64 + 8 = 72

    threadgroup half As[BM * LDA_S]; // 64 x 24
    threadgroup half Bs[BK * LDB_S]; // 16 x 72

    uint c_row = tgid.y * BM;
    uint c_col = tgid.x * BN;
    uint wm_id = simd_id / WN;
    uint wn_id = simd_id % WN;
    uint thread_idx = tid.y * 16 + tid.x;

    // Each simdgroup owns 32x32 — 4x4 grid of 8x8 fragments
    simdgroup_matrix<half, 8, 8> acc[4][4] = {};
    simdgroup_matrix<half, 8, 8> a_frag[4], b_frag[4];

    for (uint k_tile = 0; k_tile < K; k_tile += BK) {

        // ── Vectorized load A tile: BM x BK = 64 x 16 = 1024 halfs ──
        // 1024 / 4 = 256 half4 loads, 256 / 128 threads = 2 per thread
        for (uint idx = thread_idx; idx < (BM * BK) / 4; idx += THREADS) {
            uint elem = idx * 4;
            uint r = elem / BK;
            uint c = elem % BK;
            *((threadgroup half4*)(&As[r * LDA_S + c])) =
                *((device const half4*)(&A[(c_row + r) * K + k_tile + c]));
        }

        // ── Vectorized load B tile: BK x BN = 16 x 64 = 1024 halfs ──
        for (uint idx = thread_idx; idx < (BK * BN) / 4; idx += THREADS) {
            uint elem = idx * 4;
            uint r = elem / BN;
            uint c = elem % BN;
            *((threadgroup half4*)(&Bs[r * LDB_S + c])) =
                *((device const half4*)(&B[(k_tile + r) * N + c_col + c]));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── MMA over BK=16 in chunks of 8 ──
        for (uint kk = 0; kk < BK; kk += 8) {
            for (uint fi = 0; fi < 4; fi++)
                simdgroup_load(a_frag[fi], &As[(wm_id * 32 + fi * 8) * LDA_S + kk], LDA_S);

            for (uint fj = 0; fj < 4; fj++)
                simdgroup_load(b_frag[fj], &Bs[kk * LDB_S + wn_id * 32 + fj * 8], LDB_S);

            for (uint fi = 0; fi < 4; fi++)
                for (uint fj = 0; fj < 4; fj++)
                    simdgroup_multiply_accumulate(acc[fi][fj], a_frag[fi], b_frag[fj], acc[fi][fj]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Store directly to C ──
    for (uint fi = 0; fi < 4; fi++) {
        for (uint fj = 0; fj < 4; fj++) {
            uint out_row = c_row + wm_id * 32 + fi * 8;
            uint out_col = c_col + wn_id * 32 + fj * 8;
            simdgroup_store(acc[fi][fj], C + out_row * N + out_col, N);
        }
    }
}

// Instantiation
template [[host_name("fp16_2_gemm")]]
kernel void fp16_2_gemm<64, 64, 16>(
    device const half*, device const half*, device half*,
    constant uint&, constant uint&, constant uint&,
    uint2, uint2, uint, uint);
