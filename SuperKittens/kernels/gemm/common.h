//
//  common.h
//  SuperKittens
//

#ifndef SUPERKITTENS_GEMM_COMMON_METAL
#define SUPERKITTENS_GEMM_COMMON_METAL

#include <metal_stdlib>
using namespace metal;

namespace meow {
namespace gemm {

METAL_FUNC bool matches_specialization(constant GemmParams& p, int m, int n, int k) {
    return p.m == m && p.n == n && p.k == k;
}

METAL_FUNC void copy_half4(
    threadgroup half* dst,
    uint dst_offset,
    const device half* src,
    uint src_offset)
{
    reinterpret_cast<threadgroup packed_half4*>(dst + dst_offset)[0] =
        reinterpret_cast<const device packed_half4*>(src + src_offset)[0];
}

METAL_FUNC void copy_half4(
    device half* dst,
    uint dst_offset,
    const threadgroup half* src,
    uint src_offset)
{
    reinterpret_cast<device packed_half4*>(dst + dst_offset)[0] =
        reinterpret_cast<const threadgroup packed_half4*>(src + src_offset)[0];
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int MMA_ROWS, int MMA_COLS>
METAL_FUNC void gemm_mma_nn(
    const device half* A,
    const device half* B,
    device half* C,
    constant GemmParams& p,
    const device half* bias,
    uint2 group_id,
    uint lid,
    uint simd_id,
    threadgroup half* As,
    threadgroup half* Bs,
    threadgroup half* Os)
{
    constexpr uint THREADS_PER_GROUP =
        (BLOCK_M / (MMA_ROWS * 8)) * (BLOCK_N / (MMA_COLS * 8)) * 32;
    const uint block_row = group_id.y * BLOCK_M;
    const uint block_col = group_id.x * BLOCK_N;
    const uint col_tiles = BLOCK_N / (MMA_COLS * 8);
    const uint row_tile = simd_id / col_tiles;
    const uint col_tile = simd_id % col_tiles;
    const uint row_base = row_tile * MMA_ROWS * 8;
    const uint col_base = col_tile * MMA_COLS * 8;
    const bool full_c_tile =
        block_row + BLOCK_M <= static_cast<uint>(p.m) &&
        block_col + BLOCK_N <= static_cast<uint>(p.n);
    const bool plain_copy_fast_path =
        bias == nullptr &&
        p.epilogue == GEMM_EPILOGUE_NONE &&
        p.alpha == 1.0f &&
        p.beta == 0.0f &&
        full_c_tile;

    meow::mma::Tile<MMA_ROWS, MMA_COLS> acc;
    acc.clear();

    for (int k0 = 0; k0 < p.k; k0 += BLOCK_K) {
        const uint k0u = static_cast<uint>(k0);
        const bool full_a_tile =
            block_row + BLOCK_M <= static_cast<uint>(p.m) &&
            k0u + BLOCK_K <= static_cast<uint>(p.k);
        if (full_a_tile) {
            for (uint idx4 = lid; idx4 < (BLOCK_M * BLOCK_K) / 4; idx4 += THREADS_PER_GROUP) {
                const uint idx = idx4 * 4;
                const uint r = idx / BLOCK_K;
                const uint c = idx % BLOCK_K;
                copy_half4(As, idx, A, static_cast<uint>((block_row + r) * p.lda) + (k0u + c));
            }
        } else {
            for (uint idx = lid; idx < BLOCK_M * BLOCK_K; idx += THREADS_PER_GROUP) {
                const uint r = idx / BLOCK_K;
                const uint c = idx % BLOCK_K;
                const uint gr = block_row + r;
                const uint gc = k0u + c;
                As[idx] = (gr < static_cast<uint>(p.m) && gc < static_cast<uint>(p.k))
                    ? load_matrix_element(A, gr, gc, p.lda, p.op_a)
                    : half(0.0f);
            }
        }

        const bool full_b_tile =
            k0u + BLOCK_K <= static_cast<uint>(p.k) &&
            block_col + BLOCK_N <= static_cast<uint>(p.n);
        if (full_b_tile) {
            for (uint idx4 = lid; idx4 < (BLOCK_K * BLOCK_N) / 4; idx4 += THREADS_PER_GROUP) {
                const uint idx = idx4 * 4;
                const uint r = idx / BLOCK_N;
                const uint c = idx % BLOCK_N;
                copy_half4(Bs, idx, B, static_cast<uint>((k0u + r) * p.ldb) + (block_col + c));
            }
        } else {
            for (uint idx = lid; idx < BLOCK_K * BLOCK_N; idx += THREADS_PER_GROUP) {
                const uint r = idx / BLOCK_N;
                const uint c = idx % BLOCK_N;
                const uint gr = k0u + r;
                const uint gc = block_col + c;
                Bs[idx] = (gr < static_cast<uint>(p.k) && gc < static_cast<uint>(p.n))
                    ? load_matrix_element(B, gr, gc, p.ldb, p.op_b)
                    : half(0.0f);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        meow::mma::mm_AB<BLOCK_K, MMA_ROWS, MMA_COLS>(
            acc,
            As + row_base * BLOCK_K, BLOCK_K,
            Bs + col_base, BLOCK_N);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    acc.store(Os + row_base * BLOCK_N + col_base, BLOCK_N);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (plain_copy_fast_path) {
        for (uint idx4 = lid; idx4 < (BLOCK_M * BLOCK_N) / 4; idx4 += THREADS_PER_GROUP) {
            const uint idx = idx4 * 4;
            const uint r = idx / BLOCK_N;
            const uint c = idx % BLOCK_N;
            copy_half4(C, static_cast<uint>((block_row + r) * p.ldc) + block_col + c, Os, idx);
        }
        return;
    }

    for (uint idx = lid; idx < BLOCK_M * BLOCK_N; idx += THREADS_PER_GROUP) {
        const uint r = idx / BLOCK_N;
        const uint c = idx % BLOCK_N;
        const uint row = block_row + r;
        const uint col = block_col + c;
        if (row < static_cast<uint>(p.m) && col < static_cast<uint>(p.n)) {
            float out = apply_alpha_beta(float(Os[idx]),
                                         float(load_output_element(C, row, col, p.ldc)),
                                         p.alpha, p.beta);
            if (bias != nullptr) out += load_bias_element(bias, col);
            out = apply_epilogue(out, p.epilogue);
            store_output_element(C, row, col, p.ldc, half(out));
        }
    }
}

} // namespace gemm
} // namespace meow

#endif // SUPERKITTENS_GEMM_COMMON_METAL
