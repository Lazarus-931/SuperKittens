//
//  fp16_2.metal
//  SuperKittens
//
//  First fused FP16 GEMM path:
//  C = silu(alpha * (A @ B) + beta * C + bias)
//

#include "../../../meow.h"
#include "../gemm.h"
#include "../ops.h"
#include "../utils/loader.h"
#include "../common.h"

namespace meow {
namespace gemm {
namespace fp16 {

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int MMA_ROWS, int MMA_COLS,
          int EXPECT_M, int EXPECT_N, int EXPECT_K>
[[kernel]]
void gemm_fp16_nn_bias_silu_kernel(
    const device half* A [[buffer(0)]],
    const device half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant GemmParams& p [[buffer(3)]],
    const device half* bias [[buffer(4)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    if (p.op_a != GEMM_OP_N || p.op_b != GEMM_OP_N) return;
    if (p.epilogue != GEMM_EPILOGUE_BIAS_SILU) return;

    if (EXPECT_M > 0 && !matches_specialization(p, EXPECT_M, EXPECT_N, EXPECT_K)) return;

    threadgroup half As[BLOCK_M * BLOCK_K];
    threadgroup half Bs[BLOCK_K * BLOCK_N];
    threadgroup half Os[BLOCK_M * BLOCK_N];

    gemm_mma_nn<BLOCK_M, BLOCK_N, BLOCK_K, MMA_ROWS, MMA_COLS>(
        A, B, C, p, bias, group_id, lid, simd_id, As, Bs, Os);
}

template [[host_name("gemm_fp16_nn_bias_silu_generic")]]
[[kernel]]
void gemm_fp16_nn_bias_silu_kernel<32, 64, 32, 2, 4, 0, 0, 0>(
    const device half*,
    const device half*,
    device half*,
    constant GemmParams&,
    const device half*,
    uint2,
    uint,
    uint);

template [[host_name("gemm_fp16_nn_bias_silu_2048x3072x4096")]]
[[kernel]]
void gemm_fp16_nn_bias_silu_kernel<32, 64, 32, 2, 4, 2048, 3072, 4096>(
    const device half*,
    const device half*,
    device half*,
    constant GemmParams&,
    const device half*,
    uint2,
    uint,
    uint);

template [[host_name("gemm_fp16_nn_bias_silu_3072x2048x4096")]]
[[kernel]]
void gemm_fp16_nn_bias_silu_kernel<32, 64, 32, 2, 4, 3072, 2048, 4096>(
    const device half*,
    const device half*,
    device half*,
    constant GemmParams&,
    const device half*,
    uint2,
    uint,
    uint);

template [[host_name("gemm_fp16_nn_bias_silu_4096x4096x4096")]]
[[kernel]]
void gemm_fp16_nn_bias_silu_kernel<32, 64, 32, 2, 4, 4096, 4096, 4096>(
    const device half*,
    const device half*,
    device half*,
    constant GemmParams&,
    const device half*,
    uint2,
    uint,
    uint);

} // namespace fp16
} // namespace gemm
} // namespace meow
