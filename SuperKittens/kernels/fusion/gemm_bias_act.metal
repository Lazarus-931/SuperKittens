//
//  gemm_bias_act.metal — fused GEMM + bias + activation
//
//  Standard simdgroup GEMM with epilogue: bias + GELU/SiLU/ReLU
//  Act=0: identity, 1: GELU, 2: SiLU, 3: ReLU
//

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

enum : uint {
    BM = 64, BN = 64, BK = 32,
    MR = 2, MC = 8, THREADS = 128
};

[[host_name("gemm_bias_act")]]
[[kernel, max_total_threads_per_threadgroup(THREADS)]]
void gemm_bias_act(
    device const half* A,
    device const half* B,
    device const half* bias,
    device half* C,
    constant uint& M, constant uint& N, constant uint& K,
    constant uint& act,  // 0=none, 1=gelu, 2=silu, 3=relu
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    const uint row = gid.y * BM + simd * 16;
    const uint col = gid.x * BN;

    simdgroup_float8x8 acc[MR][MC] = {};
    threadgroup half As[BM * BK];
    threadgroup half Bs[BK * BN];

    const uint tid = simd * 32 + lane;
    const device half4* a4 = reinterpret_cast<const device half4*>(A);
    const device half4* b4 = reinterpret_cast<const device half4*>(B);

    for (uint bk = 0; bk < K; bk += BK) {
        // Load A tile
        for (uint i = tid; i < (BM * BK) / 4; i += THREADS) {
            uint r = (i * 4) / BK, c = (i * 4) % BK;
            uint gr = row + r, gc = bk + c;
            reinterpret_cast<threadgroup half4*>(As)[i] =
                (gr < M && gc < K) ? a4[(gr * K + gc) / 4] : half4(0);
        }
        // Load B tile
        for (uint i = tid; i < (BK * BN) / 4; i += THREADS) {
            uint r = (i * 4) / BN, c = (i * 4) % BN;
            uint gr = bk + r, gc = col + c;
            reinterpret_cast<threadgroup half4*>(Bs)[i] =
                (gr < K && gc < N) ? b4[(gr * N + gc) / 4] : half4(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < BK / 8; k++) {
            for (uint r = 0; r < MR; r++) {
                simdgroup_half8x8 a;
                simdgroup_load(a, As + (simd * 16 + r * 8) * BK + k * 8, BK);
                for (uint c = 0; c < MC; c++) {
                    simdgroup_half8x8 b;
                    simdgroup_load(b, Bs + k * 8 * BN + c * 8, BN);
                    simdgroup_multiply_accumulate(acc[r][c], a, b, acc[r][c]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Epilogue: bias + activation fused into writeback
    const device half* bp = reinterpret_cast<const device half*>(bias);
    for (uint r = 0; r < MR; r++) {
        for (uint c = 0; c < MC; c++) {
            float2 v = reinterpret_cast<thread float2&>(acc[r][c].thread_elements());
            uint qid = lane / 4;
            uint lr = (qid & 4) + ((lane / 2) % 4);
            uint lc = (qid & 2) * 2 + (lane % 2) * 2;
            uint gr = row + r * 8 + lr;
            uint gc = col + c * 8 + lc;

            if (gr < M && gc < N) {
                float v0 = v.x + (bias ? float(bp[gc]) : 0.0f);
                float v1 = v.y + (bias ? float(bp[gc + 1]) : 0.0f);

                if (act == 1) {  // GELU
                    float a0 = 0.044715f * v0 * v0 * v0;
                    float a1 = 0.044715f * v1 * v1 * v1;
                    v0 = 0.5f * v0 * (1.0f + metal::fast::tanh(0.79788456f * (v0 + a0)));
                    v1 = 0.5f * v1 * (1.0f + metal::fast::tanh(0.79788456f * (v1 + a1)));
                } else if (act == 2) {  // SiLU
                    v0 = v0 / (1.0f + metal::fast::exp(-v0));
                    v1 = v1 / (1.0f + metal::fast::exp(-v1));
                } else if (act == 3) {  // ReLU
                    v0 = fmax(v0, 0.0f);
                    v1 = fmax(v1, 0.0f);
                }

                size_t off = (size_t)gr * N + gc;
                (reinterpret_cast<device half*>(C))[off]     = half(v0);
                (reinterpret_cast<device half*>(C))[off + 1] = half(v1);
            }
        }
    }
}
