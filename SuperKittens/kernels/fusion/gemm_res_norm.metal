//
//  gemm_res_norm.metal — fused GEMM + residual + RMSNorm
//
//  GEMM → writeback+residual → RMSNorm → output.
//  Element-addressed scratch buffer avoids half4 sub-element conflicts.
//

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

enum : uint { BM = 64, BN = 64, BK = 32, MR = 2, MC = 8, THREADS = 128 };

[[host_name("gemm_res_norm")]]
[[kernel, max_total_threads_per_threadgroup(THREADS)]]
void gemm_res_norm(
    device const half* A, device const half* B,
    device const half* residual, device const half* norm_weight,
    device half* C,
    constant uint& M, constant uint& N, constant uint& K, constant float& eps,
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    const uint row0 = gid.y * BM + simd * 16;
    const uint col0 = gid.x * BN;

    simdgroup_float8x8 acc[MR][MC] = {};
    threadgroup half As[BM * BK], Bs[BK * BN];

    const uint tid = simd * 32 + lane;
    const device half4* a4 = reinterpret_cast<const device half4*>(A);
    const device half4* b4 = reinterpret_cast<const device half4*>(B);

    // GEMM
    for (uint bk = 0; bk < K; bk += BK) {
        for (uint i = tid; i < (BM * BK) / 4; i += THREADS) {
            uint r = (i * 4) / BK, c = (i * 4) % BK;
            uint gr = row0 + r, gc = bk + c;
            reinterpret_cast<threadgroup half4*>(As)[i] =
                (gr < M && gc < K) ? a4[(gr * K + gc) / 4] : half4(0);
        }
        for (uint i = tid; i < (BK * BN) / 4; i += THREADS) {
            uint r = (i * 4) / BN, c = (i * 4) % BN;
            uint gr = bk + r, gc = col0 + c;
            reinterpret_cast<threadgroup half4*>(Bs)[i] =
                (gr < K && gc < N) ? b4[(gr * N + gc) / 4] : half4(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < BK / 8; k++)
            for (uint r = 0; r < MR; r++) {
                simdgroup_half8x8 a;
                simdgroup_load(a, As + (simd * 16 + r * 8) * BK + k * 8, BK);
                for (uint c = 0; c < MC; c++) {
                    simdgroup_half8x8 b;
                    simdgroup_load(b, Bs + k * 8 * BN + c * 8, BN);
                    simdgroup_multiply_accumulate(acc[r][c], a, b, acc[r][c]);
                }
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Residual add + write to scratch (element-addressed)
    threadgroup half scratch[BM * BN];
    const device half* res = reinterpret_cast<const device half*>(residual);

    for (uint r = 0; r < MR; r++)
        for (uint c = 0; c < MC; c++) {
            float2 v = reinterpret_cast<thread float2&>(acc[r][c].thread_elements());
            uint qid = lane / 4, lr = (qid & 4) + ((lane / 2) % 4), lc = (qid & 2) * 2 + (lane % 2) * 2;
            uint gr = row0 + r * 8 + lr;
            uint gc = col0 + c * 8 + lc;
            if (gr < M && gc < N) {
                scratch[(gr % BM) * BN + gc]     = half(v.x + float(res[gr * N + gc]));
                scratch[(gr % BM) * BN + gc + 1] = half(v.y + float(res[gr * N + gc + 1]));
            }
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // RMSNorm from scratch
    for (uint i = 0; i < BM; i++) {
        uint gr = row0 + i;
        if (gr >= M) break;

        float sq = 0;
        const threadgroup half4* sr4 = reinterpret_cast<const threadgroup half4*>(scratch + i * BN);
        for (uint c = lane; c < BN / 4; c += 32) {
            float4 v = float4(sr4[c]);
            sq += dot(v, v);
        }
        sq = simd_sum(sq);
        float inv_rms = metal::precise::rsqrt(sq / float(N) + eps);
        const device half4* nw4 = reinterpret_cast<const device half4*>(norm_weight);
        for (uint c = lane; c < BN / 4; c += 32) {
            float4 v = float4(sr4[c]) * inv_rms;
            reinterpret_cast<device half4*>(C + (size_t)gr * N)[c] = half4(v * float4(nw4[c]));
        }
    }
}
