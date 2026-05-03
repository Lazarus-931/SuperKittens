//
//  gemm.metal
//  SuperKittens — FP16 GEMM (NN, NT, TN)
//
//  64 threads, simdgroup MMA, single barrier/K-iter. Tile: 32×64×32.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

enum : uint { BM = 32, BN = 64, BK = 32, MR = 2, MC = 8 };

[[host_name("gemm_fp16")]]
[[kernel]]
void gemm_fp16(
    device const half* A     [[buffer(0)]],
    device const half* B     [[buffer(1)]],
    device half* C           [[buffer(2)]],
    constant uint& M         [[buffer(3)]],
    constant uint& N         [[buffer(4)]],
    constant uint& K         [[buffer(5)]],
    constant uint& ldA       [[buffer(6)]],
    constant uint& ldB       [[buffer(7)]],
    constant uint& ldC       [[buffer(8)]],
    constant bool& transA    [[buffer(9)]],
    constant bool& transB    [[buffer(10)]],
    constant bool& has_bias [[buffer(11)]],
    device const half* bias [[buffer(12)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const uint br = gid.y * BM, bc = gid.x * BN;
    const uint r0 = simd * MR * 8;

    threadgroup half As[BM * BK];  // 2KB, always [BM][BK] layout
    threadgroup half Bs[BK * BN];  // 4KB, always [BK][BN] layout
    simdgroup_float8x8 acc[MR][MC] = {};

    const uint n_a = BM * BK, n_b = BK * BN;

    for (uint k0 = 0; k0 < K; k0 += BK) {
        // Cooperative load A and B tiles.
        // Use half4 vectorized loads when the matrix is in standard layout (contiguous);
        // fall back to scalar when transposed (strided source access).
        if (!transA) {
            // ── A[br:br+BM][k0:k0+BK]: contiguous in K, half4 load ──
            for (uint i = simd * 32 + lane; i < n_a / 4; i += 64) {
                uint r = (i * 4) / BK, c = (i * 4) % BK;
                uint gr = br + r, gc = k0 + c;
                reinterpret_cast<threadgroup half4*>(As)[i] =
                    (gr < M && gc < K) ? reinterpret_cast<const device half4*>(A + gr * ldA + gc)[0] : half4(0);
            }
        } else {
            // ── A^T: scalar, strided source ──
            for (uint i = simd * 32 + lane; i < n_a; i += 64) {
                uint r = i / BK, c = i % BK;
                uint gr = k0 + c, gc = br + r;
                As[i] = (gr < K && gc < M) ? A[gr * ldA + gc] : half(0);
            }
        }

        if (!transB) {
            // ── B[k0:k0+BK][bc:bc+BN]: contiguous in N, half4 load ──
            for (uint i = simd * 32 + lane; i < n_b / 4; i += 64) {
                uint r = (i * 4) / BN, c = (i * 4) % BN;
                uint gr = k0 + r, gc = bc + c;
                reinterpret_cast<threadgroup half4*>(Bs)[i] =
                    (gr < K && gc < N) ? reinterpret_cast<const device half4*>(B + gr * ldB + gc)[0] : half4(0);
            }
        } else {
            // ── B^T: scalar, strided source ──
            for (uint i = simd * 32 + lane; i < n_b; i += 64) {
                uint r = i / BN, c = i % BN;
                uint gr = bc + c, gc = k0 + r;
                Bs[i] = (gr < N && gc < K) ? B[gr * ldB + gc] : half(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MMA: A[BM][BK] @ B[BK][BN] → acc[MR][MC]
        for (uint k = 0; k < BK / 8; k++) {
            for (uint r = 0; r < MR; r++) {
                simdgroup_half8x8 a;
                simdgroup_load(a, As + (r0 + r * 8) * BK + k * 8, BK);
                for (uint c = 0; c < MC; c++) {
                    simdgroup_half8x8 b;
                    simdgroup_load(b, Bs + k * 8 * BN + c * 8, BN);
                    simdgroup_multiply_accumulate(acc[r][c], a, b, acc[r][c]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store: accumulator → threadgroup → device
    threadgroup half Cs[BM * BN];
    for (uint r = 0; r < MR; r++) {
        for (uint c = 0; c < MC; c++) {
            float2 v = reinterpret_cast<thread float2&>(acc[r][c].thread_elements());
            uint qid = lane / 4;
            uint lr = (qid & 4) + ((lane / 2) % 4);
            uint lc = (qid & 2) * 2 + (lane % 2) * 2;
            Cs[(r0 + r * 8 + lr) * BN + c * 8 + lc]     = half(v.x);
            Cs[(r0 + r * 8 + lr) * BN + c * 8 + lc + 1] = half(v.y);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = simd * 32 + lane; i < (BM * BN) / 4; i += 64) {
        uint r = (i * 4) / BN, c = (i * 4) % BN;
        uint gr = br + r, gc = bc + c;
        if (gr < M && gc < N) {
            half4 val = reinterpret_cast<threadgroup half4*>(Cs)[i];
            if (has_bias) {
                half4 b4 = reinterpret_cast<const device half4*>(bias + gc)[0];
                val = half4(float4(val) + float4(b4));
            }
            reinterpret_cast<device half4*>(C + gr * ldC + gc)[0] = val;
        }
    }
}
