//
//  gemm.metal
//  SuperKittens — FP8 (E4M3) GEMM (NN, NT, TN)
//
//  A, B stored as uchar. Converted to half on load. Same architecture as fp16.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

enum : uint { BM = 32, BN = 64, BK = 32, MR = 2, MC = 8 };

// ── E4M3 conversion ──

inline float fp8_to_f32(uchar v) {
    uint e = (v >> 3) & 0xF, m = v & 0x7;
    if (e == 0) return (m == 0) ? 0.0f : ldexp(float(m) / 8.0f, -6);
    uint bits = ((uint(v) >> 7) << 31) | ((e - 7 + 127) << 23) | (m << 20);
    return as_type<float>(bits);
}

inline uchar f32_to_fp8(float x) {
    uint bits = as_type<uint>(x);
    uint s = (bits >> 31) & 1;
    int  e = int((bits >> 23) & 0xFF) - 127 + 7;
    uint m = (bits >> 20) & 0x7;
    uint round = (bits >> 19) & 1;
    bool sticky = (bits & 0x7FFFF) != 0;
    if (round && (sticky || (m & 1))) m++;
    if (m > 7) { m = 0; e++; }
    if (e <= 0) return uchar(s << 7);
    if (e >= 15) return uchar((s << 7) | 0x7B);
    return uchar((s << 7) | (e << 3) | m);
}

// ── Kernel ──

[[host_name("gemm_fp8")]]
[[kernel]]
void gemm_fp8(
    device const uchar* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device uchar* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    constant uint& ldA    [[buffer(6)]],
    constant uint& ldB    [[buffer(7)]],
    constant uint& ldC    [[buffer(8)]],
    constant bool& transA    [[buffer(9)]],
    constant bool& transB    [[buffer(10)]],
    constant bool& has_bias  [[buffer(11)]],
    device const half* bias  [[buffer(12)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const uint br = gid.y * BM, bc = gid.x * BN;
    const uint r0 = simd * MR * 8;

    threadgroup half As[BM * BK];
    threadgroup half Bs[BK * BN];
    simdgroup_float8x8 acc[MR][MC] = {};

    const uint n_a = BM * BK, n_b = BK * BN;
    const uint n_max = max(n_a, n_b);

    for (uint k0 = 0; k0 < K; k0 += BK) {
        for (uint i = simd * 32 + lane; i < n_max; i += 64) {
            if (i < n_a) {
                uint r = i / BK, c = i % BK;
                uint gr = transA ? (k0 + c) : (br + r);
                uint gc = transA ? (br + r) : (k0 + c);
                As[i] = (gr < (transA ? K : M) && gc < (transA ? M : K))
                    ? half(fp8_to_f32(A[gr * ldA + gc])) : half(0);
            }
            if (i < n_b) {
                uint r = i / BN, c = i % BN;
                uint gr = transB ? (bc + c) : (k0 + r);
                uint gc = transB ? (k0 + r) : (bc + c);
                Bs[i] = (gr < (transB ? N : K) && gc < (transB ? K : N))
                    ? half(fp8_to_f32(B[gr * ldB + gc])) : half(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

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

    // Store
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

    for (uint i = simd * 32 + lane; i < BM * BN; i += 64) {
        uint r = i / BN, c = i % BN;
        uint gr = br + r, gc = bc + c;
        if (gr < M && gc < N) {
            float val = float(Cs[i]);
            if (has_bias) val += float(bias[gc]);
            C[gr * ldC + gc] = f32_to_fp8(val);
        }
    }
}
