//  kv_up_pair_v2.metal — Tile-MMA fused MLA K-up / V-up (fp16).

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// 128 threads = 4 simdgroups. MR=1 row of 8 per simdgroup × 4 = BM=32.
enum : uint { BM = 32, BN = 64, BK = 32, MR = 1, MC = 8, NSG = 4, TGSZ = 128 };

[[host_name("kv_up_pair")]]
[[kernel, max_total_threads_per_threadgroup(TGSZ)]]
void kv_up_pair(
    device const half*  c_kv      [[buffer(0)]],
    device const half*  w_k_up    [[buffer(1)]],
    device const half*  w_v_up    [[buffer(2)]],
    device half*        k_no_pe   [[buffer(3)]],
    device half*        v_out     [[buffer(4)]],
    constant uint&      T         [[buffer(5)]],
    constant uint&      R         [[buffer(6)]],
    constant uint&      K_OUT     [[buffer(7)]],
    constant uint&      V_OUT     [[buffer(8)]],
    uint2  gid  [[threadgroup_position_in_grid]],
    uint   simd [[simdgroup_index_in_threadgroup]],
    uint   lane [[thread_index_in_simdgroup]])
{
    const uint br = gid.y * BM;
    const uint bc = gid.x * BN;
    if (br >= T) return;

    const bool do_k = (bc < K_OUT);
    const bool do_v = (bc < V_OUT);
    if (!do_k && !do_v) return;

    const uint r0 = simd * 8;  // each of 4 simdgroups owns 8 rows

    threadgroup half As[BM * BK];        // 2 KB
    threadgroup half Bs[BK * BN];        // 4 KB (reused across K and V)

    simdgroup_float8x8 acc_k[MR][MC] = {};
    simdgroup_float8x8 acc_v[MR][MC] = {};

    const uint n_a = BM * BK;
    const uint n_b = BK * BN;
    const uint tid = simd * 32 + lane;

    for (uint k0 = 0; k0 < R; k0 += BK) {
        // Load A = c_kv[br:br+BM, k0:k0+BK] (used for BOTH K and V MMA).
        for (uint i = tid; i < n_a / 4; i += TGSZ) {
            uint r = (i * 4) / BK, c = (i * 4) % BK;
            uint gr = br + r, gc = k0 + c;
            reinterpret_cast<threadgroup half4*>(As)[i] =
                (gr < T && gc < R)
                    ? reinterpret_cast<const device half4*>(c_kv + gr * R + gc)[0]
                    : half4(0);
        }

        // ── Pass 1: K MMA ──
        if (do_k) {
            for (uint i = tid; i < n_b / 4; i += TGSZ) {
                uint r = (i * 4) / BN, c = (i * 4) % BN;
                uint gr = k0 + r, gc = bc + c;
                reinterpret_cast<threadgroup half4*>(Bs)[i] =
                    (gr < R && gc < K_OUT)
                        ? reinterpret_cast<const device half4*>(w_k_up + gr * K_OUT + gc)[0]
                        : half4(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = 0; k < BK / 8; k++) {
                for (uint r = 0; r < MR; r++) {
                    simdgroup_half8x8 a;
                    simdgroup_load(a, As + (r0 + r * 8) * BK + k * 8, BK);
                    for (uint c = 0; c < MC; c++) {
                        simdgroup_half8x8 b;
                        simdgroup_load(b, Bs + k * 8 * BN + c * 8, BN);
                        simdgroup_multiply_accumulate(acc_k[r][c], a, b, acc_k[r][c]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ── Pass 2: V MMA (reuses As, overwrites Bs) ──
        if (do_v) {
            for (uint i = tid; i < n_b / 4; i += TGSZ) {
                uint r = (i * 4) / BN, c = (i * 4) % BN;
                uint gr = k0 + r, gc = bc + c;
                reinterpret_cast<threadgroup half4*>(Bs)[i] =
                    (gr < R && gc < V_OUT)
                        ? reinterpret_cast<const device half4*>(w_v_up + gr * V_OUT + gc)[0]
                        : half4(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = 0; k < BK / 8; k++) {
                for (uint r = 0; r < MR; r++) {
                    simdgroup_half8x8 a;
                    simdgroup_load(a, As + (r0 + r * 8) * BK + k * 8, BK);
                    for (uint c = 0; c < MC; c++) {
                        simdgroup_half8x8 b;
                        simdgroup_load(b, Bs + k * 8 * BN + c * 8, BN);
                        simdgroup_multiply_accumulate(acc_v[r][c], a, b, acc_v[r][c]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Store using Bs scratch (4KB), one output at a time.
    if (do_k) {
        for (uint r = 0; r < MR; r++) {
            for (uint c = 0; c < MC; c++) {
                float2 v2 = reinterpret_cast<thread float2&>(acc_k[r][c].thread_elements());
                uint qid = lane / 4;
                uint lr = (qid & 4) + ((lane / 2) % 4);
                uint lc = (qid & 2) * 2 + (lane % 2) * 2;
                Bs[(r0 + r * 8 + lr) * BN + c * 8 + lc]     = half(v2.x);
                Bs[(r0 + r * 8 + lr) * BN + c * 8 + lc + 1] = half(v2.y);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < (BM * BN) / 4; i += TGSZ) {
            uint r = (i * 4) / BN, c = (i * 4) % BN;
            uint gr = br + r, gc = bc + c;
            if (gr < T && gc < K_OUT) {
                reinterpret_cast<device half4*>(k_no_pe + gr * K_OUT + gc)[0] =
                    reinterpret_cast<threadgroup half4*>(Bs)[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (do_v) {
        for (uint r = 0; r < MR; r++) {
            for (uint c = 0; c < MC; c++) {
                float2 v2 = reinterpret_cast<thread float2&>(acc_v[r][c].thread_elements());
                uint qid = lane / 4;
                uint lr = (qid & 4) + ((lane / 2) % 4);
                uint lc = (qid & 2) * 2 + (lane % 2) * 2;
                Bs[(r0 + r * 8 + lr) * BN + c * 8 + lc]     = half(v2.x);
                Bs[(r0 + r * 8 + lr) * BN + c * 8 + lc + 1] = half(v2.y);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < (BM * BN) / 4; i += TGSZ) {
            uint r = (i * 4) / BN, c = (i * 4) % BN;
            uint gr = br + r, gc = bc + c;
            if (gr < T && gc < V_OUT) {
                reinterpret_cast<device half4*>(v_out + gr * V_OUT + gc)[0] =
                    reinterpret_cast<threadgroup half4*>(Bs)[i];
            }
        }
    }
}
