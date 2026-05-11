//
//  router_v2.metal — MoE router with simdgroup-parallel dot product.
//
//  v1: 128 threads, each thread fully computes one expert's logit (serial dot
//      over D). Wastes lane parallelism on the dot.
//
//  v2: One simdgroup (32 lanes) per expert. Lanes cooperatively dot via half4
//      strided loads + simd_sum. N_SG simdgroups per TG cover N_SG experts at a
//      time; loop if N_expert > N_SG. Softmax + top-K stay sequential on lane 0
//      (N is small).
//

#include <metal_stdlib>
using namespace metal;

constant constexpr uint MAX_EXPERTS = 256;
constant constexpr uint N_SG        = 8;
constant constexpr uint TGSZ        = 32 * N_SG;   // 256

[[host_name("moe_router")]]
[[kernel, max_total_threads_per_threadgroup(TGSZ)]]
void moe_router_v2(
    device const half*  x          [[buffer(0)]],
    device const half*  W          [[buffer(1)]],
    device int*         top_idx    [[buffer(2)]],
    device half*        top_score  [[buffer(3)]],
    constant uint&      T          [[buffer(4)]],
    constant uint&      D          [[buffer(5)]],
    constant uint&      N          [[buffer(6)]],
    constant uint&      K          [[buffer(7)]],
    uint    tgid       [[threadgroup_position_in_grid]],
    uint    tid        [[thread_index_in_threadgroup]],
    ushort  sg         [[simdgroup_index_in_threadgroup]],
    ushort  lane       [[thread_index_in_simdgroup]])
{
    const uint row = tgid;
    if (row >= T) return;

    threadgroup float logits[MAX_EXPERTS];
    threadgroup float sm_max;
    threadgroup float sm_sum;

    const device half* xrow = x + (size_t)row * D;

    // Each simdgroup handles one expert per pass.
    // W is (D, N) row-major. Logit for expert e = sum_d x[d] * W[d, e].
    // Lanes stride d by 32; expert e fixed per SG. Coalescing is weak because
    // adjacent lanes touch W[d, e], W[d+1, e] (stride N apart). But at small N
    // (<=128) those land in same cacheline group; and x is broadcast.
    for (uint e0 = sg; e0 < N; e0 += N_SG) {
        float acc = 0.0f;
        for (uint d = lane; d < D; d += 32) {
            acc += float(xrow[d]) * float(W[(size_t)d * N + e0]);
        }
        acc = simd_sum(acc);
        if (lane == 0) logits[e0] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float m = -INFINITY;
        for (uint e = 0; e < N; ++e) m = max(m, logits[e]);
        sm_max = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = tid; e < N; e += TGSZ) {
        logits[e] = exp(logits[e] - sm_max);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float s = 0.0f;
        for (uint e = 0; e < N; ++e) s += logits[e];
        sm_sum = s > 0.0f ? s : 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = tid; e < N; e += TGSZ) {
        logits[e] = logits[e] / sm_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        for (uint k = 0; k < K; ++k) {
            float best = -1.0f;
            int   bi   = 0;
            for (uint e = 0; e < N; ++e) {
                if (logits[e] > best) { best = logits[e]; bi = (int)e; }
            }
            top_idx[(size_t)row * K + k]   = bi;
            top_score[(size_t)row * K + k] = half(best);
            logits[bi] = -1.0f;
        }
    }
}
