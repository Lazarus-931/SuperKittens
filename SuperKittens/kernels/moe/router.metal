//  router_v2.metal — MoE router with simdgroup-parallel dot product.

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

// ── Occupancy expert-split router (decode T=1) ──────────────────────────────
// The single-TG moe_router above pins all 64 experts × D=2048 onto one GPU
// core for T=1 (grid = MTL::Size(1,1,1)), running at ~2 GB/s — sub-occupancy.
// Spread the experts across N_EGRP threadgroups (grid.y): each TG owns a
// contiguous expert-block and computes those logits with the *identical*
// full-D loop + single simd_sum as moe_router (same accumulation order →
// bit-for-bit identical logits, so routing/top-K is unchanged). A tiny reduce
// kernel then gathers the per-expert logits and does the softmax+top-K. The
// per-SG simd_sum over the full D is preserved (NOT collapsed to a serial
// per-thread loop — that was the confirmed-NEG coalescing approach), and the
// reduce does NO partial-sum (no fp reorder), unlike a split-D scheme.
constant constexpr uint N_EGRP = 8;

[[host_name("moe_router_partial")]]
[[kernel, max_total_threads_per_threadgroup(TGSZ)]]
void moe_router_partial(
    device const half*  x          [[buffer(0)]],
    device const half*  W          [[buffer(1)]],
    device float*       logits_g   [[buffer(2)]],   // [T][N] full per-expert logits
    constant uint&      T          [[buffer(4)]],
    constant uint&      D          [[buffer(5)]],
    constant uint&      N          [[buffer(6)]],
    uint2   tgid       [[threadgroup_position_in_grid]],
    ushort  sg         [[simdgroup_index_in_threadgroup]],
    ushort  lane       [[thread_index_in_simdgroup]])
{
    const uint row = tgid.x;
    const uint g   = tgid.y;        // expert-group index in [0, N_EGRP)
    if (row >= T) return;

    // Contiguous expert-block [e_lo, e_hi) for this threadgroup.
    const uint per   = (N + N_EGRP - 1u) / N_EGRP;
    const uint e_lo  = g * per;
    const uint e_hi  = min(e_lo + per, N);

    const device half* xrow = x + (size_t)row * D;

    // IDENTICAL dot-product to moe_router: full D, lane-stride-32, single
    // simd_sum. Same term order → bit-for-bit identical logit per expert.
    for (uint e0 = e_lo + sg; e0 < e_hi; e0 += N_SG) {
        float acc = 0.0f;
        for (uint d = lane; d < D; d += 32) {
            acc += float(xrow[d]) * float(W[(size_t)d * N + e0]);
        }
        acc = simd_sum(acc);
        if (lane == 0) logits_g[(size_t)row * N + e0] = acc;
    }
}

[[host_name("moe_router_reduce")]]
[[kernel, max_total_threads_per_threadgroup(TGSZ)]]
void moe_router_reduce(
    device const float* logits_g   [[buffer(0)]],   // [T][N] full per-expert logits
    device int*         top_idx    [[buffer(2)]],
    device half*        top_score  [[buffer(3)]],
    constant uint&      T          [[buffer(4)]],
    constant uint&      N          [[buffer(6)]],
    constant uint&      K          [[buffer(7)]],
    uint    tgid       [[threadgroup_position_in_grid]],
    uint    tid        [[thread_index_in_threadgroup]])
{
    const uint row = tgid;
    if (row >= T) return;

    threadgroup float logits[MAX_EXPERTS];
    threadgroup float sm_max;
    threadgroup float sm_sum;

    const device float* lrow = logits_g + (size_t)row * N;
    for (uint e = tid; e < N; e += TGSZ) {
        logits[e] = lrow[e];        // gather only — no partial-sum reorder
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── identical softmax + top-K to moe_router (routing bit-for-bit) ──
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
        float ssum = 0.0f;
        for (uint e = 0; e < N; ++e) ssum += logits[e];
        sm_sum = ssum > 0.0f ? ssum : 1.0f;
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
