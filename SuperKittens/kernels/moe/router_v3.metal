// V3/V4 MoE router:
//   logits  = sigmoid(x @ W.T)                            (modeling_deepseek_v3.py:150)
//   scores  = logits + e_score_correction_bias            (line 217)
//   reshape (n_group, experts_per_group); per-group score = sum of top-2 within group (line 222)
//   keep topk_group groups; mask others to -inf            (line 224)
//   top_k over allowed; gather weights from raw sigmoid    (line 228)
//   if norm_topk_prob: weights /= sum(+1e-20)              (line 232)
//   weights *= routed_scaling_factor                       (line 236)
//
// V2-Lite: n_group=0 -> skip group restriction; norm_topk_prob=False; scaling=1.0; no bias.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint MAX_EXPERTS = 256;
constant constexpr uint MAX_GROUPS  = 16;
constant constexpr uint N_SG        = 8;
constant constexpr uint TGSZ        = 32 * N_SG;

struct RouterV3Args {
    uint  T;
    uint  D;
    uint  N;                  // total experts
    uint  K;                  // top_k
    uint  n_group;            // 0 disables grouping (V2-Lite)
    uint  topk_group;
    float routed_scaling;
    uint  norm_topk_prob;     // 0/1
    uint  has_bias;           // 0/1
};

[[host_name("moe_router_v3")]]
[[kernel, max_total_threads_per_threadgroup(TGSZ)]]
void moe_router_v3(
    device const half*   x         [[buffer(0)]],
    device const half*   W         [[buffer(1)]],
    device const float*  bias      [[buffer(2)]],   // e_score_correction_bias, fp32, len N
    device int*          top_idx   [[buffer(3)]],
    device half*         top_score [[buffer(4)]],
    constant RouterV3Args& A       [[buffer(5)]],
    uint   tgid [[threadgroup_position_in_grid]],
    uint   tid  [[thread_index_in_threadgroup]],
    ushort sg   [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    const uint row = tgid;
    if (row >= A.T) return;

    threadgroup float logits[MAX_EXPERTS];   // raw sigmoid (gather source)
    threadgroup float scores[MAX_EXPERTS];   // sigmoid + bias (selection source)
    threadgroup float gscore[MAX_GROUPS];
    threadgroup float gthresh;               // topk_group-th largest group score
    threadgroup float wsum;

    const device half* xrow = x + (size_t)row * A.D;

    // 1) logits = sigmoid(x @ W.T)
    for (uint e0 = sg; e0 < A.N; e0 += N_SG) {
        float acc = 0.0f;
        for (uint d = lane; d < A.D; d += 32) {
            acc += float(xrow[d]) * float(W[(size_t)d * A.N + e0]);
        }
        acc = simd_sum(acc);
        if (lane == 0) {
            float p = 1.0f / (1.0f + exp(-acc));
            logits[e0] = p;
            scores[e0] = p + (A.has_bias ? bias[e0] : 0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2) Group restriction (V3/V4 only).
    if (A.n_group > 0 && tid == 0) {
        const uint per = A.N / A.n_group;
        for (uint g = 0; g < A.n_group; ++g) {
            // top-2 within group
            float a = -INFINITY, b = -INFINITY;
            for (uint i = 0; i < per; ++i) {
                float s = scores[g * per + i];
                if (s > a)      { b = a; a = s; }
                else if (s > b) { b = s; }
            }
            gscore[g] = a + b;
        }
        // find topk_group-th largest threshold via selection sort
        float tmp[MAX_GROUPS];
        for (uint g = 0; g < A.n_group; ++g) tmp[g] = gscore[g];
        float thr = -INFINITY;
        for (uint k = 0; k < A.topk_group; ++k) {
            float best = -INFINITY; uint bi = 0;
            for (uint g = 0; g < A.n_group; ++g) {
                if (tmp[g] > best) { best = tmp[g]; bi = g; }
            }
            thr = best;
            tmp[bi] = -INFINITY;
        }
        gthresh = thr;
        // mask scores in non-selected groups
        for (uint g = 0; g < A.n_group; ++g) {
            if (gscore[g] < gthresh) {
                for (uint i = 0; i < per; ++i) scores[g * per + i] = -INFINITY;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3) top-K over scores; gather weights from logits (raw sigmoid).
    if (tid == 0) {
        float ws = 0.0f;
        for (uint k = 0; k < A.K; ++k) {
            float best = -INFINITY;
            int   bi   = 0;
            for (uint e = 0; e < A.N; ++e) {
                float s = scores[e];
                if (s > best) { best = s; bi = (int)e; }
            }
            float w = logits[bi];
            top_idx[(size_t)row * A.K + k]   = bi;
            top_score[(size_t)row * A.K + k] = half(w);
            ws += w;
            scores[bi] = -INFINITY;
        }
        if (A.norm_topk_prob) {
            float inv = 1.0f / (ws + 1e-20f);
            for (uint k = 0; k < A.K; ++k) {
                float w = float(top_score[(size_t)row * A.K + k]);
                top_score[(size_t)row * A.K + k] = half(w * inv * A.routed_scaling);
            }
        } else if (A.routed_scaling != 1.0f) {
            for (uint k = 0; k < A.K; ++k) {
                float w = float(top_score[(size_t)row * A.K + k]);
                top_score[(size_t)row * A.K + k] = half(w * A.routed_scaling);
            }
        }
        (void)wsum;
    }
}
