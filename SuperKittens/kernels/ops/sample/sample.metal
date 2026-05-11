//
//  sample.metal — argmax / softmax_temp / multinomial.
//  One TG (1024 threads) per row; logits/probs shape (rows, V).
//  Top-k/top-p masking is host-side; these are the GPU-hot primitives.
//

#include <metal_stdlib>
using namespace metal;

namespace meow::ops::sample {

// ─── argmax: greedy decoding ──────────────────────────────────────────
// One TG per row. 1024 threads cooperate via simdgroup + threadgroup reduce.
// Returns the index of the largest logit (ties broken by lowest index).

[[host_name("argmax")]]
[[kernel, max_total_threads_per_threadgroup(1024)]]
void argmax(
    device const half* logits  [[buffer(0)]],   // (rows, V)
    device       int*  out     [[buffer(1)]],   // (rows,)
    constant uint& V           [[buffer(2)]],
    uint  row    [[threadgroup_position_in_grid]],
    uint  tid    [[thread_index_in_threadgroup]],
    uint  tcount [[threads_per_threadgroup]])
{
    const size_t row_off = (size_t)row * V;

    // Per-thread scan: scan strided slice of V, keep best (val, idx).
    float best_val = -INFINITY;
    int   best_idx = 0;
    for (uint i = tid; i < V; i += tcount) {
        float v = (float)logits[row_off + i];
        if (v > best_val) { best_val = v; best_idx = (int)i; }
    }

    // Simdgroup-wide reduce by max-on-val.
    threadgroup float vals[32];
    threadgroup int   idxs[32];
    const uint sg = tid / 32;
    const uint ln = tid % 32;
    float v_red = simd_max(best_val);
    // We want the lowest-index lane that holds the max. simd_min over
    // indices among the lanes whose val equals v_red.
    int   i_pick = simd_min(best_val == v_red ? best_idx : INT_MAX);

    if (ln == 0) { vals[sg] = v_red; idxs[sg] = i_pick; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Threadgroup-wide reduce (up to 32 simdgroups).
    if (sg == 0) {
        const uint n_sg = (tcount + 31) / 32;
        float v2 = (ln < n_sg) ? vals[ln] : -INFINITY;
        int   i2 = (ln < n_sg) ? idxs[ln] : INT_MAX;
        float v3 = simd_max(v2);
        int   i3 = simd_min(v2 == v3 ? i2 : INT_MAX);
        if (ln == 0) out[row] = i3;
    }
}


// ─── softmax with temperature ─────────────────────────────────────────
// probs[row, v] = exp((logits[row, v] - max) / T) / Σ exp(...)
// Standard numerically-stable softmax. In-place is safe iff `probs == logits`
// in fp16 (caller passes the same pointer twice).

[[host_name("softmax_temp")]]
[[kernel, max_total_threads_per_threadgroup(1024)]]
void softmax_temp(
    device const half*  logits  [[buffer(0)]],   // (rows, V)
    device       half*  probs   [[buffer(1)]],   // (rows, V) — may alias logits
    constant uint&  V           [[buffer(2)]],
    constant float& temperature [[buffer(3)]],
    uint  row    [[threadgroup_position_in_grid]],
    uint  tid    [[thread_index_in_threadgroup]],
    uint  tcount [[threads_per_threadgroup]])
{
    const size_t row_off = (size_t)row * V;
    const float inv_T = 1.0f / max(temperature, 1e-4f);

    // Pass 1: row max
    float local_max = -INFINITY;
    for (uint i = tid; i < V; i += tcount) {
        float v = (float)logits[row_off + i] * inv_T;
        if (v > local_max) local_max = v;
    }
    threadgroup float scratch[32];
    const uint sg = tid / 32;
    const uint ln = tid % 32;
    float sg_max = simd_max(local_max);
    if (ln == 0) scratch[sg] = sg_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const uint n_sg = (tcount + 31) / 32;
        float t = (ln < n_sg) ? scratch[ln] : -INFINITY;
        t = simd_max(t);
        if (ln == 0) scratch[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float row_max = scratch[0];

    // Pass 2: write exp(scaled - max) and accumulate sum
    float local_sum = 0.0f;
    for (uint i = tid; i < V; i += tcount) {
        float v = (float)logits[row_off + i] * inv_T - row_max;
        float e = metal::fast::exp(v);
        probs[row_off + i] = (half)e;
        local_sum += e;
    }
    float sg_sum = simd_sum(local_sum);
    if (ln == 0) scratch[sg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const uint n_sg = (tcount + 31) / 32;
        float t = (ln < n_sg) ? scratch[ln] : 0.0f;
        t = simd_sum(t);
        if (ln == 0) scratch[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float row_sum = scratch[0];
    const float inv_sum = 1.0f / max(row_sum, 1e-30f);

    // Pass 3: normalize
    for (uint i = tid; i < V; i += tcount) {
        probs[row_off + i] = (half)((float)probs[row_off + i] * inv_sum);
    }
}


// ─── multinomial draw given a uniform [0, 1) float per row ────────────
// out[row] = smallest idx s.t. cumsum(probs[row, :idx+1]) >= u[row]
// Single TG per row; 1024 threads scan strided then reduce.
//
// Approach: row CDF computed in-place via a Hillis-Steele scan would be
// expensive at V=256K. Simpler: each thread scans a strided slice, keeps a
// running prefix sum and notes the threshold-crossing index. Then a TG
// reduction picks the lane with the smallest "found_idx".
//
// For inference batch size 1 (typical), this is one TG. We compute a global
// prefix sum first via threadgroup reduction, then re-scan for the index.

[[host_name("multinomial")]]
[[kernel, max_total_threads_per_threadgroup(1024)]]
void multinomial(
    device const half*  probs  [[buffer(0)]],   // (rows, V) — must be normalized
    device const float* u      [[buffer(1)]],   // (rows,)   — uniform draw
    device       int*   out    [[buffer(2)]],   // (rows,)
    constant uint& V           [[buffer(3)]],
    uint  row    [[threadgroup_position_in_grid]],
    uint  tid    [[thread_index_in_threadgroup]],
    uint  tcount [[threads_per_threadgroup]])
{
    const size_t row_off = (size_t)row * V;
    const float target = u[row];

    // Step 1: each thread accumulates its strided slice's total.
    float local_sum = 0.0f;
    for (uint i = tid; i < V; i += tcount) {
        local_sum += (float)probs[row_off + i];
    }

    // Step 2: TG-wide prefix sum on local sums (at most 1024 entries).
    threadgroup float partials[1024];
    partials[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Sequential prefix in a single thread for simplicity (1024 adds is fine).
    threadgroup float prefix[1024];   // exclusive prefix
    if (tid == 0) {
        float run = 0.0f;
        for (uint k = 0; k < tcount; ++k) {
            prefix[k] = run;
            run += partials[k];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: each thread re-walks its slice, checks threshold against
    // prefix[tid] + running. Records the global index it crossed at.
    int   found = INT_MAX;
    float run = prefix[tid];
    for (uint i = tid; i < V; i += tcount) {
        run += (float)probs[row_off + i];
        if (run >= target) { found = (int)i; break; }
    }

    // Step 4: reduce over found-indices; smallest wins.
    int min_found = simd_min(found);
    threadgroup int sg_min[32];
    const uint sg = tid / 32;
    const uint ln = tid % 32;
    if (ln == 0) sg_min[sg] = min_found;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const uint n_sg = (tcount + 31) / 32;
        int v = (ln < n_sg) ? sg_min[ln] : INT_MAX;
        v = simd_min(v);
        if (ln == 0) {
            // INT_MAX means probs didn't sum to 1 for some reason; clamp to V-1.
            out[row] = (v == INT_MAX) ? (int)(V - 1) : v;
        }
    }
}

} // namespace meow::ops::sample
