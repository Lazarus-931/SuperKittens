// argmax_bf16.metal — BF16 argmax for Gemma 4.

#include <metal_stdlib>
using namespace metal;

namespace meow::gemma4::argmaxbf {

// Drop-in 2-pass bf16 argmax (gemma4 LM head, V=262144).
// Same host_name + buffer/dispatch contract; numerically equivalent to the
// prior scalar single-pass implementation. Uses bf16x4 loads to cut memory
// ops 4× on the inner scan; threadgroup reduce unchanged. Ties broken by
// lowest index.
[[host_name("argmax_bf16")]]
[[kernel, max_total_threads_per_threadgroup(1024)]]
void argmax_bf16(
    device const bfloat* logits  [[buffer(0)]],
    device       int*    out     [[buffer(1)]],
    constant uint&       V       [[buffer(2)]],
    uint  row    [[threadgroup_position_in_grid]],
    uint  tid    [[thread_index_in_threadgroup]],
    uint  tcount [[threads_per_threadgroup]])
{
    const size_t row_off = (size_t)row * V;

    // Pass 1: per-thread scan with bfloat4 loads on the aligned bulk;
    // scalar tail clean-up for the last (V % 4) elements.
    const uint V4 = V / 4;
    const uint tail = V & 3u;
    const device bfloat4* logits4 = (const device bfloat4*)(logits + row_off);

    float best_val = -INFINITY;
    int   best_idx = INT_MAX;
    for (uint g = tid; g < V4; g += tcount) {
        float4 v = float4(logits4[g]);
        const uint i0 = g * 4u;
        if (v.x > best_val) { best_val = v.x; best_idx = (int)(i0 + 0u); }
        if (v.y > best_val) { best_val = v.y; best_idx = (int)(i0 + 1u); }
        if (v.z > best_val) { best_val = v.z; best_idx = (int)(i0 + 2u); }
        if (v.w > best_val) { best_val = v.w; best_idx = (int)(i0 + 3u); }
    }
    if (tid < tail) {
        const uint i = V4 * 4u + tid;
        float v = (float)logits[row_off + i];
        if (v > best_val) { best_val = v; best_idx = (int)i; }
    }

    // Pass 2: simdgroup-wide reduce by max-on-val; lowest-index tie break.
    threadgroup float vals[32];
    threadgroup int   idxs[32];
    const uint sg = tid / 32;
    const uint ln = tid % 32;
    float v_red = simd_max(best_val);
    int   i_pick = simd_min(best_val == v_red ? best_idx : INT_MAX);

    if (ln == 0) { vals[sg] = v_red; idxs[sg] = i_pick; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        const uint n_sg = (tcount + 31) / 32;
        float v2 = (ln < n_sg) ? vals[ln] : -INFINITY;
        int   i2 = (ln < n_sg) ? idxs[ln] : INT_MAX;
        float v3 = simd_max(v2);
        int   i3 = simd_min(v2 == v3 ? i2 : INT_MAX);
        if (ln == 0) out[row] = i3;
    }
}

} // namespace
