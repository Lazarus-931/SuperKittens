// gemma4_argmax_bf16.metal — BF16 argmax for Gemma 4.

#include <metal_stdlib>
using namespace metal;

namespace meow::gemma4::argmaxbf {

[[host_name("gemma4_argmax_bf16")]]
[[kernel, max_total_threads_per_threadgroup(1024)]]
void gemma4_argmax_bf16(
    device const bfloat* logits  [[buffer(0)]],
    device       int*    out     [[buffer(1)]],
    constant uint&       V       [[buffer(2)]],
    uint  row    [[threadgroup_position_in_grid]],
    uint  tid    [[thread_index_in_threadgroup]],
    uint  tcount [[threads_per_threadgroup]])
{
    const size_t row_off = (size_t)row * V;

    float best_val = -INFINITY;
    int   best_idx = 0;
    for (uint i = tid; i < V; i += tcount) {
        float v = (float)logits[row_off + i];
        if (v > best_val) { best_val = v; best_idx = (int)i; }
    }

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
