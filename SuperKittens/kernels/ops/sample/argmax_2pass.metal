//
//  argmax_2pass.metal — 2-PSO 2-pass parallel block-reduce argmax.
//
//  Drop-in faster alternative to the single-TG argmax for large vocabularies.
//  Ported from temp/ops_lab/kernels.metal after bit-exact verification vs the
//  in-tree single-pass kernel (V=151936 → 98765, V=262144 → 12345).
//
//  Pass 1 (argmax_partial / argmax_bf16_partial):
//    - Grid = n_tiles, each TG = 1024 threads.
//    - Each TG covers ARGMAX2PASS_ELTS_PER_TG=16384 contiguous logits.
//    - Writes (partial_val, partial_idx) into scratch[tg].
//
//  Pass 2 (argmax_reduce):
//    - Grid = 1, TG = 1024 threads.
//    - Reduces n_tiles partials → single output index.
//
//  Host launch contract:
//    - Compute n_blocks = ceil(V / 16384).
//    - Allocate scratch_val: n_blocks * sizeof(float).
//    - Allocate scratch_idx: n_blocks * sizeof(int32_t).
//    - Dispatch partial(grid=n_blocks, tg=1024) with (logits, scratch_val,
//      scratch_idx, V).
//    - Dispatch reduce(grid=1, tg=1024) with (scratch_val, scratch_idx, out, n_blocks).
//
//  Bench (M4, V=262144 bf16): 7.5 µs vs 43.9 µs single-pass (5.9×).
//                 V=151936 f16:  6.8 µs vs 25.4 µs (3.7×).
//                 V=102400 f16:  5.1 µs vs 19.5 µs (3.8×).
//                 V= 50288 f16:  5.1 µs vs  9.4 µs (1.8×).

#include <metal_stdlib>
using namespace metal;

namespace meow::ops::sample {

constant uint ARGMAX2PASS_TG_THREADS  = 1024;
constant uint ARGMAX2PASS_ELTS_PER_TG = 16384;

// ─── Pass 1: fp16 per-tile partial ────────────────────────────────────
[[host_name("argmax_partial")]]
[[kernel, max_total_threads_per_threadgroup(1024)]]
void argmax_partial(
    device const half*   logits  [[buffer(0)]],
    device       float*  val_buf [[buffer(1)]],
    device       int*    idx_buf [[buffer(2)]],
    constant uint&       V       [[buffer(3)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tcnt [[threads_per_threadgroup]])
{
    const uint base = tg * ARGMAX2PASS_ELTS_PER_TG;
    const uint end  = min(base + ARGMAX2PASS_ELTS_PER_TG, V);
    float best_v = -INFINITY; int best_i = INT_MAX;
    for (uint i = base + tid; i < end; i += tcnt) {
        float v = (float)logits[i];
        if (v > best_v) { best_v = v; best_i = (int)i; }
    }
    threadgroup float vals[32]; threadgroup int idxs[32];
    uint sg = tid / 32, ln = tid % 32;
    float vr = simd_max(best_v);
    int   ir = simd_min(best_v == vr ? best_i : INT_MAX);
    if (ln == 0) { vals[sg] = vr; idxs[sg] = ir; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        uint nsg = (tcnt + 31) / 32;
        float v2 = (ln < nsg) ? vals[ln] : -INFINITY;
        int   i2 = (ln < nsg) ? idxs[ln] : INT_MAX;
        float v3 = simd_max(v2);
        int   i3 = simd_min(v2 == v3 ? i2 : INT_MAX);
        if (ln == 0) { val_buf[tg] = v3; idx_buf[tg] = i3; }
    }
}

// ─── Pass 1: bf16 per-tile partial ────────────────────────────────────
[[host_name("argmax_bf16_partial")]]
[[kernel, max_total_threads_per_threadgroup(1024)]]
void argmax_bf16_partial(
    device const bfloat* logits  [[buffer(0)]],
    device       float*  val_buf [[buffer(1)]],
    device       int*    idx_buf [[buffer(2)]],
    constant uint&       V       [[buffer(3)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tcnt [[threads_per_threadgroup]])
{
    const uint base = tg * ARGMAX2PASS_ELTS_PER_TG;
    const uint end  = min(base + ARGMAX2PASS_ELTS_PER_TG, V);
    float best_v = -INFINITY; int best_i = INT_MAX;
    for (uint i = base + tid; i < end; i += tcnt) {
        float v = (float)logits[i];
        if (v > best_v) { best_v = v; best_i = (int)i; }
    }
    threadgroup float vals[32]; threadgroup int idxs[32];
    uint sg = tid / 32, ln = tid % 32;
    float vr = simd_max(best_v);
    int   ir = simd_min(best_v == vr ? best_i : INT_MAX);
    if (ln == 0) { vals[sg] = vr; idxs[sg] = ir; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        uint nsg = (tcnt + 31) / 32;
        float v2 = (ln < nsg) ? vals[ln] : -INFINITY;
        int   i2 = (ln < nsg) ? idxs[ln] : INT_MAX;
        float v3 = simd_max(v2);
        int   i3 = simd_min(v2 == v3 ? i2 : INT_MAX);
        if (ln == 0) { val_buf[tg] = v3; idx_buf[tg] = i3; }
    }
}

// ─── Pass 2: reduce partials (dtype-agnostic) ─────────────────────────
[[host_name("argmax_reduce")]]
[[kernel, max_total_threads_per_threadgroup(1024)]]
void argmax_reduce(
    device const float* val_buf [[buffer(0)]],
    device const int*   idx_buf [[buffer(1)]],
    device       int*   out     [[buffer(2)]],
    constant uint&      N       [[buffer(3)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tcnt [[threads_per_threadgroup]])
{
    float best_v = -INFINITY; int best_i = INT_MAX;
    for (uint i = tid; i < N; i += tcnt) {
        float v = val_buf[i];
        if (v > best_v) { best_v = v; best_i = idx_buf[i]; }
    }
    threadgroup float vals[32]; threadgroup int idxs[32];
    uint sg = tid / 32, ln = tid % 32;
    float vr = simd_max(best_v);
    int   ir = simd_min(best_v == vr ? best_i : INT_MAX);
    if (ln == 0) { vals[sg] = vr; idxs[sg] = ir; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        uint nsg = (tcnt + 31) / 32;
        float v2 = (ln < nsg) ? vals[ln] : -INFINITY;
        int   i2 = (ln < nsg) ? idxs[ln] : INT_MAX;
        float v3 = simd_max(v2);
        int   i3 = simd_min(v2 == v3 ? i2 : INT_MAX);
        if (ln == 0) out[0] = i3;
    }
}

} // namespace meow::ops::sample
