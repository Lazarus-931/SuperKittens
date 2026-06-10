// dg_kernels.metal — DiffusionGemma family kernels.
//
// dg_softmax_mask: masked row softmax for the GEMM-composed unified attention
// (QK^T tiles come from kernels/gemm/gemm_mma.metal at M = P+C). The D=128
// production attention kernels don't apply at head_dim 256/512, so Stage 1
// runs QK^T -> this kernel -> @V as three plain dispatches.
//
// S    : fp16 [R, ncols], R = n_heads * n_tok (head-major), softmaxed in place
// mask : f32 [n_tok, ncols] additive (0 / -inf); row r uses mask row r % n_tok
//        (one mask per layer, shared across heads; pad cols carry -inf)
// scale: kq scale applied before the mask (1.0 for this family — qk-norm)

#include <metal_stdlib>
using namespace metal;

kernel void dg_softmax_mask(
    device half        *S      [[buffer(0)]],
    device const float *mask   [[buffer(1)]],
    constant uint      &ncols  [[buffer(2)]],
    constant uint      &ntok   [[buffer(3)]],
    constant float     &scale  [[buffer(4)]],
    uint3  tgpig  [[threadgroup_position_in_grid]],
    uint3  tid3   [[thread_position_in_threadgroup]],
    uint3  tptg3  [[threads_per_threadgroup]])
{
    const uint tid  = tid3.x;
    const uint tptg = tptg3.x;
    const uint r = tgpig.x;
    device half        *row  = S + (size_t)r * ncols;
    device const float *mrow = mask + (size_t)(r % ntok) * ncols;

    threadgroup float red[256];

    float mx = -INFINITY;
    for (uint c = tid; c < ncols; c += tptg) {
        mx = max(mx, (float)row[c] * scale + mrow[c]);
    }
    red[tid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tptg / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = max(red[tid], red[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float rmax = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum = 0.0f;
    for (uint c = tid; c < ncols; c += tptg) {
        sum += exp((float)row[c] * scale + mrow[c] - rmax);
    }
    red[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tptg / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = 1.0f / red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // recompute exp from the original score so each prob is rounded to fp16 once
    for (uint c = tid; c < ncols; c += tptg) {
        row[c] = (half)(exp((float)row[c] * scale + mrow[c] - rmax) * inv);
    }
}
