//
//  granite_ops.metal — Granite-4.x hybrid elementwise ops.
//
//  Granite carries four scalar multipliers (embedding 12, residual 0.22,
//  attention 1/128, logits 1/6) that no shared SK kernel applies; these two
//  kernels plumb them without touching shared-kernel signatures.

#include <metal_stdlib>
using namespace metal;

// x *= s, in place. Also used to pre-scale Q so mha_causal's hardcoded
// 1/sqrt(D) softmax scale becomes granite's attention_multiplier:
// s = attention_multiplier * sqrt(D).
[[host_name("granite_scale_f16")]]
[[kernel]]
void granite_scale_f16(
    device half*    x [[buffer(0)]],
    constant float& s [[buffer(1)]],
    constant uint&  n [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;
    x[tid] = half(float(x[tid]) * s);
}

// y = a + s*b (residual_multiplier on every mixer/FFN block output).
[[host_name("granite_add_scaled_f16")]]
[[kernel]]
void granite_add_scaled_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half*       y [[buffer(2)]],
    constant float&    s [[buffer(3)]],
    constant uint&     n [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;
    y[tid] = half(float(a[tid]) + s * float(b[tid]));
}
