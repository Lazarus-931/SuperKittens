//  bias_add.metal — C[t, n] += bias[n], broadcast over T rows of a [T, N] fp16
//  buffer. Standalone so the shared quant matvec (q4k/q8_0/q6k) stays
//  byte-identical; only the QKV-bias families (Qwen2/Qwen2.5) dispatch it.
//
//  bias is bound with its per-layer byte offset already applied by the host
//  (the launcher passes a [n_layers, N] slab), so it indexes [0, N) here.

#include <metal_stdlib>
using namespace metal;

[[host_name("bias_add")]]
[[kernel]]
void bias_add(
    device       half* C    [[buffer(0)]],
    device const half* bias [[buffer(1)]],
    constant uint& N        [[buffer(2)]],
    constant uint& T        [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = N * T;
    if (gid >= total) return;
    const uint n = gid % N;
    C[gid] = half(float(C[gid]) + float(bias[n]));
}
