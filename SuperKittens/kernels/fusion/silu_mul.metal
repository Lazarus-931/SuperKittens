//  silu_mul.metal — y[i] = silu(gate[i]) * up[i] for two fp16 buffers.

#include <metal_stdlib>
using namespace metal;

[[host_name("silu_mul_f16")]]
[[kernel]]
void silu_mul_f16(
    device const half* gate [[buffer(0)]],
    device const half* up   [[buffer(1)]],
    device       half* out  [[buffer(2)]],
    constant uint& N        [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= N) return;
    float g = float(gate[gid]);
    float u = float(up[gid]);
    float silu = g / (1.0f + metal::fast::exp(-g));
    out[gid] = half(silu * u);
}
