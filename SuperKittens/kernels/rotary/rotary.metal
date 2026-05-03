//
//  rotary.metal
//  SuperKittens — RoPE (Rotary Position Embedding)
//
//  1024 threads per head, grid-stride, half4 vectorized.
//  All buffers fp16 — cos/sin in [-1,1].
//

#include <metal_stdlib>
using namespace metal;

namespace meow::rotary {

[[host_name("rope_qk")]]
[[kernel, max_total_threads_per_threadgroup(1024)]]
void rope_qk(
    device half* Q          [[buffer(0)]],
    device half* K          [[buffer(1)]],
    device const half* cos  [[buffer(2)]],
    device const half* sin  [[buffer(3)]],
    constant uint& seq      [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& n_heads  [[buffer(6)]],
    uint  gid [[threadgroup_position_in_grid]],
    uint  lid [[thread_index_in_threadgroup]])
{
    const uint head = gid;
    if (head >= n_heads) return;

    const uint hd  = head_dim / 2;
    const uint hd4 = hd / 4;
    const size_t head_off = (size_t)head * seq * head_dim;
    const uint total = seq * hd4;

    for (uint idx = lid; idx < total; idx += 1024) {
        uint row = idx / hd4;
        uint d4  = idx % hd4;
        uint i   = d4 * 4;

        const size_t qk_off = head_off + (size_t)row * head_dim;
        const size_t cs_off = (size_t)row * hd + i;

        half4 q_lo = reinterpret_cast<device half4*>(Q + qk_off + i)[0];
        half4 q_hi = reinterpret_cast<device half4*>(Q + qk_off + i + hd)[0];
        half4 k_lo = reinterpret_cast<device half4*>(K + qk_off + i)[0];
        half4 k_hi = reinterpret_cast<device half4*>(K + qk_off + i + hd)[0];

        float4 c = float4(reinterpret_cast<const device half4*>(cos + cs_off)[0]);
        float4 s = float4(reinterpret_cast<const device half4*>(sin + cs_off)[0]);

        float4 q0 = float4(q_lo), q1 = float4(q_hi);
        float4 k0 = float4(k_lo), k1 = float4(k_hi);

        reinterpret_cast<device half4*>(Q + qk_off + i)[0]     = half4(q0 * c - q1 * s);
        reinterpret_cast<device half4*>(Q + qk_off + i + hd)[0] = half4(q0 * s + q1 * c);
        reinterpret_cast<device half4*>(K + qk_off + i)[0]     = half4(k0 * c - k1 * s);
        reinterpret_cast<device half4*>(K + qk_off + i + hd)[0] = half4(k0 * s + k1 * c);
    }
}

} // namespace meow::rotary
