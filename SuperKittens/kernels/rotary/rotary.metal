//
//  rotary.metal
//  SuperKittens — RoPE (Rotary Position Embedding)
//
//  Per-row, per-head TG. half4 vectorized, fp32 math. cos/sin in [-1,1].
//  Grid: (n_heads, row_blocks, 1). TG: (hd4, rows_per_tg, 1).
//

#include <metal_stdlib>
using namespace metal;

namespace meow::rotary {

[[host_name("rope_qk")]]
[[kernel, max_total_threads_per_threadgroup(512)]]
void rope_qk(
    device half* Q          [[buffer(0)]],
    device half* K          [[buffer(1)]],
    device const half* cos  [[buffer(2)]],
    device const half* sin  [[buffer(3)]],
    constant uint& seq      [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& n_heads  [[buffer(6)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tpg [[threads_per_threadgroup]])
{
    const uint head = gid.x;
    if (head >= n_heads) return;
    const uint row_blk = gid.y;
    const uint rows_per_tg = tpg.y;
    const uint row = row_blk * rows_per_tg + tid.y;
    if (row >= seq) return;

    const uint hd  = head_dim / 2;
    const uint hd4 = hd / 4;
    const uint d4  = tid.x;
    if (d4 >= hd4) return;
    const uint i = d4 * 4;

    const size_t head_off = (size_t)head * seq * head_dim;
    const size_t qk_off = head_off + (size_t)row * head_dim;
    const size_t cs_off = (size_t)row * hd + i;

    float4 q_lo = float4(*reinterpret_cast<device half4*>(Q + qk_off + i));
    float4 q_hi = float4(*reinterpret_cast<device half4*>(Q + qk_off + i + hd));
    float4 k_lo = float4(*reinterpret_cast<device half4*>(K + qk_off + i));
    float4 k_hi = float4(*reinterpret_cast<device half4*>(K + qk_off + i + hd));

    float4 c  = float4(*reinterpret_cast<const device half4*>(cos + cs_off));
    float4 sv = float4(*reinterpret_cast<const device half4*>(sin + cs_off));

    *reinterpret_cast<device half4*>(Q + qk_off + i)      = half4(q_lo * c - q_hi * sv);
    *reinterpret_cast<device half4*>(Q + qk_off + i + hd) = half4(q_lo * sv + q_hi * c);
    *reinterpret_cast<device half4*>(K + qk_off + i)      = half4(k_lo * c - k_hi * sv);
    *reinterpret_cast<device half4*>(K + qk_off + i + hd) = half4(k_lo * sv + k_hi * c);
}

} // namespace meow::rotary
