//
//  prope.metal — proportional RoPE (p=0.25) for Gemma 4 global layers.
//  Rotates the first p_pairs frequency pairs; leaves the upper 75% untouched.
//  In-place on Q, K. Grid (n_heads, seq) × TG (128).
//

#include <metal_stdlib>
using namespace metal;

namespace meow::gemma4::prope {

[[host_name("gemma4_prope_qk")]]
[[kernel, max_total_threads_per_threadgroup(128)]]
void gemma4_prope_qk(
    device bfloat* Q          [[buffer(0)]],
    device bfloat* K          [[buffer(1)]],
    device const bfloat* cos  [[buffer(2)]],
    device const bfloat* sin  [[buffer(3)]],
    constant uint& seq      [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& n_heads  [[buffer(6)]],
    constant uint& p_pairs  [[buffer(7)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint  lid [[thread_index_in_threadgroup]])
{
    const uint head = gid.x;
    const uint t    = gid.y;
    if (head >= n_heads || t >= seq) return;

    const uint hd_half  = head_dim / 2;
    const size_t base   = (size_t)head * seq * head_dim + (size_t)t * head_dim;
    const size_t cs_off = (size_t)t * hd_half;

    // For each pair index p in [0, p_pairs): rotate (lo, hi) where
    //     lo = elem at index p
    //     hi = elem at index p + hd_half
    //
    // Splits the head_dim into 2 halves (lo: [0, hd_half), hi: [hd_half, head_dim)).
    // Standard RoPE pair rotation:
    //     lo' = lo * c - hi * s
    //     hi' = lo * s + hi * c
    //
    // Pairs at index >= p_pairs are NOT touched (Q/K pass through verbatim
    // in those dims). Caller is responsible for ensuring those pairs were
    // never rotated by some upstream pass.

    for (uint p = lid; p < p_pairs; p += 128) {
        const float c = (float)cos[cs_off + p];
        const float s = (float)sin[cs_off + p];

        // Q
        {
            float qlo = (float)Q[base + p];
            float qhi = (float)Q[base + p + hd_half];
            Q[base + p]            = bfloat(qlo * c - qhi * s);
            Q[base + p + hd_half]  = bfloat(qlo * s + qhi * c);
        }
        // K
        {
            float klo = (float)K[base + p];
            float khi = (float)K[base + p + hd_half];
            K[base + p]            = bfloat(klo * c - khi * s);
            K[base + p + hd_half]  = bfloat(klo * s + khi * c);
        }
    }
    // Pairs in [p_pairs, hd_half) are deliberately untouched.
}

} // namespace meow::gemma4::prope
