//  causal_mask.metal — fill a (q_seq, kv_len) fp16 causal mask in-place.

#include <metal_stdlib>
using namespace metal;

[[host_name("causal_mask_fill")]]
[[kernel]]
void causal_mask_fill(
    device half*    mask     [[buffer(0)]],
    constant uint&  q_seq    [[buffer(1)]],
    constant uint&  kv_len   [[buffer(2)]],
    constant uint&  q_offset [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.y;
    const uint j = gid.x;
    if (i >= q_seq || j >= kv_len) return;
    const uint q_logical = q_offset + i;
    mask[i * kv_len + j] = (j <= q_logical) ? half(0.0) : half(-65504.0);
}
