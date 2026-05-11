//
//  cast.metal — fp16 ↔ fp32 element-wise casts.
//
//  Needed in the DS4 path because dsv4_rope_tail_f32 and flash_attn_ext_vec
//  consume/emit fp32, while the surrounding GEMMs are fp16. These trivial
//  kernels bridge the gap until we write fp16 variants of those ds4 kernels.
//

#include <metal_stdlib>
using namespace metal;

[[host_name("cast_f16_to_f32")]]
[[kernel]]
void cast_f16_to_f32(
    device const half*  src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    dst[gid] = float(src[gid]);
}

[[host_name("cast_f32_to_f16")]]
[[kernel]]
void cast_f32_to_f16(
    device const float* src [[buffer(0)]],
    device half*        dst [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    dst[gid] = half(src[gid]);
}
