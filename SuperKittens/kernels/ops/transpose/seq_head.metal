//  seq_head.metal — transpose between (T, H, D) and (H, T, D) for fp16 buffers.
//  Each thread copies one half element. Grid: (D, T, H).

#include <metal_stdlib>
using namespace metal;

// (T, H, D) seq-major  ->  (H, T, D) head-major
[[host_name("transpose_seq_to_head_f16")]]
[[kernel]]
void transpose_seq_to_head_f16(
    device const half* src [[buffer(0)]],   // (T, H, D)
    device       half* dst [[buffer(1)]],   // (H, T, D)
    constant uint& T       [[buffer(2)]],
    constant uint& H       [[buffer(3)]],
    constant uint& D       [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint d = gid.x, t = gid.y, h = gid.z;
    if (d >= D || t >= T || h >= H) return;
    dst[((size_t)h * T + t) * D + d] = src[((size_t)t * H + h) * D + d];
}

// (H, T, D) head-major  ->  (T, H, D) seq-major
[[host_name("transpose_head_to_seq_f16")]]
[[kernel]]
void transpose_head_to_seq_f16(
    device const half* src [[buffer(0)]],   // (H, T, D)
    device       half* dst [[buffer(1)]],   // (T, H, D)
    constant uint& T       [[buffer(2)]],
    constant uint& H       [[buffer(3)]],
    constant uint& D       [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint d = gid.x, t = gid.y, h = gid.z;
    if (d >= D || t >= T || h >= H) return;
    dst[((size_t)t * H + h) * D + d] = src[((size_t)h * T + t) * D + d];
}
