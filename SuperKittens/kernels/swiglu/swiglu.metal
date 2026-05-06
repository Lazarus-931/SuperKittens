//
//  swiglu.metal
//  SuperKittens
//
//  Fused SwiGLU: y = silu(gate) * up
//  x is [rows, 2*d] with gate in first half, up in second half.
//  SIMD-group per row, zero barriers, half4 vectorized.

#include <metal_stdlib>
using namespace metal;

enum : uint { THREADS = 128, ROWS_PER_GRP = 4 };

inline float  silu(float  x) { return x / (1.0f + metal::fast::exp(-x)); }
inline float4 silu(float4 x) { return x / (1.0f + metal::fast::exp(-x)); }

[[host_name("fused_swiglu")]]
[[kernel, max_total_threads_per_threadgroup(THREADS)]]
void fused_swiglu(
    device const half* x   [[buffer(0)]],
    device half* y         [[buffer(1)]],
    constant uint& rows    [[buffer(2)]],
    constant uint& d       [[buffer(3)]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint2 gid  [[threadgroup_position_in_grid]])
{
    const uint row = gid.y * ROWS_PER_GRP + simd;
    if (row >= rows) return;

    const size_t off = (size_t)row * 2 * d;
    const device half4* x4 = reinterpret_cast<const device half4*>(x + off);
    const device half4* u4 = reinterpret_cast<const device half4*>(x + off + d);
    device half4* y4 = reinterpret_cast<device half4*>(y + (size_t)row * d);

    const uint n4 = d / 4;

    for (uint k = lane; k < n4; k += 32) {
        y4[k] = half4(silu(float4(x4[k])) * float4(u4[k]));
    }

    for (uint k = n4 * 4 + lane; k < d; k += 32) {
        float g = silu(float(x[off + k])) * float(x[off + d + k]);
        y[(size_t)row * d + k] = half(g);
    }
}
