//
//  ple.metal — Per-Layer Embeddings for Gemma 4 E-models.
//      x[n, :] += ple_table[layer_idx, ids[n], :]
//  PLE table layout: (n_layers, V, d_model) fp16. Vectorized half4.
//

#include <metal_stdlib>
using namespace metal;

namespace meow::gemma4::ple {

[[host_name("gemma4_ple_add")]]
[[kernel]]
void gemma4_ple_add(
    device const half* ple_table  [[buffer(0)]],   // (n_layers, V, d_model)
    device const int*  ids        [[buffer(1)]],   // (N,) int32
    device       half* x          [[buffer(2)]],   // (N, d_model) fp16 in-place
    constant uint& N              [[buffer(3)]],
    constant uint& d_model        [[buffer(4)]],
    constant uint& V              [[buffer(5)]],
    constant uint& layer_idx      [[buffer(6)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint n  = gid.y;
    const uint d4 = gid.x * 128 + tid;
    const uint D4 = d_model / 4;
    if (n >= N || d4 >= D4) return;

    int  id  = ids[n];
    bool oob = (id < 0) || ((uint)id >= V);
    if (oob) return;   // zero-fill behavior: skip add (treat as +0)

    // Layer slab base: layer_idx * (V * d_model) halves
    const uint layer_off = layer_idx * V * d_model;
    const uint row_off   = (uint)id * d_model;

    half4 add  = reinterpret_cast<const device half4*>(
        ple_table + layer_off + row_off)[d4];
    half4 cur  = reinterpret_cast<device half4*>(x + n * d_model)[d4];
    reinterpret_cast<device half4*>(x + n * d_model)[d4] = cur + add;
}

} // namespace meow::gemma4::ple
