//
//  embedding.metal — token-id → row gather.
//
//  out[n, :] = table[ids[n], :]    for n in 0..N
//
//  Vectorized half4 copy. One thread per 4-element chunk of D.
//  Grid: ((D/4 + 127) / 128, N, 1). TG: (128, 1, 1).
//  D must be divisible by 4 (true for every practical d_model).
//  Out-of-range ids zero-fill instead of crashing — matches torch
//  semantics with padding_idx=0 well enough for inference.
//

#include <metal_stdlib>
using namespace metal;

[[host_name("embedding_lookup")]]
[[kernel]]
void embedding_lookup(
    device const half* table   [[buffer(0)]],   // (V, D)
    device const int*  ids     [[buffer(1)]],   // (N,)
    device       half* out     [[buffer(2)]],   // (N, D)
    constant uint& N           [[buffer(3)]],
    constant uint& D           [[buffer(4)]],
    constant uint& V           [[buffer(5)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint n  = gid.y;
    const uint d4 = gid.x * 128 + tid;
    const uint D4 = D / 4;
    if (n >= N || d4 >= D4) return;

    int  id  = ids[n];
    bool oob = (id < 0) || ((uint)id >= V);

    half4 v = oob
        ? half4(0)
        : reinterpret_cast<const device half4*>(table + (uint)id * D)[d4];
    reinterpret_cast<device half4*>(out + n * D)[d4] = v;
}
