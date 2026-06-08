//
//  embedding.metal — token-id → row gather.
//
//  out[n, :] = table[ids[n], :]    for n in 0..N
//
//  Vectorized bfloat4 copy. One thread per 4-element chunk of D.
//  Grid: ((D/4 + 127) / 128, N, 1). TG: (128, 1, 1).
//  D must be divisible by 4 (true for every practical d_model).
//  Out-of-range ids zero-fill instead of crashing — matches torch
//  semantics with padding_idx=0 well enough for inference.
//

#include <metal_stdlib>
using namespace metal;

[[host_name("embedding_lookup_bf16")]]
[[kernel]]
void embedding_lookup_bf16(
    device const bfloat* table   [[buffer(0)]],   // (V, D)
    device const int*  ids     [[buffer(1)]],   // (N,)
    device       bfloat* out     [[buffer(2)]],   // (N, D)
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

    bfloat4 v = oob
        ? bfloat4(0)
        : reinterpret_cast<const device bfloat4*>(table + (uint)id * D)[d4];
    reinterpret_cast<device bfloat4*>(out + n * D)[d4] = v;
}

// Q8_0 block: half d (per-32 scale) + int8 qs[32] → 34 bytes / 32 weights.
struct __attribute__((packed)) emb_q8_block { half d; int8_t qs[32]; };

// embedding_lookup_q8_bf16 — Q8_0 token-embed gather + dequant.
// Table stored Q8_0 row-major (V, D); reuses the tied lm_head Q8 buffer so the
// bf16 embed table need not stay resident. One thread per output element.
[[host_name("embedding_lookup_q8_bf16")]]
[[kernel]]
void embedding_lookup_q8_bf16(
    device const uchar*  table_q8 [[buffer(0)]],   // (V, D) q8_0
    device const int*    ids      [[buffer(1)]],   // (N,)
    device       bfloat* out      [[buffer(2)]],   // (N, D)
    constant uint& N            [[buffer(3)]],
    constant uint& D            [[buffer(4)]],
    constant uint& V            [[buffer(5)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint n = gid.y;
    const uint d = gid.x * 256 + tid;
    if (n >= N || d >= D) return;

    int  id  = ids[n];
    bool oob = (id < 0) || ((uint)id >= V);

    bfloat val = bfloat(0);
    if (!oob) {
        const uint nb_row = D / 32u;
        device const emb_q8_block* row =
            (device const emb_q8_block*)(table_q8) + (size_t)id * nb_row;
        const uint blk  = d >> 5;
        const uint lane = d & 31u;
        val = bfloat((float)row[blk].qs[lane] * (float)row[blk].d);
    }
    out[n * D + d] = val;
}
