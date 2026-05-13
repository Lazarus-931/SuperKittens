//
//  kv_cache.metal — generic K/V cache write for transformer-style decoders.
//      buf_t = (pos + t_in) % cache_size
//  cache_size = window for SWA, = seq_max for unbounded append.
//  Used by all dense-attention models. Mamba/paged_attn have their own.
//

#include <metal_stdlib>
using namespace metal;

namespace meow::gemma4::kv_cache_bf16 {

[[host_name("gemma4_kv_cache_write_bf16")]]
[[kernel]]
void gemma4_kv_cache_write_bf16(
    device const bfloat* new_k       [[buffer(0)]],   // (B, H_kv, seq_in, D)
    device const bfloat* new_v       [[buffer(1)]],
    device       bfloat* k_cache     [[buffer(2)]],   // (B, H_kv, cache_size, D)
    device       bfloat* v_cache     [[buffer(3)]],
    constant uint& B               [[buffer(4)]],
    constant uint& H_kv            [[buffer(5)]],
    constant uint& D_head          [[buffer(6)]],
    constant uint& seq_in          [[buffer(7)]],
    constant uint& pos             [[buffer(8)]],   // logical position of new_k[0]
    constant uint& cache_size      [[buffer(9)]],   // physical buffer dim (window or seq_max)
    uint3 gid  [[thread_position_in_grid]])
{
    const uint d4    = gid.x;
    const uint t_in  = gid.y;
    const uint bh    = gid.z;
    const uint D4    = D_head / 4;
    if (d4 >= D4 || t_in >= seq_in || bh >= B * H_kv) return;

    const uint buf_t = (pos + t_in) % cache_size;

    // Per-head row offsets in halves.
    const size_t in_row  = ((size_t)bh * seq_in     + t_in)  * D_head;
    const size_t out_row = ((size_t)bh * cache_size + buf_t) * D_head;

    // Vectorized bfloat4 copy.
    bfloat4 k = reinterpret_cast<const device bfloat4*>(new_k + in_row)[d4];
    bfloat4 v = reinterpret_cast<const device bfloat4*>(new_v + in_row)[d4];
    reinterpret_cast<device bfloat4*>(k_cache + out_row)[d4] = k;
    reinterpret_cast<device bfloat4*>(v_cache + out_row)[d4] = v;
}

} // namespace meow::gemma4::kv_cache_bf16
