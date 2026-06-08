// mla_glue.metal — DeepSeek V2-Lite MLA decode glue feeding kernel_mla_decode_v2.
//
// kernel_mla_decode_v2 wants per-head dense fp16 K rows of width dk=192
// (qk_nope ++ qk_rope) and V rows of width dv=128, indexed [head, kv_pos, dim].
// The up-projection (kv_up_pair) gives k_nope [T,H,128] and v [T,H,128]; the
// shared rotated key k_pe [T,64] is broadcast across heads. This kernel writes
// the current decode step's K/V into the per-head cache at write_pos.

#include <metal_stdlib>
using namespace metal;

// Assemble + cache-write for ONE decode step (T tokens, decode T usually 1).
// K cache layout: [head, cache_max, 192] fp16; V cache: [head, cache_max, 128].
[[host_name("deepseek_mla_kv_write")]]
kernel void deepseek_mla_kv_write(
        device const half * k_nope   [[buffer(0)]],   // [T, H, 128] fp16
        device const half * k_pe     [[buffer(1)]],   // [T, 64]     fp16 (shared)
        device const half * v        [[buffer(2)]],   // [T, H, 128] fp16
        device       half * k_cache  [[buffer(3)]],   // [H, cache_max, 192]
        device       half * v_cache  [[buffer(4)]],   // [H, cache_max, 128]
        constant uint &     T         [[buffer(5)]],
        constant uint &     H         [[buffer(6)]],
        constant uint &     qk_nope   [[buffer(7)]],   // 128
        constant uint &     qk_rope   [[buffer(8)]],   // 64
        constant uint &     v_dim     [[buffer(9)]],   // 128
        constant uint &     cache_max [[buffer(10)]],
        constant uint &     write_pos [[buffer(11)]],
        uint3 gid [[thread_position_in_grid]]) {
    const uint t = gid.z;   // token in this step
    const uint h = gid.y;   // head
    const uint i = gid.x;   // element within the dk row (0..dk-1)
    const uint dk = qk_nope + qk_rope;
    if (t >= T || h >= H || i >= dk) return;

    const uint pos = write_pos + t;
    if (pos >= cache_max) return;

    device half * krow = k_cache + ((size_t)h * cache_max + pos) * dk;
    if (i < qk_nope) {
        krow[i] = k_nope[((size_t)t * H + h) * qk_nope + i];
    } else {
        // rope half is shared across heads
        krow[i] = k_pe[(size_t)t * qk_rope + (i - qk_nope)];
    }

    if (i < v_dim) {
        device half * vrow = v_cache + ((size_t)h * cache_max + pos) * v_dim;
        vrow[i] = v[((size_t)t * H + h) * v_dim + i];
    }
}
