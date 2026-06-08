#include <metal_stdlib>
using namespace metal;

[[host_name("qwen_rope_qk")]]
[[kernel, max_total_threads_per_threadgroup(1024)]]
void qwen_rope_qk(
    device half* Q          [[buffer(0)]],
    device half* K          [[buffer(1)]],
    device const half* cos  [[buffer(2)]],
    device const half* sin  [[buffer(3)]],
    constant uint& seq      [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& n_heads  [[buffer(6)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tpg [[threads_per_threadgroup]])
{
    const uint head = gid.x;
    if (head >= n_heads) return;
    const uint row_blk = gid.y;
    const uint rows_per_tg = tpg.y;
    const uint row = row_blk * rows_per_tg + tid.y;
    if (row >= seq) return;

    const uint hd  = head_dim / 2;
    const uint hd4 = hd / 4;
    const uint d4  = tid.x;
    if (d4 >= hd4) return;
    const uint i = d4 * 4;

    const size_t qk_off = ((size_t)row * n_heads + head) * head_dim;
    const size_t cs_off = (size_t)row * hd + i;

    float4 q_lo = float4(*reinterpret_cast<device half4*>(Q + qk_off + i));
    float4 q_hi = float4(*reinterpret_cast<device half4*>(Q + qk_off + i + hd));
    float4 k_lo = float4(*reinterpret_cast<device half4*>(K + qk_off + i));
    float4 k_hi = float4(*reinterpret_cast<device half4*>(K + qk_off + i + hd));

    float4 c  = float4(*reinterpret_cast<const device half4*>(cos + cs_off));
    float4 sv = float4(*reinterpret_cast<const device half4*>(sin + cs_off));

    *reinterpret_cast<device half4*>(Q + qk_off + i)      = half4(q_lo * c - q_hi * sv);
    *reinterpret_cast<device half4*>(Q + qk_off + i + hd) = half4(q_lo * sv + q_hi * c);
    *reinterpret_cast<device half4*>(K + qk_off + i)      = half4(k_lo * c - k_hi * sv);
    *reinterpret_cast<device half4*>(K + qk_off + i + hd) = half4(k_lo * sv + k_hi * c);
}

// Interleaved-pair RoPE (GGML rope type 0 / GGML_ROPE_TYPE_NORM): rotates adjacent
// dims (2i, 2i+1) with freq inv_freq[i]. Llama-arch GGUFs (Nemotron-Nano) store Q/K
// permuted by the HF->GGUF converter so this NORM convention reproduces HF
// rotate_half; the split-half kernel above (rope type 2 / NEOX, Qwen3) would
// scramble positions. cos/sin tables are the same (freq-indexed). Grid/TG identical
// to qwen_rope_qk; each lane handles 4 contiguous dims = 2 pairs (freq d4*2, d4*2+1).
[[host_name("qwen_rope_qk_interleaved")]]
[[kernel, max_total_threads_per_threadgroup(1024)]]
void qwen_rope_qk_interleaved(
    device half* Q          [[buffer(0)]],
    device half* K          [[buffer(1)]],
    device const half* cos  [[buffer(2)]],
    device const half* sin  [[buffer(3)]],
    constant uint& seq      [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& n_heads  [[buffer(6)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tpg [[threads_per_threadgroup]])
{
    const uint head = gid.x;
    if (head >= n_heads) return;
    const uint rows_per_tg = tpg.y;
    const uint row = gid.y * rows_per_tg + tid.y;
    if (row >= seq) return;

    const uint hd  = head_dim / 2;
    const uint hd4 = hd / 4;
    const uint d4  = tid.x;
    if (d4 >= hd4) return;

    const size_t qk_off = ((size_t)row * n_heads + head) * head_dim;
    // hd4 lanes cover head_dim dims => 8 dims (4 pairs) per lane. Lane d4 owns dims
    // [d4*8 .. d4*8+7] = pairs (d4*8+2p, d4*8+2p+1) with freq d4*4+p, p in 0..3.
    const uint e  = d4 * 8;        // first dim index this lane writes
    const uint k0 = d4 * 4;        // first freq index
    const size_t cs_off = (size_t)row * hd + k0;
    float4 cv = float4(*reinterpret_cast<const device half4*>(cos + cs_off));
    float4 sv = float4(*reinterpret_cast<const device half4*>(sin + cs_off));

    device half* q = Q + qk_off + e;
    device half* kk = K + qk_off + e;
    for (uint p = 0; p < 4; ++p) {
        const float c = cv[p], s = sv[p];
        const float q0 = float(q[2*p]),  q1 = float(q[2*p+1]);
        const float kv0 = float(kk[2*p]), kv1 = float(kk[2*p+1]);
        q[2*p]    = half(q0 * c - q1 * s);
        q[2*p+1]  = half(q0 * s + q1 * c);
        kk[2*p]   = half(kv0 * c - kv1 * s);
        kk[2*p+1] = half(kv0 * s + kv1 * c);
    }
}
