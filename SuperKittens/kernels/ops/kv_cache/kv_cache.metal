//
//  kv_cache.metal — generic K/V cache write for transformer-style decoders.
//      buf_t = (pos + t_in) % cache_size
//  cache_size = window for SWA, = seq_max for unbounded append.
//  Used by all dense-attention models. Mamba/paged_attn have their own.
//

#include <metal_stdlib>
using namespace metal;

namespace meow::ops::kv_cache {

[[host_name("kv_cache_write")]]
[[kernel]]
void kv_cache_write(
    device const half* new_k       [[buffer(0)]],   // (B, H_kv, seq_in, D)
    device const half* new_v       [[buffer(1)]],
    device       half* k_cache     [[buffer(2)]],   // (B, H_kv, cache_size, D)
    device       half* v_cache     [[buffer(3)]],
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

    // Vectorized half4 copy.
    half4 k = reinterpret_cast<const device half4*>(new_k + in_row)[d4];
    half4 v = reinterpret_cast<const device half4*>(new_v + in_row)[d4];
    reinterpret_cast<device half4*>(k_cache + out_row)[d4] = k;
    reinterpret_cast<device half4*>(v_cache + out_row)[d4] = v;
}

// Q8_0 KV write. Each new (B,H_kv,seq_in,D) row is split into D/32 blocks of 32
// elements; per block we store one fp16 scale (max|x|/127) and 32 int8 values.
// Layout: qs in (B,H_kv,cache_size,D) int8, sc in (B,H_kv,cache_size,D/32) fp16.
// One simdgroup of 32 lanes owns one (bh, t_in, block): each lane holds 1 elt,
// simd-reduce the block absmax, quantize, store. D must be a multiple of 32.
[[host_name("kv_cache_write_q8")]]
[[kernel]]
void kv_cache_write_q8(
    device const half* new_k       [[buffer(0)]],
    device const half* new_v       [[buffer(1)]],
    device       char* k_cache_q   [[buffer(2)]],   // (B,H_kv,cache_size,D) int8
    device       char* v_cache_q   [[buffer(3)]],
    device       half* k_cache_s   [[buffer(4)]],   // (B,H_kv,cache_size,D/32) fp16
    device       half* v_cache_s   [[buffer(5)]],
    constant uint& B               [[buffer(6)]],
    constant uint& H_kv            [[buffer(7)]],
    constant uint& D_head          [[buffer(8)]],
    constant uint& seq_in          [[buffer(9)]],
    constant uint& pos             [[buffer(10)]],
    constant uint& cache_size      [[buffer(11)]],
    uint3 gid  [[thread_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const uint nblk = D_head / 32u;                 // blocks per head row
    const uint blk  = gid.x >> 5;                    // 32 lanes per block
    const uint t_in = gid.y;
    const uint bh   = gid.z;
    if (blk >= nblk || t_in >= seq_in || bh >= B * H_kv) return;

    const uint buf_t = (pos + t_in) % cache_size;
    const size_t in_row  = ((size_t)bh * seq_in     + t_in)  * D_head + blk * 32u;
    const size_t out_row = ((size_t)bh * cache_size + buf_t) * D_head + blk * 32u;
    const size_t sc_row  = ((size_t)bh * cache_size + buf_t) * nblk   + blk;

    const float kf = float(new_k[in_row + lane]);
    const float vf = float(new_v[in_row + lane]);
    const float kd = simd_max(fabs(kf)) / 127.0f;
    const float vd = simd_max(fabs(vf)) / 127.0f;
    const float kinv = kd > 0.0f ? 1.0f / kd : 0.0f;
    const float vinv = vd > 0.0f ? 1.0f / vd : 0.0f;

    k_cache_q[out_row + lane] = (char)round(kf * kinv);
    v_cache_q[out_row + lane] = (char)round(vf * vinv);
    if (lane == 0) {
        k_cache_s[sc_row] = half(kd);
        v_cache_s[sc_row] = half(vd);
    }
}

// Q4 KV write. Same per-32-block fp16-scale layout as Q8, but values are 4-bit:
// scale = max|x|/7, q = round(x/scale) clamped to [-8,7], stored as a nibble
// biased by +8 into [0,15] so two nibbles pack into one uint8 (low nibble =
// even lane, high nibble = odd lane). Halves the qs bytes vs Q8 (D/2 vs D).
// Layout: qs in (B,H_kv,cache_size,D/2) uint8, sc in (B,H_kv,cache_size,D/32)
// fp16. One simdgroup of 32 lanes owns one (bh, t_in, block); D mult of 32.
[[host_name("kv_cache_write_q4")]]
[[kernel]]
void kv_cache_write_q4(
    device const half*  new_k      [[buffer(0)]],
    device const half*  new_v      [[buffer(1)]],
    device       uchar* k_cache_q  [[buffer(2)]],   // (B,H_kv,cache_size,D/2) packed nibbles
    device       uchar* v_cache_q  [[buffer(3)]],
    device       half*  k_cache_s  [[buffer(4)]],   // (B,H_kv,cache_size,D/32) fp16
    device       half*  v_cache_s  [[buffer(5)]],
    constant uint& B               [[buffer(6)]],
    constant uint& H_kv            [[buffer(7)]],
    constant uint& D_head          [[buffer(8)]],
    constant uint& seq_in          [[buffer(9)]],
    constant uint& pos             [[buffer(10)]],
    constant uint& cache_size      [[buffer(11)]],
    uint3 gid  [[thread_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const uint nblk = D_head / 32u;
    const uint blk  = gid.x >> 5;
    const uint t_in = gid.y;
    const uint bh   = gid.z;
    if (blk >= nblk || t_in >= seq_in || bh >= B * H_kv) return;

    const uint buf_t = (pos + t_in) % cache_size;
    const size_t in_row  = ((size_t)bh * seq_in     + t_in)  * D_head      + blk * 32u;
    const size_t out_row = ((size_t)bh * cache_size + buf_t) * (D_head / 2) + blk * 16u;
    const size_t sc_row  = ((size_t)bh * cache_size + buf_t) * nblk         + blk;

    const float kf = float(new_k[in_row + lane]);
    const float vf = float(new_v[in_row + lane]);
    const float kd = simd_max(fabs(kf)) / 7.0f;
    const float vd = simd_max(fabs(vf)) / 7.0f;
    const float kinv = kd > 0.0f ? 1.0f / kd : 0.0f;
    const float vinv = vd > 0.0f ? 1.0f / vd : 0.0f;

    // Bias signed [-8,7] by +8 -> [0,15]; clamp guards round-half overshoot at
    // the absmax element (x*inv = +-7 exactly).
    const int kq = clamp(int(round(kf * kinv)) + 8, 0, 15);
    const int vq = clamp(int(round(vf * vinv)) + 8, 0, 15);

    // Even lane owns byte (lane/2): low nibble = even lane, high nibble = odd.
    const uint kq_hi = simd_shuffle_down((uint)kq, 1u);
    const uint vq_hi = simd_shuffle_down((uint)vq, 1u);
    if ((lane & 1u) == 0u) {
        const uint byte_idx = lane >> 1;
        k_cache_q[out_row + byte_idx] = uchar((uint)kq | (kq_hi << 4));
        v_cache_q[out_row + byte_idx] = uchar((uint)vq | (vq_hi << 4));
    }
    if (lane == 0) {
        k_cache_s[sc_row] = half(kd);
        v_cache_s[sc_row] = half(vd);
    }
}

// Mixed-precision KV write: K as Q4 (D/2 nibbles), V as Q8 (D int8); both with
// per-32-block fp16 scales. Attention error is V-dominated (softmax averages V
// directly; K only perturbs scores, which the softmax normalizes away), so
// K-Q4 V-Q8 keeps near-Q8 coherence at ~0.41x the fp16 cache bytes vs Q8's
// 0.53x. K nibbles biased +8 like kv_cache_write_q4; V int8 like q8.
[[host_name("kv_cache_write_q4k_q8v")]]
[[kernel]]
void kv_cache_write_q4k_q8v(
    device const half*  new_k      [[buffer(0)]],
    device const half*  new_v      [[buffer(1)]],
    device       uchar* k_cache_q  [[buffer(2)]],   // (B,H_kv,cache_size,D/2) packed nibbles
    device       char*  v_cache_q  [[buffer(3)]],   // (B,H_kv,cache_size,D) int8
    device       half*  k_cache_s  [[buffer(4)]],
    device       half*  v_cache_s  [[buffer(5)]],
    constant uint& B               [[buffer(6)]],
    constant uint& H_kv            [[buffer(7)]],
    constant uint& D_head          [[buffer(8)]],
    constant uint& seq_in          [[buffer(9)]],
    constant uint& pos             [[buffer(10)]],
    constant uint& cache_size      [[buffer(11)]],
    uint3 gid  [[thread_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const uint nblk = D_head / 32u;
    const uint blk  = gid.x >> 5;
    const uint t_in = gid.y;
    const uint bh   = gid.z;
    if (blk >= nblk || t_in >= seq_in || bh >= B * H_kv) return;

    const uint buf_t = (pos + t_in) % cache_size;
    const size_t in_row    = ((size_t)bh * seq_in     + t_in)  * D_head      + blk * 32u;
    const size_t kout_row  = ((size_t)bh * cache_size + buf_t) * (D_head / 2) + blk * 16u;
    const size_t vout_row  = ((size_t)bh * cache_size + buf_t) * D_head       + blk * 32u;
    const size_t sc_row    = ((size_t)bh * cache_size + buf_t) * nblk         + blk;

    const float kf = float(new_k[in_row + lane]);
    const float vf = float(new_v[in_row + lane]);
    const float kd = simd_max(fabs(kf)) / 7.0f;
    const float vd = simd_max(fabs(vf)) / 127.0f;
    const float kinv = kd > 0.0f ? 1.0f / kd : 0.0f;
    const float vinv = vd > 0.0f ? 1.0f / vd : 0.0f;

    const int kq = clamp(int(round(kf * kinv)) + 8, 0, 15);
    const uint kq_hi = simd_shuffle_down((uint)kq, 1u);
    if ((lane & 1u) == 0u)
        k_cache_q[kout_row + (lane >> 1)] = uchar((uint)kq | (kq_hi << 4));

    v_cache_q[vout_row + lane] = (char)round(vf * vinv);

    if (lane == 0) {
        k_cache_s[sc_row] = half(kd);
        v_cache_s[sc_row] = half(vd);
    }
}

} // namespace meow::ops::kv_cache
