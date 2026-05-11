//
//  attn.metal — Gemma 4 attention (local sliding window + global extreme GQA)
//
//  Adapted from kernels/attn/attn.metal (d=64, d=128 paths). Two variants
//  here, matching Gemma 4's 5:1 local:global layer pattern:
//
//    LOCAL  (5 of 6 layers): head_dim=256, sliding window 4K tokens, GQA
//    GLOBAL (1 of 6 layers): head_dim=512, FULL causal, EXTREME GQA (4 KV heads)

#include <metal_stdlib>
using namespace metal;

namespace meow::gemma4 {

// ─── Local: d=256, sliding window, causal ─────────────────────────────────────
// 128 threads, Br=4, Bc=16. Lane i → elems [i*8, i*8+7]; held as 2 half4.

// Cache-aware attention. K, V are read from a (possibly circular) cache buffer
// of physical size `cache_size`, with the logical token at index 0 located at
// buffer position `kv_buf_start`. q_seq and kv_len decouple the Q-row count
// from the K/V-position count (so decode can be q_seq=1 over kv_len=cached_t).
//
// Backward-compat (prefill, no cache): pass q_seq=kv_len=seq, kv_buf_start=0,
// cache_size=seq — math reduces to the original sliding-window prefill.
[[host_name("gemma4_attn_local_d256")]]
[[kernel, max_total_threads_per_threadgroup(128)]]
void gemma4_attn_local_d256(
    device const half* Q          [[buffer(0)]],
    device const half* K          [[buffer(1)]],
    device const half* V          [[buffer(2)]],
    device half*       O          [[buffer(3)]],
    constant uint& q_seq          [[buffer(4)]],   // # Q rows
    constant uint& kv_len         [[buffer(5)]],   // # K/V positions to attend over
    constant uint& nheads         [[buffer(6)]],
    constant uint& n_kv_heads     [[buffer(7)]],
    constant uint& window         [[buffer(8)]],   // sliding window length
    constant uint& kv_buf_start   [[buffer(9)]],   // cache offset for logical K[0]
    constant uint& cache_size     [[buffer(10)]],  // physical K/V buffer dim
    uint3 gid  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    constexpr uint D = 256, D4 = D / 4, Bc = 16, Br = 4, NT = 128;

    const uint head = gid.x, batch = gid.z;
    if (head >= nheads) return;
    const uint q_row = gid.y * Br + simd;
    if (q_row >= q_seq) return;

    const uint kv_head = head * n_kv_heads / nheads;
    const size_t q_off  = (size_t)(batch * nheads     + head)    * q_seq      * D;
    const size_t kv_off = (size_t)(batch * n_kv_heads + kv_head) * cache_size * D;

    // 2× 8 KB threadgroup tiles. Each row of K/V is held as two half4 chunks.
    threadgroup half4 k_smem[Bc * D4];   // 8 KB
    threadgroup half4 v_smem[Bc * D4];   // 8 KB

    // Q register: 8 elements per lane, held as two float4 (lo / hi halves of D).
    const float scale = 1.0f / sqrt(float(D));
    const float4 q_lo = float4(
        reinterpret_cast<const device half4*>(Q + q_off + (size_t)q_row * D)[lane])      * scale;
    const float4 q_hi = float4(
        reinterpret_cast<const device half4*>(Q + q_off + (size_t)q_row * D)[lane + 32]) * scale;

    float m = -INFINITY, s = 0.0f;
    float4 acc_lo = float4(0.0f), acc_hi = float4(0.0f);

    // Causal mask in K-coords: k_idx <= q_row + (kv_len - q_seq).
    // Window mask: k_idx >= upper - window. (For prefill, kv_len = q_seq → standard causal.)
    const uint causal_extra = kv_len - q_seq;
    const uint upper        = q_row + 1u + causal_extra;
    const uint lower        = (upper > window) ? (upper - window) : 0u;
    const uint start_tile   = lower / Bc;
    const uint end_tile     = upper / Bc;
    const uint partial_lim  = upper - end_tile * Bc;

    const uint kl_base = lane;
    const uint kh_base = lane + 32;

    for (uint t = start_tile; t < end_tile; ++t) {
        const uint c0 = t * Bc;

        #pragma clang loop unroll(full)
        for (uint i = lid; i < Bc * D4; i += NT) {
            const uint r = i / D4, d = i % D4;
            const uint col_log = c0 + r;
            const uint col_buf = (kv_buf_start + col_log) % cache_size;
            k_smem[i] = reinterpret_cast<const device half4*>(K + kv_off + (size_t)col_buf * D)[d];
            v_smem[i] = reinterpret_cast<const device half4*>(V + kv_off + (size_t)col_buf * D)[d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Inner loop: 8 paired iterations covering Bc=16 K rows.
        // Each iteration computes dot products across both half-D regions.
        #pragma clang loop unroll(full)
        for (uint j = 0; j < Bc; j += 2) {
            const uint o0 = j * D4, o1 = (j + 1) * D4;
            half4 k0_lo = k_smem[o0 + kl_base];
            half4 k0_hi = k_smem[o0 + kh_base];
            half4 k1_lo = k_smem[o1 + kl_base];
            half4 k1_hi = k_smem[o1 + kh_base];

            // Per-lane partial dot, then simd_sum across lanes for the row total.
            float p0 = dot(q_lo, float4(k0_lo)) + dot(q_hi, float4(k0_hi));
            float p1 = dot(q_lo, float4(k1_lo)) + dot(q_hi, float4(k1_hi));
            float s0 = simd_sum(p0);
            float s1 = simd_sum(p1);

            float new_m = max(m, max(s0, s1));
            float alpha = metal::fast::exp(m  - new_m);
            float b0    = metal::fast::exp(s0 - new_m);
            float b1    = metal::fast::exp(s1 - new_m);
            m = new_m;
            s = fma(s, alpha, b0 + b1);
            acc_lo *= alpha; acc_hi *= alpha;
            acc_lo += b0 * float4(v_smem[o0 + kl_base]) + b1 * float4(v_smem[o1 + kl_base]);
            acc_hi += b0 * float4(v_smem[o0 + kh_base]) + b1 * float4(v_smem[o1 + kh_base]);
        }

        threadgroup_barrier(mem_flags::mem_none);
    }

    if (partial_lim > 0) {
        const uint c0 = end_tile * Bc;
        for (uint i = lid; i < Bc * D4; i += NT) {
            const uint r = i / D4, d = i % D4;
            const uint col_log = c0 + r;
            if (col_log < kv_len && col_log >= lower) {
                const uint col_buf = (kv_buf_start + col_log) % cache_size;
                k_smem[i] = reinterpret_cast<const device half4*>(K + kv_off + (size_t)col_buf * D)[d];
                v_smem[i] = reinterpret_cast<const device half4*>(V + kv_off + (size_t)col_buf * D)[d];
            } else {
                k_smem[i] = half4(0.0h);
                v_smem[i] = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = 0; j < partial_lim; ++j) {
            const uint o = j * D4;
            half4 kl = k_smem[o + kl_base], kh = k_smem[o + kh_base];
            float score = simd_sum(dot(q_lo, float4(kl)) + dot(q_hi, float4(kh)));
            float new_m = max(m, score);
            float alpha = metal::fast::exp(m     - new_m);
            float beta  = metal::fast::exp(score - new_m);
            m = new_m;
            s = fma(s, alpha, beta);
            acc_lo *= alpha; acc_hi *= alpha;
            acc_lo += beta * float4(v_smem[o + kl_base]);
            acc_hi += beta * float4(v_smem[o + kh_base]);
        }
        threadgroup_barrier(mem_flags::mem_none);
    }

    const float inv_s = s > 0.0f ? 1.0f / s : 0.0f;
    reinterpret_cast<device half4*>(O + q_off + (size_t)q_row * D)[lane]      = half4(acc_lo * inv_s);
    reinterpret_cast<device half4*>(O + q_off + (size_t)q_row * D)[lane + 32] = half4(acc_hi * inv_s);
}


// ─── Global: d=512, full causal, EXTREME GQA (e.g. 4 KV heads) ─────────────────
// 128 threads, Br=4, Bc=8. Lane i → elems [i*16, i*16+15]; held as 4 half4.

[[host_name("gemma4_attn_global_d512")]]
[[kernel, max_total_threads_per_threadgroup(128)]]
void gemma4_attn_global_d512(
    device const half* Q          [[buffer(0)]],
    device const half* K          [[buffer(1)]],
    device const half* V          [[buffer(2)]],
    device half*       O          [[buffer(3)]],
    constant uint& q_seq          [[buffer(4)]],
    constant uint& kv_len         [[buffer(5)]],
    constant uint& nheads         [[buffer(6)]],
    constant uint& n_kv_heads     [[buffer(7)]],
    constant uint& kv_buf_start   [[buffer(8)]],
    constant uint& cache_size     [[buffer(9)]],
    uint3 gid  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    constexpr uint D = 512, D4 = D / 4, Bc = 8, Br = 4, NT = 128;

    const uint head = gid.x, batch = gid.z;
    if (head >= nheads) return;
    const uint q_row = gid.y * Br + simd;
    if (q_row >= q_seq) return;

    const uint kv_head = head * n_kv_heads / nheads;
    const size_t q_off  = (size_t)(batch * nheads     + head)    * q_seq      * D;
    const size_t kv_off = (size_t)(batch * n_kv_heads + kv_head) * cache_size * D;

    // Bc=8 rows × D=512 × 2 bytes = 8 KB per buffer; 2 buffers = 16 KB total.
    threadgroup half4 k_smem[Bc * D4];   // 8 KB
    threadgroup half4 v_smem[Bc * D4];   // 8 KB

    // Q register: 16 elements per lane, held as four float4 chunks q[0..3].
    // chunk i covers dims [i*128 + lane*4 .. i*128 + lane*4+3].
    const float scale = 1.0f / sqrt(float(D));
    float4 q_chunk[4];
    [[clang::unroll]]
    for (uint c = 0; c < 4; ++c) {
        q_chunk[c] = float4(
            reinterpret_cast<const device half4*>(Q + q_off + (size_t)q_row * D)[lane + c * 32]) * scale;
    }

    float m = -INFINITY, s = 0.0f;
    float4 acc[4] = { float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f) };

    // Causal: k_idx <= q_row + (kv_len - q_seq).
    const uint causal_extra = kv_len - q_seq;
    const uint upper        = q_row + 1u + causal_extra;
    const uint full_tiles   = upper / Bc;
    const uint partial_lim  = upper - full_tiles * Bc;

    for (uint t = 0; t < full_tiles; ++t) {
        const uint c0 = t * Bc;

        [[clang::unroll(8)]]
        for (uint i = lid; i < Bc * D4; i += NT) {
            const uint r = i / D4, d = i % D4;
            const uint col_log = c0 + r;
            const uint col_buf = (kv_buf_start + col_log) % cache_size;
            k_smem[i] = reinterpret_cast<const device half4*>(K + kv_off + (size_t)col_buf * D)[d];
            v_smem[i] = reinterpret_cast<const device half4*>(V + kv_off + (size_t)col_buf * D)[d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Inner loop: 2 iterations of j+=4 covering Bc=8 K rows. Four K-rows
        // dot-producted per iteration with a fused running-max + alpha + accum
        // step. Wins +43% at seq=2048 vs the j+=2 form (j+=4 inner unroll
        // benchmark, see temp/gemma4_attn_global_opt/REPORT.md).
        [[clang::unroll]]
        for (uint j = 0; j < Bc; j += 4) {
            float p[4] = {0, 0, 0, 0};
            [[clang::unroll]]
            for (uint c = 0; c < 4; ++c) {
                p[0] += dot(q_chunk[c], float4(k_smem[(j+0) * D4 + lane + c * 32]));
                p[1] += dot(q_chunk[c], float4(k_smem[(j+1) * D4 + lane + c * 32]));
                p[2] += dot(q_chunk[c], float4(k_smem[(j+2) * D4 + lane + c * 32]));
                p[3] += dot(q_chunk[c], float4(k_smem[(j+3) * D4 + lane + c * 32]));
            }
            float ss[4];
            ss[0] = simd_sum(p[0]); ss[1] = simd_sum(p[1]);
            ss[2] = simd_sum(p[2]); ss[3] = simd_sum(p[3]);
            float new_m = max(m, max(max(ss[0], ss[1]), max(ss[2], ss[3])));
            float alpha = metal::fast::exp(m - new_m);
            float b[4];
            [[clang::unroll]] for (uint k = 0; k < 4; ++k) b[k] = metal::fast::exp(ss[k] - new_m);
            m = new_m;
            s = s * alpha + b[0] + b[1] + b[2] + b[3];
            [[clang::unroll]]
            for (uint c = 0; c < 4; ++c) {
                acc[c] = acc[c] * alpha
                       + b[0] * float4(v_smem[(j+0) * D4 + lane + c * 32])
                       + b[1] * float4(v_smem[(j+1) * D4 + lane + c * 32])
                       + b[2] * float4(v_smem[(j+2) * D4 + lane + c * 32])
                       + b[3] * float4(v_smem[(j+3) * D4 + lane + c * 32]);
            }
        }

        threadgroup_barrier(mem_flags::mem_none);
    }

    // ── partial (ragged tail) tile ──────────────────────────────────
    if (partial_lim > 0) {
        const uint c0 = full_tiles * Bc;
        for (uint i = lid; i < Bc * D4; i += NT) {
            const uint r = i / D4, d = i % D4;
            const uint col_log = c0 + r;
            if (col_log < kv_len) {
                const uint col_buf = (kv_buf_start + col_log) % cache_size;
                k_smem[i] = reinterpret_cast<const device half4*>(K + kv_off + (size_t)col_buf * D)[d];
                v_smem[i] = reinterpret_cast<const device half4*>(V + kv_off + (size_t)col_buf * D)[d];
            } else {
                k_smem[i] = half4(0.0h);
                v_smem[i] = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = 0; j < partial_lim; ++j) {
            float p = 0.0f;
            [[clang::unroll]]
            for (uint c = 0; c < 4; ++c) {
                half4 kc = k_smem[j * D4 + lane + c * 32];
                p += dot(q_chunk[c], float4(kc));
            }
            float score = simd_sum(p);
            float new_m = max(m, score);
            float alpha = metal::fast::exp(m     - new_m);
            float beta  = metal::fast::exp(score - new_m);
            m = new_m;
            s = fma(s, alpha, beta);
            [[clang::unroll]]
            for (uint c = 0; c < 4; ++c) {
                acc[c] *= alpha;
                acc[c] += beta * float4(v_smem[j * D4 + lane + c * 32]);
            }
        }
        threadgroup_barrier(mem_flags::mem_none);
    }

    const float inv_s = s > 0.0f ? 1.0f / s : 0.0f;
    [[clang::unroll]]
    for (uint c = 0; c < 4; ++c) {
        reinterpret_cast<device half4*>(O + q_off + (size_t)q_row * D)[lane + c * 32] =
            half4(acc[c] * inv_s);
    }
}


} // namespace meow::gemma4
