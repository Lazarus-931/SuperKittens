#include <metal_stdlib>
using namespace metal;

namespace meow::gemma4 {

[[host_name("gemma4_attn_local_d256")]]
[[kernel, max_total_threads_per_threadgroup(128)]]
void gemma4_attn_local_d256(
    device const half* Q          [[buffer(0)]],
    device const half* K          [[buffer(1)]],
    device const half* V          [[buffer(2)]],
    device half*       O          [[buffer(3)]],
    constant uint& q_seq          [[buffer(4)]],
    constant uint& kv_len         [[buffer(5)]],
    constant uint& nheads         [[buffer(6)]],
    constant uint& n_kv_heads     [[buffer(7)]],
    constant uint& window         [[buffer(8)]],
    constant uint& kv_buf_start   [[buffer(9)]],
    constant uint& cache_size     [[buffer(10)]],
    uint3 gid  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    constexpr uint D = 256, D4 = D / 4, Bc = 16, Br = 4, NT = 128;

    const uint head = gid.x, batch = gid.z;
    if (head >= nheads) return;

    const uint kv_head = head * n_kv_heads / nheads;
    const size_t q_off  = (size_t)(batch * nheads     + head)    * q_seq      * D;
    const size_t kv_off = (size_t)(batch * n_kv_heads + kv_head) * cache_size * D;

    threadgroup half4 k_smem[Bc * D4];
    threadgroup half4 v_smem[Bc * D4];

    // ============================ DECODE FAST PATH ============================
    // q_seq == 1: only one Q row, all 4 simdgroups would be wasted in baseline.
    // Use split-Bc: each simdgroup processes Bc/4 = 4 rows of K per tile,
    // maintains its own online softmax, then merges across simdgroups at end.
    if (q_seq == 1) {
        const float scale = 1.0f / sqrt(float(D));
        const float4 q_lo = float4(reinterpret_cast<const device half4*>(Q + q_off)[lane])      * scale;
        const float4 q_hi = float4(reinterpret_cast<const device half4*>(Q + q_off)[lane + 32]) * scale;

        const uint causal_extra = kv_len - 1u;
        const uint upper        = 1u + causal_extra;
        const uint lower        = (upper > window) ? (upper - window) : 0u;
        const uint start_tile   = lower / Bc;
        const uint end_tile     = upper / Bc;
        const uint partial_lim  = upper - end_tile * Bc;

        float m = -INFINITY, s = 0.0f;
        float4 acc_lo = float4(0.0f), acc_hi = float4(0.0f);

        const uint kl_base = lane;
        const uint kh_base = lane + 32;

        // Per-simd K-row span within Bc tile: simd s owns rows [s*4, s*4+4).
        const uint j_lo = simd * 4u;
        const uint j_hi = j_lo + 4u;

        for (uint t = start_tile; t < end_tile; ++t) {
            const uint c0 = t * Bc;
            for (uint i = lid; i < Bc * D4; i += NT) {
                const uint r = i / D4, d = i % D4;
                const uint col_log = c0 + r;
                const uint col_buf = (kv_buf_start + col_log) % cache_size;
                k_smem[i] = reinterpret_cast<const device half4*>(K + kv_off + (size_t)col_buf * D)[d];
                v_smem[i] = reinterpret_cast<const device half4*>(V + kv_off + (size_t)col_buf * D)[d];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint j = j_lo; j < j_hi; j += 2) {
                const uint o0 = j * D4, o1 = (j + 1) * D4;
                half4 k0_lo = k_smem[o0 + kl_base];
                half4 k0_hi = k_smem[o0 + kh_base];
                half4 k1_lo = k_smem[o1 + kl_base];
                half4 k1_hi = k_smem[o1 + kh_base];

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

        // Partial (ragged) tile: simd 0 handles it, others contribute nothing.
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

            if (simd == 0) {
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
            }
            threadgroup_barrier(mem_flags::mem_none);
        }

        // ───── Cross-simdgroup merge ─────
        // Stash (m, s) and acc per simdgroup into tg memory, then have every
        // lane recompute the merged result. acc layout per lane is identical
        // across simdgroups (lane → dims [lane*4..+3] for lo / hi).
        threadgroup float tg_ms[4][2];
        // Reuse k_smem (8 KB = 4096 half = 2048 float) for acc_lo (4 simd × 32
        // lane × 4 floats = 512 floats), and v_smem for acc_hi.
        threadgroup float* tg_acc_lo = reinterpret_cast<threadgroup float*>(k_smem);
        threadgroup float* tg_acc_hi = reinterpret_cast<threadgroup float*>(v_smem);

        if (lane == 0) { tg_ms[simd][0] = m; tg_ms[simd][1] = s; }
        const uint slot = (simd * 32u + lane) * 4u;
        tg_acc_lo[slot+0] = acc_lo.x; tg_acc_lo[slot+1] = acc_lo.y;
        tg_acc_lo[slot+2] = acc_lo.z; tg_acc_lo[slot+3] = acc_lo.w;
        tg_acc_hi[slot+0] = acc_hi.x; tg_acc_hi[slot+1] = acc_hi.y;
        tg_acc_hi[slot+2] = acc_hi.z; tg_acc_hi[slot+3] = acc_hi.w;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Only simd 0 produces output. Each lane merges 4 partial states.
        if (simd == 0) {
            float mg = -INFINITY;
            for (uint i = 0; i < 4u; ++i) mg = max(mg, tg_ms[i][0]);
            float sg = 0.0f;
            float4 al = float4(0.0f), ah = float4(0.0f);
            for (uint i = 0; i < 4u; ++i) {
                float mi = tg_ms[i][0];
                float si = tg_ms[i][1];
                float w  = (si > 0.0f) ? metal::fast::exp(mi - mg) : 0.0f;
                sg += si * w;
                const uint sl = (i * 32u + lane) * 4u;
                float4 ali = float4(tg_acc_lo[sl+0], tg_acc_lo[sl+1], tg_acc_lo[sl+2], tg_acc_lo[sl+3]);
                float4 ahi = float4(tg_acc_hi[sl+0], tg_acc_hi[sl+1], tg_acc_hi[sl+2], tg_acc_hi[sl+3]);
                al += w * ali;
                ah += w * ahi;
            }
            const float inv_s = sg > 0.0f ? 1.0f / sg : 0.0f;
            reinterpret_cast<device half4*>(O + q_off)[lane]      = half4(al * inv_s);
            reinterpret_cast<device half4*>(O + q_off)[lane + 32] = half4(ah * inv_s);
        }
        return;
    }

    // ============================ ORIGINAL PATH ===============================
    const uint q_row = gid.y * Br + simd;
    if (q_row >= q_seq) return;

    const float scale = 1.0f / sqrt(float(D));
    const float4 q_lo = float4(
        reinterpret_cast<const device half4*>(Q + q_off + (size_t)q_row * D)[lane])      * scale;
    const float4 q_hi = float4(
        reinterpret_cast<const device half4*>(Q + q_off + (size_t)q_row * D)[lane + 32]) * scale;

    float m = -INFINITY, s = 0.0f;
    float4 acc_lo = float4(0.0f), acc_hi = float4(0.0f);

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

        #pragma clang loop unroll(full)
        for (uint j = 0; j < Bc; j += 2) {
            const uint o0 = j * D4, o1 = (j + 1) * D4;
            half4 k0_lo = k_smem[o0 + kl_base];
            half4 k0_hi = k_smem[o0 + kh_base];
            half4 k1_lo = k_smem[o1 + kl_base];
            half4 k1_hi = k_smem[o1 + kh_base];

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

    const uint kv_head = head * n_kv_heads / nheads;
    const size_t q_off  = (size_t)(batch * nheads     + head)    * q_seq      * D;
    const size_t kv_off = (size_t)(batch * n_kv_heads + kv_head) * cache_size * D;

    threadgroup half4 k_smem[Bc * D4];   // 8 KB
    threadgroup half4 v_smem[Bc * D4];   // 8 KB

    // ============================ DECODE FAST PATH ============================
    if (q_seq == 1) {
        const float scale = 1.0f / sqrt(float(D));
        float4 q_chunk[4];
        for (uint c = 0; c < 4; ++c) {
            q_chunk[c] = float4(
                reinterpret_cast<const device half4*>(Q + q_off)[lane + c * 32]) * scale;
        }

        const uint causal_extra = kv_len - 1u;
        const uint upper        = 1u + causal_extra;
        const uint full_tiles   = upper / Bc;
        const uint partial_lim  = upper - full_tiles * Bc;

        // Per-simd K-row span within Bc=8 tile: simd s owns rows [s*2, s*2+2).
        const uint j_lo = simd * 2u;

        float m = -INFINITY, s = 0.0f;
        float4 acc[4] = { float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f) };

        for (uint t = 0; t < full_tiles; ++t) {
            const uint c0 = t * Bc;
            for (uint i = lid; i < Bc * D4; i += NT) {
                const uint r = i / D4, d = i % D4;
                const uint col_log = c0 + r;
                const uint col_buf = (kv_buf_start + col_log) % cache_size;
                k_smem[i] = reinterpret_cast<const device half4*>(K + kv_off + (size_t)col_buf * D)[d];
                v_smem[i] = reinterpret_cast<const device half4*>(V + kv_off + (size_t)col_buf * D)[d];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // 2 K rows per simdgroup, j_lo and j_lo+1.
            float p0 = 0.0f, p1 = 0.0f;
            for (uint c = 0; c < 4; ++c) {
                p0 += dot(q_chunk[c], float4(k_smem[(j_lo+0) * D4 + lane + c * 32]));
                p1 += dot(q_chunk[c], float4(k_smem[(j_lo+1) * D4 + lane + c * 32]));
            }
            float s0 = simd_sum(p0);
            float s1 = simd_sum(p1);

            float new_m = max(m, max(s0, s1));
            float alpha = metal::fast::exp(m  - new_m);
            float b0    = metal::fast::exp(s0 - new_m);
            float b1    = metal::fast::exp(s1 - new_m);
            m = new_m;
            s = s * alpha + b0 + b1;
            for (uint c = 0; c < 4; ++c) {
                acc[c] = acc[c] * alpha
                       + b0 * float4(v_smem[(j_lo+0) * D4 + lane + c * 32])
                       + b1 * float4(v_smem[(j_lo+1) * D4 + lane + c * 32]);
            }

            threadgroup_barrier(mem_flags::mem_none);
        }

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

            if (simd == 0) {
                for (uint j = 0; j < partial_lim; ++j) {
                    float p = 0.0f;
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
                    for (uint c = 0; c < 4; ++c) {
                        acc[c] *= alpha;
                        acc[c] += beta * float4(v_smem[j * D4 + lane + c * 32]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_none);
        }

        // Cross-simdgroup merge.
        threadgroup float tg_ms[4][2];
        // 4 simd × 32 lane × 16 floats = 2048 floats = 8 KB. Reuse k_smem (8 KB).
        threadgroup float* tg_acc = reinterpret_cast<threadgroup float*>(k_smem);

        if (lane == 0) { tg_ms[simd][0] = m; tg_ms[simd][1] = s; }
        for (uint c = 0; c < 4; ++c) {
            const uint sl = ((simd * 32u + lane) * 4u + c) * 4u;
            tg_acc[sl+0] = acc[c].x; tg_acc[sl+1] = acc[c].y;
            tg_acc[sl+2] = acc[c].z; tg_acc[sl+3] = acc[c].w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd == 0) {
            float mg = -INFINITY;
            for (uint i = 0; i < 4u; ++i) mg = max(mg, tg_ms[i][0]);
            float sg = 0.0f;
            float4 ao[4] = { float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f) };
            for (uint i = 0; i < 4u; ++i) {
                float mi = tg_ms[i][0];
                float si = tg_ms[i][1];
                float w  = (si > 0.0f) ? metal::fast::exp(mi - mg) : 0.0f;
                sg += si * w;
                for (uint c = 0; c < 4; ++c) {
                    const uint sl = ((i * 32u + lane) * 4u + c) * 4u;
                    float4 a = float4(tg_acc[sl+0], tg_acc[sl+1], tg_acc[sl+2], tg_acc[sl+3]);
                    ao[c] += w * a;
                }
            }
            const float inv_s = sg > 0.0f ? 1.0f / sg : 0.0f;
            for (uint c = 0; c < 4; ++c) {
                reinterpret_cast<device half4*>(O + q_off)[lane + c * 32] = half4(ao[c] * inv_s);
            }
        }
        return;
    }

    // ============================ ORIGINAL PATH ===============================
    const uint q_row = gid.y * Br + simd;
    if (q_row >= q_seq) return;

    const float scale = 1.0f / sqrt(float(D));
    float4 q_chunk[4];
    [[clang::unroll]]
    for (uint c = 0; c < 4; ++c) {
        q_chunk[c] = float4(
            reinterpret_cast<const device half4*>(Q + q_off + (size_t)q_row * D)[lane + c * 32]) * scale;
    }

    float m = -INFINITY, s = 0.0f;
    float4 acc[4] = { float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f) };

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
