//
//  attn.metal — unified flash attention, d=64 and d=128, Apple Silicon
//
//  d=64:  32 SIMDs × 32 lanes = 1024 threads; Bc=64; float2/lane; K+V 2×8 KB
//  d=128:  4 SIMDs × 32 lanes =  128 threads; Bc=32; float4/lane; K+V 2×8 KB
//

#include <metal_stdlib>
using namespace metal;

namespace meow::attn {

// ─── d=64 ─────────────────────────────────────────────────────────────────────
// 1024 threads, Br=32, Bc=64. Lane i → elems [i*2, i*2+1]. TG: 2×8 KB.

template<bool Causal>
[[kernel, max_total_threads_per_threadgroup(1024)]]
void fa_d64(
    device const half* Q   [[buffer(0)]],
    device const half* K   [[buffer(1)]],
    device const half* V   [[buffer(2)]],
    device half*       O   [[buffer(3)]],
    constant uint&   seq   [[buffer(4)]],
    constant uint& nheads  [[buffer(5)]],
    uint3 gid  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    constexpr uint D = 64, D2 = D / 2, Bc = 64, Br = 32, NT = 1024;

    const uint head = gid.x, batch = gid.z;
    if (head >= nheads) return;
    const uint q_row = gid.y * Br + simd;
    if (q_row >= seq) return;

    const size_t off = (size_t)(batch * nheads + head) * seq * D;

    threadgroup half2 k_smem[Bc * D2];  // 8 KB
    threadgroup half2 v_smem[Bc * D2];  // 8 KB

    const float scale = 1.0f / sqrt(float(D));
    const float2 q_reg = float2(
        reinterpret_cast<const device half2*>(Q + off + (size_t)q_row * D)[lane]) * scale;

    float m = -INFINITY, s = 0.0f;
    float2 acc = float2(0.0f);

    // Full tiles: lim=Bc (compile-time), no bounds check in load
    const uint full_tiles = Causal ? (q_row + 1u) / Bc : seq / Bc;
    const uint partial_lim = Causal ? (q_row + 1u) - full_tiles * Bc
                                    : seq - full_tiles * Bc;

    // ── full tiles ──────────────────────────────────────────────────
    for (uint t = 0; t < full_tiles; ++t) {
        const uint c0 = t * Bc;
        // Manually unrolled load (NT=1024, Bc*D2=2048 → exactly 2 iters/thread)
        {
            const uint i0 = lid, i1 = lid + NT;
            const uint r0 = i0 >> 5, d0 = i0 & 31, col0 = c0 + r0;
            const uint r1 = i1 >> 5, d1 = i1 & 31, col1 = c0 + r1;
            k_smem[i0] = reinterpret_cast<const device half2*>(K + off + (size_t)col0 * D)[d0];
            v_smem[i0] = reinterpret_cast<const device half2*>(V + off + (size_t)col0 * D)[d0];
            k_smem[i1] = reinterpret_cast<const device half2*>(K + off + (size_t)col1 * D)[d1];
            v_smem[i1] = reinterpret_cast<const device half2*>(V + off + (size_t)col1 * D)[d1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Paired inner loop, lim=Bc=64 → 32 pairs exactly, no tail
        [[clang::unroll(4)]]
        for (uint j = 0; j < Bc; j += 2) {
            half2 k0 = k_smem[j * D2 + lane], k1 = k_smem[(j+1) * D2 + lane];
            float s0 = simd_sum(dot(q_reg, float2(k0)));
            float s1 = simd_sum(dot(q_reg, float2(k1)));
            float new_m = max(m, max(s0, s1));
            float alpha = metal::fast::exp(m - new_m);
            float b0    = metal::fast::exp(s0 - new_m);
            float b1    = metal::fast::exp(s1 - new_m);
            m   = new_m;
            s   = fma(s, alpha, b0 + b1);
            acc *= alpha;
            acc += b0 * float2(v_smem[j * D2 + lane]);
            acc += b1 * float2(v_smem[(j+1) * D2 + lane]);
        }

        threadgroup_barrier(mem_flags::mem_none);
    }

    // ── partial tile (if any) ────────────────────────────────────────
    if (partial_lim > 0) {
        const uint c0 = full_tiles * Bc;
        for (uint i = lid; i < Bc * D2; i += NT) {
            const uint r = i >> 5, d = i & 31, col = c0 + r;
            if (col < seq) {
                k_smem[i] = reinterpret_cast<const device half2*>(K + off + (size_t)col * D)[d];
                v_smem[i] = reinterpret_cast<const device half2*>(V + off + (size_t)col * D)[d];
            } else {
                k_smem[i] = half2(0.0h);
                v_smem[i] = half2(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint j = 0;
        for (; j + 1 < partial_lim; j += 2) {
            half2 k0 = k_smem[j * D2 + lane], k1 = k_smem[(j+1) * D2 + lane];
            float s0 = simd_sum(dot(q_reg, float2(k0)));
            float s1 = simd_sum(dot(q_reg, float2(k1)));
            float new_m = max(m, max(s0, s1));
            float alpha = metal::fast::exp(m - new_m);
            float b0    = metal::fast::exp(s0 - new_m);
            float b1    = metal::fast::exp(s1 - new_m);
            m   = new_m;
            s   = fma(s, alpha, b0 + b1);
            acc *= alpha;
            acc += b0 * float2(v_smem[j * D2 + lane]);
            acc += b1 * float2(v_smem[(j+1) * D2 + lane]);
        }
        if (j < partial_lim) {
            float score = simd_sum(dot(q_reg, float2(k_smem[j * D2 + lane])));
            float new_m = max(m, score);
            float alpha = metal::fast::exp(m - new_m);
            float beta  = metal::fast::exp(score - new_m);
            m   = new_m;
            s   = fma(s, alpha, beta);
            acc *= alpha;
            acc += beta * float2(v_smem[j * D2 + lane]);
        }
        threadgroup_barrier(mem_flags::mem_none);
    }

    const float inv_s = s > 0.0f ? 1.0f / s : 0.0f;
    reinterpret_cast<device half2*>(O + off + (size_t)q_row * D)[lane] = half2(acc * inv_s);
}

// ─── d=128 ────────────────────────────────────────────────────────────────────
// 128 threads, Br=4, Bc=32. Lane i → elems [i*4, i*4+3]. TG: 2×8 KB.

template<bool Causal>
[[kernel, max_total_threads_per_threadgroup(128)]]
void fa_d128(
    device const half* Q      [[buffer(0)]],
    device const half* K      [[buffer(1)]],
    device const half* V      [[buffer(2)]],
    device half*       O      [[buffer(3)]],
    constant uint&    seq     [[buffer(4)]],
    constant uint& head_dim   [[buffer(5)]],   // unused; kept for attn.h ABI
    constant uint&   nheads   [[buffer(6)]],
    uint3 gid  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    constexpr uint D = 128, D4 = D / 4, Bc = 32, Br = 4, NT = 128;

    const uint head = gid.x, batch = gid.z;
    if (head >= nheads) return;
    const uint q_row = gid.y * Br + simd;
    if (q_row >= seq) return;

    const size_t off = (size_t)(batch * nheads + head) * seq * D;

    threadgroup half4 k_smem[Bc * D4];  // 8 KB
    threadgroup half4 v_smem[Bc * D4];  // 8 KB

    const float scale = 1.0f / sqrt(float(D));
    const float4 q_reg = float4(
        reinterpret_cast<const device half4*>(Q + off + (size_t)q_row * D)[lane]) * scale;

    float m = -INFINITY, s = 0.0f;
    float4 acc = float4(0.0f);

    const uint full_tiles = Causal ? (q_row + 1u) / Bc : seq / Bc;
    const uint partial_lim = Causal ? (q_row + 1u) - full_tiles * Bc
                                    : seq - full_tiles * Bc;

    // ── full tiles (lim=Bc=32, 16 pairs exactly) ────────────────────
    for (uint t = 0; t < full_tiles; ++t) {
        const uint c0 = t * Bc;
        // NT=128, Bc*D4=1024 → 8 iters/thread; no bounds check needed
        [[clang::unroll(8)]]
        for (uint i = lid; i < Bc * D4; i += NT) {
            const uint r = i / D4, d = i % D4, col = c0 + r;
            k_smem[i] = reinterpret_cast<const device half4*>(K + off + (size_t)col * D)[d];
            v_smem[i] = reinterpret_cast<const device half4*>(V + off + (size_t)col * D)[d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        [[clang::unroll(4)]]
        for (uint j = 0; j < Bc; j += 2) {
            half4 k0 = k_smem[j * D4 + lane], k1 = k_smem[(j+1) * D4 + lane];
            float s0 = simd_sum(dot(q_reg, float4(k0)));
            float s1 = simd_sum(dot(q_reg, float4(k1)));
            float new_m = max(m, max(s0, s1));
            float alpha = metal::fast::exp(m - new_m);
            float b0    = metal::fast::exp(s0 - new_m);
            float b1    = metal::fast::exp(s1 - new_m);
            m   = new_m;
            s   = fma(s, alpha, b0 + b1);
            acc *= alpha;
            acc += b0 * float4(v_smem[j * D4 + lane]);
            acc += b1 * float4(v_smem[(j+1) * D4 + lane]);
        }

        threadgroup_barrier(mem_flags::mem_none);
    }

    // ── partial tile (if any) ────────────────────────────────────────
    if (partial_lim > 0) {
        const uint c0 = full_tiles * Bc;
        for (uint i = lid; i < Bc * D4; i += NT) {
            const uint r = i / D4, d = i % D4, col = c0 + r;
            if (col < seq) {
                k_smem[i] = reinterpret_cast<const device half4*>(K + off + (size_t)col * D)[d];
                v_smem[i] = reinterpret_cast<const device half4*>(V + off + (size_t)col * D)[d];
            } else {
                k_smem[i] = half4(0.0h);
                v_smem[i] = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint j = 0;
        for (; j + 1 < partial_lim; j += 2) {
            half4 k0 = k_smem[j * D4 + lane], k1 = k_smem[(j+1) * D4 + lane];
            float s0 = simd_sum(dot(q_reg, float4(k0)));
            float s1 = simd_sum(dot(q_reg, float4(k1)));
            float new_m = max(m, max(s0, s1));
            float alpha = metal::fast::exp(m - new_m);
            float b0    = metal::fast::exp(s0 - new_m);
            float b1    = metal::fast::exp(s1 - new_m);
            m   = new_m;
            s   = fma(s, alpha, b0 + b1);
            acc *= alpha;
            acc += b0 * float4(v_smem[j * D4 + lane]);
            acc += b1 * float4(v_smem[(j+1) * D4 + lane]);
        }
        if (j < partial_lim) {
            float score = simd_sum(dot(q_reg, float4(k_smem[j * D4 + lane])));
            float new_m = max(m, score);
            float alpha = metal::fast::exp(m - new_m);
            float beta  = metal::fast::exp(score - new_m);
            m   = new_m;
            s   = fma(s, alpha, beta);
            acc *= alpha;
            acc += beta * float4(v_smem[j * D4 + lane]);
        }
        threadgroup_barrier(mem_flags::mem_none);
    }

    const float inv_s = s > 0.0f ? 1.0f / s : 0.0f;
    reinterpret_cast<device half4*>(O + off + (size_t)q_row * D)[lane] = half4(acc * inv_s);
}

// ─── instantiations ───────────────────────────────────────────────────────────

template [[host_name("fa_causal_64")]]
[[kernel]] void fa_d64<true>(
    device const half*, device const half*, device const half*, device half*,
    constant uint&, constant uint&, uint3, uint, uint, uint);

template [[host_name("fa_noncausal_64")]]
[[kernel]] void fa_d64<false>(
    device const half*, device const half*, device const half*, device half*,
    constant uint&, constant uint&, uint3, uint, uint, uint);

template [[host_name("mha_causal")]]
[[kernel]] void fa_d128<true>(
    device const half*, device const half*, device const half*, device half*,
    constant uint&, constant uint&, constant uint&, uint3, uint, uint, uint);

template [[host_name("mha_noncausal")]]
[[kernel]] void fa_d128<false>(
    device const half*, device const half*, device const half*, device half*,
    constant uint&, constant uint&, constant uint&, uint3, uint, uint, uint);

} // namespace meow::attn
