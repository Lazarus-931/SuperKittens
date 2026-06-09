// attn.metal — flash attention, d=64 and d=128 (GQA-aware).

#include <metal_stdlib>
using namespace metal;

namespace meow::attn {

// ── head_dim template machinery ───────────────────────────────────────────
// The decode/causal kernels use a 32-lane simdgroup where each lane owns
// VW = D/32 contiguous head-dim elements (D=128 → 4, D=64 → 2). vec_t<VW> maps
// that per-lane width to the matching half/float vector so the D=128
// instantiation generates the SAME half4/float4 code as the original
// hand-written kernel (byte-identical). dot()/simd_sum collapse the per-lane
// partials to the full-D score regardless of VW.
// Only VW ∈ {2,4} are wired: reinterpret_cast<half2/half4*>(p)[lane] strides
// exactly one head-dim block. half3 is 8-byte-padded in Metal, so VW=3 (D=96,
// Phi-3) can NOT use this vector indexing — it needs a scalar load path (TODO).
template<uint VW> struct vec_t;
template<> struct vec_t<2> { using h = half2; using f = float2; };
template<> struct vec_t<4> { using h = half4; using f = float4; };


template<bool Causal>
[[kernel, max_total_threads_per_threadgroup(1024)]]
void fa_d64(
    device const half* Q   [[buffer(0)]],
    device const half* K   [[buffer(1)]],
    device const half* V   [[buffer(2)]],
    device half*       O   [[buffer(3)]],
    constant uint&   seq   [[buffer(4)]],
    constant uint& nheads  [[buffer(5)]],
    constant uint& n_kv_heads [[buffer(6)]],
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

    const uint kv_head = head * n_kv_heads / nheads;
    const size_t q_off  = (size_t)(batch * nheads + head) * seq * D;
    const size_t kv_off = (size_t)(batch * n_kv_heads + kv_head) * seq * D;

    threadgroup half2 k_smem[Bc * D2];  // 8 KB
    threadgroup half2 v_smem[Bc * D2];  // 8 KB

    const float scale = 1.0f / sqrt(float(D));
    const float2 q_reg = float2(
        reinterpret_cast<const device half2*>(Q + q_off + (size_t)q_row * D)[lane]) * scale;

    float m = -INFINITY, s = 0.0f;
    float2 acc = float2(0.0f);

    const uint full_tiles = Causal ? (q_row + 1u) / Bc : seq / Bc;
    const uint partial_lim = Causal ? (q_row + 1u) - full_tiles * Bc
                                    : seq - full_tiles * Bc;

    for (uint t = 0; t < full_tiles; ++t) {
        const uint c0 = t * Bc;
        {
            const uint i0 = lid, i1 = lid + NT;
            const uint r0 = i0 >> 5, d0 = i0 & 31, col0 = c0 + r0;
            const uint r1 = i1 >> 5, d1 = i1 & 31, col1 = c0 + r1;
            k_smem[i0] = reinterpret_cast<const device half2*>(K + kv_off + (size_t)col0 * D)[d0];
            v_smem[i0] = reinterpret_cast<const device half2*>(V + kv_off + (size_t)col0 * D)[d0];
            k_smem[i1] = reinterpret_cast<const device half2*>(K + kv_off + (size_t)col1 * D)[d1];
            v_smem[i1] = reinterpret_cast<const device half2*>(V + kv_off + (size_t)col1 * D)[d1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

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

    if (partial_lim > 0) {
        const uint c0 = full_tiles * Bc;
        for (uint i = lid; i < Bc * D2; i += NT) {
            const uint r = i >> 5, d = i & 31, col = c0 + r;
            if (col < seq) {
                k_smem[i] = reinterpret_cast<const device half2*>(K + kv_off + (size_t)col * D)[d];
                v_smem[i] = reinterpret_cast<const device half2*>(V + kv_off + (size_t)col * D)[d];
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
    reinterpret_cast<device half2*>(O + q_off + (size_t)q_row * D)[lane] = half2(acc * inv_s);
}

// GQA: Hg = n_heads/n_kv_heads sibling Q-heads share one K/V tile per group.

template<bool Causal, uint D>
[[kernel, max_total_threads_per_threadgroup(1024)]]
void fa_dN(
    device const half* Q          [[buffer(0)]],
    device const half* K          [[buffer(1)]],
    device const half* V          [[buffer(2)]],
    device half*       O          [[buffer(3)]],
    constant uint&    seq         [[buffer(4)]],
    constant uint&   nheads       [[buffer(5)]],
    constant uint& n_kv_heads     [[buffer(6)]],
    constant uint& kv_len         [[buffer(7)]],
    constant uint& cache_stride   [[buffer(8)]],
    uint3 gid  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    // Each of the 32 simd lanes owns one VW-wide head-dim block (VW=D/32), so
    // 32×VW = D. NB = D/VW = 32 blocks per column (= lane count). smem stride is
    // NB; for D=128 (VW=4) this is the original half4[Bc*32] layout, byte-identical.
    constexpr uint VW = D / 32, NB = D / VW, Bc = 64, Br = 2;
    using hvec = typename vec_t<VW>::h;
    using fvec = typename vec_t<VW>::f;

    const uint kv_head = gid.x;
    const uint batch   = gid.z;
    const uint Hg      = nheads / n_kv_heads;
    const uint NT      = Hg * Br * 32u;

    const uint q_head_local = simd / Br;
    const uint r            = simd - q_head_local * Br;
    const uint head         = kv_head * Hg + q_head_local;
    const uint q_row        = gid.y * Br + r;
    const bool active       = (q_row < seq) && (head < nheads) && (kv_head < n_kv_heads);

    const size_t q_off  = (size_t)(batch * nheads     + head)    * seq          * D;
    const size_t kv_off = (size_t)(batch * n_kv_heads + kv_head) * cache_stride * D;

    threadgroup hvec k_smem[Bc * NB];
    threadgroup hvec v_smem[Bc * NB];

    const float scale = 1.0f / sqrt(float(D));
    fvec q_reg = fvec(0.0f);
    if (active) {
        q_reg = fvec(
            reinterpret_cast<const device hvec*>(Q + q_off + (size_t)q_row * D)[lane]) * scale;
    }

    float m = -INFINITY, s = 0.0f;
    fvec acc = fvec(0.0f);

    const uint q_pos = kv_len - seq + q_row;
    const uint total_lim = Causal ? q_pos + 1u : kv_len;

    // WHY: k_smem/v_smem are threadgroup-shared but each simdgroup owns a
    // different q_row → different total_lim. Gating the cooperative load and the
    // tile walk on the per-thread total_lim drops keys a sibling row (larger
    // total_lim, same TG) needs, and divergent full_tiles makes per-simdgroup
    // barrier counts differ (UB). Drive the walk by a threadgroup-uniform extent
    // (last active row's total_lim, clamped to seq-1) and re-impose each row's
    // causal cutoff per key column. For current_pos=0 prefill and seq=1 decode
    // tg_lim equals the driving row's total_lim, so those paths are unchanged.
    uint tg_last_row = gid.y * Br + (Br - 1u);
    if (tg_last_row >= seq) tg_last_row = seq - 1u;
    const uint tg_q_pos = kv_len - seq + tg_last_row;
    const uint tg_lim   = Causal ? tg_q_pos + 1u : kv_len;
    const uint full_tiles  = tg_lim / Bc;
    const uint partial_lim = tg_lim - full_tiles * Bc;

    for (uint t = 0; t < full_tiles; ++t) {
        const uint c0 = t * Bc;
        for (uint i = lid; i < Bc * NB; i += NT) {
            const uint rr = i / NB, dd = i % NB, col = c0 + rr;
            k_smem[i] = reinterpret_cast<const device hvec*>(K + kv_off + (size_t)col * D)[dd];
            v_smem[i] = reinterpret_cast<const device hvec*>(V + kv_off + (size_t)col * D)[dd];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = 0; j < Bc; j += 2) {
            const uint col0 = c0 + j, col1 = c0 + j + 1u;
            hvec k0 = k_smem[j * NB + lane], k1 = k_smem[(j+1) * NB + lane];
            float s0 = (col0 < total_lim) ? simd_sum(dot(q_reg, fvec(k0))) : -INFINITY;
            float s1 = (col1 < total_lim) ? simd_sum(dot(q_reg, fvec(k1))) : -INFINITY;
            float new_m = max(m, max(s0, s1));
            float alpha = metal::fast::exp(m - new_m);
            float b0    = metal::fast::exp(s0 - new_m);
            float b1    = metal::fast::exp(s1 - new_m);
            m   = new_m;
            s   = fma(s, alpha, b0 + b1);
            acc *= alpha;
            acc += b0 * fvec(v_smem[j * NB + lane]);
            acc += b1 * fvec(v_smem[(j+1) * NB + lane]);
        }

        threadgroup_barrier(mem_flags::mem_none);
    }

    if (partial_lim > 0) {
        const uint c0 = full_tiles * Bc;
        for (uint i = lid; i < Bc * NB; i += NT) {
            const uint rr = i / NB, dd = i % NB, col = c0 + rr;
            if (col < tg_lim) {
                k_smem[i] = reinterpret_cast<const device hvec*>(K + kv_off + (size_t)col * D)[dd];
                v_smem[i] = reinterpret_cast<const device hvec*>(V + kv_off + (size_t)col * D)[dd];
            } else {
                k_smem[i] = hvec(0.0h);
                v_smem[i] = hvec(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint j = 0;
        for (; j + 1 < partial_lim; j += 2) {
            const uint col0 = c0 + j, col1 = c0 + j + 1u;
            hvec k0 = k_smem[j * NB + lane], k1 = k_smem[(j+1) * NB + lane];
            float s0 = (col0 < total_lim) ? simd_sum(dot(q_reg, fvec(k0))) : -INFINITY;
            float s1 = (col1 < total_lim) ? simd_sum(dot(q_reg, fvec(k1))) : -INFINITY;
            float new_m = max(m, max(s0, s1));
            float alpha = metal::fast::exp(m - new_m);
            float b0    = metal::fast::exp(s0 - new_m);
            float b1    = metal::fast::exp(s1 - new_m);
            m   = new_m;
            s   = fma(s, alpha, b0 + b1);
            acc *= alpha;
            acc += b0 * fvec(v_smem[j * NB + lane]);
            acc += b1 * fvec(v_smem[(j+1) * NB + lane]);
        }
        if (j < partial_lim) {
            const uint col0 = c0 + j;
            float score = (col0 < total_lim) ? simd_sum(dot(q_reg, fvec(k_smem[j * NB + lane])))
                                             : -INFINITY;
            float new_m = max(m, score);
            float alpha = metal::fast::exp(m - new_m);
            float beta  = metal::fast::exp(score - new_m);
            m   = new_m;
            s   = fma(s, alpha, beta);
            acc *= alpha;
            acc += beta * fvec(v_smem[j * NB + lane]);
        }
        threadgroup_barrier(mem_flags::mem_none);
    }

    if (active) {
        const float inv_s = s > 0.0f ? 1.0f / s : 0.0f;
        reinterpret_cast<device hvec*>(O + q_off + (size_t)q_row * D)[lane] = hvec(acc * inv_s);
    }
}


// ── D=64 decode-only (seq==1) attention ───────────────────────────────────
// WHY: fa_dN<…,64> reuses the Br=2 prefill layout at decode too (NT=Hg*Br*32),
// so at seq==1 the r==1 simdgroups (half the threadgroup) compute nothing —
// their q_row=1 is out of range. A naive Br=1 (one simdgroup/head, no smem)
// loses the cooperative K/V HBM amortisation and regresses ~0.5-0.7x. This
// keeps the identical Bc=64 smem staging + Br=2 math loop but with Br=1: Hg
// simdgroups (Hg*32 threads) do BOTH the cooperative load and the compute, so
// no lane is idle. Output is bit-identical to fa_dN<true,64> at seq==1 (same
// tile walk, same fp16 reductions). Gated head_dim==64 && seq==1 in the host;
// the D=128 path and the seq>1 prefill path stay on fa_dN, byte-identical.
template<uint D>
[[kernel, max_total_threads_per_threadgroup(256)]]
void mha_decode_dN_br1(
    device const half* Q          [[buffer(0)]],
    device const half* K          [[buffer(1)]],
    device const half* V          [[buffer(2)]],
    device half*       O          [[buffer(3)]],
    constant uint&    seq         [[buffer(4)]],
    constant uint&   nheads       [[buffer(5)]],
    constant uint& n_kv_heads     [[buffer(6)]],
    constant uint& kv_len         [[buffer(7)]],
    constant uint& cache_stride   [[buffer(8)]],
    uint3 gid  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    constexpr uint VW = D / 32, NB = D / VW, Bc = 64;
    using hvec = typename vec_t<VW>::h;
    using fvec = typename vec_t<VW>::f;

    const uint kv_head = gid.x;
    const uint batch   = gid.z;
    const uint Hg      = nheads / n_kv_heads;
    const uint NT      = Hg * 32u;
    const uint head    = kv_head * Hg + simd;
    const bool active  = (simd < Hg) && (head < nheads) && (kv_head < n_kv_heads);

    const size_t q_off  = (size_t)(batch * nheads     + head)    * seq          * D;
    const size_t kv_off = (size_t)(batch * n_kv_heads + kv_head) * cache_stride * D;

    threadgroup hvec k_smem[Bc * NB];
    threadgroup hvec v_smem[Bc * NB];

    const float scale = 1.0f / sqrt(float(D));
    fvec q_reg = active
        ? fvec(reinterpret_cast<const device hvec*>(Q + q_off)[lane]) * scale
        : fvec(0.0f);

    float m = -INFINITY, s = 0.0f;
    fvec  acc = fvec(0.0f);

    const uint full_tiles  = kv_len / Bc;
    const uint partial_lim = kv_len - full_tiles * Bc;

    for (uint t = 0; t < full_tiles; ++t) {
        const uint c0 = t * Bc;
        for (uint i = lid; i < Bc * NB; i += NT) {
            const uint rr = i / NB, dd = i % NB, col = c0 + rr;
            k_smem[i] = reinterpret_cast<const device hvec*>(K + kv_off + (size_t)col * D)[dd];
            v_smem[i] = reinterpret_cast<const device hvec*>(V + kv_off + (size_t)col * D)[dd];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = 0; j < Bc; j += 2) {
            hvec k0 = k_smem[j * NB + lane], k1 = k_smem[(j+1) * NB + lane];
            float s0 = simd_sum(dot(q_reg, fvec(k0)));
            float s1 = simd_sum(dot(q_reg, fvec(k1)));
            float new_m = max(m, max(s0, s1));
            float alpha = metal::fast::exp(m - new_m);
            float b0    = metal::fast::exp(s0 - new_m);
            float b1    = metal::fast::exp(s1 - new_m);
            m   = new_m;
            s   = fma(s, alpha, b0 + b1);
            acc *= alpha;
            acc += b0 * fvec(v_smem[j * NB + lane]);
            acc += b1 * fvec(v_smem[(j+1) * NB + lane]);
        }
        threadgroup_barrier(mem_flags::mem_none);
    }

    if (partial_lim > 0) {
        const uint c0 = full_tiles * Bc;
        for (uint i = lid; i < Bc * NB; i += NT) {
            const uint rr = i / NB, dd = i % NB, col = c0 + rr;
            if (col < kv_len) {
                k_smem[i] = reinterpret_cast<const device hvec*>(K + kv_off + (size_t)col * D)[dd];
                v_smem[i] = reinterpret_cast<const device hvec*>(V + kv_off + (size_t)col * D)[dd];
            } else {
                k_smem[i] = hvec(0.0h);
                v_smem[i] = hvec(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint j = 0;
        for (; j + 1 < partial_lim; j += 2) {
            hvec k0 = k_smem[j * NB + lane], k1 = k_smem[(j+1) * NB + lane];
            float s0 = simd_sum(dot(q_reg, fvec(k0)));
            float s1 = simd_sum(dot(q_reg, fvec(k1)));
            float new_m = max(m, max(s0, s1));
            float alpha = metal::fast::exp(m - new_m);
            float b0    = metal::fast::exp(s0 - new_m);
            float b1    = metal::fast::exp(s1 - new_m);
            m   = new_m;
            s   = fma(s, alpha, b0 + b1);
            acc *= alpha;
            acc += b0 * fvec(v_smem[j * NB + lane]);
            acc += b1 * fvec(v_smem[(j+1) * NB + lane]);
        }
        if (j < partial_lim) {
            float score = simd_sum(dot(q_reg, fvec(k_smem[j * NB + lane])));
            float new_m = max(m, score);
            float alpha = metal::fast::exp(m - new_m);
            float beta  = metal::fast::exp(score - new_m);
            m   = new_m;
            s   = fma(s, alpha, beta);
            acc *= alpha;
            acc += beta * fvec(v_smem[j * NB + lane]);
        }
        threadgroup_barrier(mem_flags::mem_none);
    }

    if (active) {
        const float inv_s = s > 0.0f ? 1.0f / s : 0.0f;
        reinterpret_cast<device hvec*>(O + q_off)[lane] = hvec(acc * inv_s);
    }
}


// ── Prefill-specialized attention (seq>1) ─────────────────────────────────
// Same flash math + GQA design as fa_d128, but BR query rows per threadgroup
// (vs Br=2 in the decode-tuned fa_d128). BR is the K/V-tile reuse factor: each
// Bc=64 tile is staged in smem once per TG and reused across Hg*BR simdgroups,
// so the cooperative K/V HBM read is amortised over BR rows instead of 2. At
// long-seq prefill mha_causal re-streams the full causal K/V seq/Br times
// (O(seq^2) HBM); BR=8 cuts that traffic ~4x. NT = Hg*BR*32 must stay <=1024,
// so the host gates this kernel on Hg*BR*32<=1024 (else falls back to fa_d128).
// The per-key math (simd_sum dot, online softmax) is bit-identical to fa_d128;
// only the threadgroup row-fan-out + tile reuse differ. Decode (seq=1) NEVER
// dispatches this — it stays on fa_d128 (mha_causal), byte-identical.
template<bool Causal, uint BR>
[[kernel, max_total_threads_per_threadgroup(1024)]]
void fa_d128_prefill(
    device const half* Q          [[buffer(0)]],
    device const half* K          [[buffer(1)]],
    device const half* V          [[buffer(2)]],
    device half*       O          [[buffer(3)]],
    constant uint&    seq         [[buffer(4)]],
    constant uint&   nheads       [[buffer(5)]],
    constant uint& n_kv_heads     [[buffer(6)]],
    constant uint& kv_len         [[buffer(7)]],
    constant uint& cache_stride   [[buffer(8)]],
    uint3 gid  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    constexpr uint D = 128, D4 = D / 4, Bc = 64;

    const uint kv_head = gid.x;
    const uint batch   = gid.z;
    const uint Hg      = nheads / n_kv_heads;
    const uint NT      = Hg * BR * 32u;

    const uint q_head_local = simd / BR;
    const uint r            = simd - q_head_local * BR;
    const uint head         = kv_head * Hg + q_head_local;
    const uint q_row        = gid.y * BR + r;
    const bool active       = (q_row < seq) && (head < nheads) && (kv_head < n_kv_heads);

    const size_t q_off  = (size_t)(batch * nheads     + head)    * seq          * D;
    const size_t kv_off = (size_t)(batch * n_kv_heads + kv_head) * cache_stride * D;

    threadgroup half4 k_smem[Bc * D4];
    threadgroup half4 v_smem[Bc * D4];

    const float scale = 1.0f / sqrt(float(D));
    float4 q_reg = float4(0.0f);
    if (active) {
        q_reg = float4(
            reinterpret_cast<const device half4*>(Q + q_off + (size_t)q_row * D)[lane]) * scale;
    }

    float m = -INFINITY, s = 0.0f;
    float4 acc = float4(0.0f);

    const uint q_pos = kv_len - seq + q_row;
    const uint total_lim = Causal ? q_pos + 1u : kv_len;

    // Threadgroup-uniform tile walk (last active row's causal extent), with each
    // row's own cutoff re-imposed per key column — identical contract to fa_d128.
    uint tg_last_row = gid.y * BR + (BR - 1u);
    if (tg_last_row >= seq) tg_last_row = seq - 1u;
    const uint tg_q_pos = kv_len - seq + tg_last_row;
    const uint tg_lim   = Causal ? tg_q_pos + 1u : kv_len;
    const uint full_tiles  = tg_lim / Bc;
    const uint partial_lim = tg_lim - full_tiles * Bc;

    for (uint t = 0; t < full_tiles; ++t) {
        const uint c0 = t * Bc;
        for (uint i = lid; i < Bc * D4; i += NT) {
            const uint rr = i / D4, dd = i % D4, col = c0 + rr;
            k_smem[i] = reinterpret_cast<const device half4*>(K + kv_off + (size_t)col * D)[dd];
            v_smem[i] = reinterpret_cast<const device half4*>(V + kv_off + (size_t)col * D)[dd];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = 0; j < Bc; j += 2) {
            const uint col0 = c0 + j, col1 = c0 + j + 1u;
            half4 k0 = k_smem[j * D4 + lane], k1 = k_smem[(j+1) * D4 + lane];
            float s0 = (col0 < total_lim) ? simd_sum(dot(q_reg, float4(k0))) : -INFINITY;
            float s1 = (col1 < total_lim) ? simd_sum(dot(q_reg, float4(k1))) : -INFINITY;
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

    if (partial_lim > 0) {
        const uint c0 = full_tiles * Bc;
        for (uint i = lid; i < Bc * D4; i += NT) {
            const uint rr = i / D4, dd = i % D4, col = c0 + rr;
            if (col < tg_lim) {
                k_smem[i] = reinterpret_cast<const device half4*>(K + kv_off + (size_t)col * D)[dd];
                v_smem[i] = reinterpret_cast<const device half4*>(V + kv_off + (size_t)col * D)[dd];
            } else {
                k_smem[i] = half4(0.0h);
                v_smem[i] = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint j = 0;
        for (; j + 1 < partial_lim; j += 2) {
            const uint col0 = c0 + j, col1 = c0 + j + 1u;
            half4 k0 = k_smem[j * D4 + lane], k1 = k_smem[(j+1) * D4 + lane];
            float s0 = (col0 < total_lim) ? simd_sum(dot(q_reg, float4(k0))) : -INFINITY;
            float s1 = (col1 < total_lim) ? simd_sum(dot(q_reg, float4(k1))) : -INFINITY;
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
            const uint col0 = c0 + j;
            float score = (col0 < total_lim) ? simd_sum(dot(q_reg, float4(k_smem[j * D4 + lane])))
                                             : -INFINITY;
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

    if (active) {
        const float inv_s = s > 0.0f ? 1.0f / s : 0.0f;
        reinterpret_cast<device half4*>(O + q_off + (size_t)q_row * D)[lane] = half4(acc * inv_s);
    }
}


// ── Q8_0 KV-cache attention ───────────────────────────────────────────────
// K/V cached as Q8_0: int8 qs (…,kv,D) + fp16 scales (…,kv,D/32). Dequant
// happens during the cooperative smem load, so the math loops below are
// byte-identical to the fp16 fa_d128 / mha_decode_split. D=128 → 4 blocks/row,
// 8 half4-lanes per block; lane d4∈[0,32) reads block d4/8's scale.

// Load+dequant one cached row (col) into a half4 for a given d4 index.
static inline half4 deq_kv_q8(device const char* QS, device const half* SC,
                              size_t col, uint d4) {
    constexpr uint D = 128;
    const half  sc = SC[col * (D / 32u) + (d4 >> 3)];
    device const char* p = QS + col * D + (size_t)d4 * 4u;
    return half4(half(p[0]), half(p[1]), half(p[2]), half(p[3])) * sc;
}

template<bool Causal>
[[kernel, max_total_threads_per_threadgroup(1024)]]
void fa_d128_q8(
    device const half* Q          [[buffer(0)]],
    device const char* KQ         [[buffer(1)]],
    device const char* VQ         [[buffer(2)]],
    device const half* KS         [[buffer(3)]],
    device const half* VS         [[buffer(4)]],
    device half*       O          [[buffer(5)]],
    constant uint&    seq         [[buffer(6)]],
    constant uint&   nheads       [[buffer(7)]],
    constant uint& n_kv_heads     [[buffer(8)]],
    constant uint& kv_len         [[buffer(9)]],
    constant uint& cache_stride   [[buffer(10)]],
    uint3 gid  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    constexpr uint D = 128, D4 = D / 4, Bc = 64, Br = 2;

    const uint kv_head = gid.x;
    const uint batch   = gid.z;
    const uint Hg      = nheads / n_kv_heads;
    const uint NT      = Hg * Br * 32u;

    const uint q_head_local = simd / Br;
    const uint r            = simd - q_head_local * Br;
    const uint head         = kv_head * Hg + q_head_local;
    const uint q_row        = gid.y * Br + r;
    const bool active       = (q_row < seq) && (head < nheads) && (kv_head < n_kv_heads);

    const size_t q_off  = (size_t)(batch * nheads     + head)    * seq          * D;
    const size_t kv_q   = (size_t)(batch * n_kv_heads + kv_head) * cache_stride * D;
    const size_t kv_s   = (size_t)(batch * n_kv_heads + kv_head) * cache_stride * (D / 32u);
    device const char* KQp = KQ + kv_q;
    device const char* VQp = VQ + kv_q;
    device const half* KSp = KS + kv_s;
    device const half* VSp = VS + kv_s;

    threadgroup half4 k_smem[Bc * D4];
    threadgroup half4 v_smem[Bc * D4];

    const float scale = 1.0f / sqrt(float(D));
    float4 q_reg = float4(0.0f);
    if (active) {
        q_reg = float4(
            reinterpret_cast<const device half4*>(Q + q_off + (size_t)q_row * D)[lane]) * scale;
    }

    float m = -INFINITY, s = 0.0f;
    float4 acc = float4(0.0f);

    const uint q_pos = kv_len - seq + q_row;
    const uint total_lim = Causal ? q_pos + 1u : kv_len;

    uint tg_last_row = gid.y * Br + (Br - 1u);
    if (tg_last_row >= seq) tg_last_row = seq - 1u;
    const uint tg_q_pos = kv_len - seq + tg_last_row;
    const uint tg_lim   = Causal ? tg_q_pos + 1u : kv_len;
    const uint full_tiles  = tg_lim / Bc;
    const uint partial_lim = tg_lim - full_tiles * Bc;

    for (uint t = 0; t < full_tiles; ++t) {
        const uint c0 = t * Bc;
        for (uint i = lid; i < Bc * D4; i += NT) {
            const uint rr = i / D4, dd = i % D4, col = c0 + rr;
            k_smem[i] = deq_kv_q8(KQp, KSp, col, dd);
            v_smem[i] = deq_kv_q8(VQp, VSp, col, dd);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = 0; j < Bc; j += 2) {
            const uint col0 = c0 + j, col1 = c0 + j + 1u;
            half4 k0 = k_smem[j * D4 + lane], k1 = k_smem[(j+1) * D4 + lane];
            float s0 = (col0 < total_lim) ? simd_sum(dot(q_reg, float4(k0))) : -INFINITY;
            float s1 = (col1 < total_lim) ? simd_sum(dot(q_reg, float4(k1))) : -INFINITY;
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

    if (partial_lim > 0) {
        const uint c0 = full_tiles * Bc;
        for (uint i = lid; i < Bc * D4; i += NT) {
            const uint rr = i / D4, dd = i % D4, col = c0 + rr;
            if (col < tg_lim) {
                k_smem[i] = deq_kv_q8(KQp, KSp, col, dd);
                v_smem[i] = deq_kv_q8(VQp, VSp, col, dd);
            } else {
                k_smem[i] = half4(0.0h);
                v_smem[i] = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint j = 0;
        for (; j + 1 < partial_lim; j += 2) {
            const uint col0 = c0 + j, col1 = c0 + j + 1u;
            half4 k0 = k_smem[j * D4 + lane], k1 = k_smem[(j+1) * D4 + lane];
            float s0 = (col0 < total_lim) ? simd_sum(dot(q_reg, float4(k0))) : -INFINITY;
            float s1 = (col1 < total_lim) ? simd_sum(dot(q_reg, float4(k1))) : -INFINITY;
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
            const uint col0 = c0 + j;
            float score = (col0 < total_lim) ? simd_sum(dot(q_reg, float4(k_smem[j * D4 + lane])))
                                             : -INFINITY;
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

    if (active) {
        const float inv_s = s > 0.0f ? 1.0f / s : 0.0f;
        reinterpret_cast<device half4*>(O + q_off + (size_t)q_row * D)[lane] = half4(acc * inv_s);
    }
}


template<uint MAX_HG, uint NS, uint SPLITS, uint D>
[[kernel, max_total_threads_per_threadgroup(NS * 32)]]
void mha_decode_split_t(
    device const half* Q          [[buffer(0)]],
    device const half* K          [[buffer(1)]],
    device const half* V          [[buffer(2)]],
    device float*      PM         [[buffer(3)]],
    device float*      PS         [[buffer(4)]],
    device float*      PO         [[buffer(5)]],
    constant uint&    seq         [[buffer(6)]],
    constant uint&   nheads       [[buffer(7)]],
    constant uint& n_kv_heads     [[buffer(8)]],
    constant uint& kv_len         [[buffer(9)]],
    constant uint& cache_stride   [[buffer(10)]],
    constant uint& n_splits       [[buffer(11)]],
    uint3 gid  [[threadgroup_position_in_grid]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    constexpr uint VW = D / 32;  // per-lane head-dim block width (D=128→float4)
    using hvec = typename vec_t<VW>::h;
    using fvec = typename vec_t<VW>::f;

    const uint kv_head = gid.x;
    const uint split   = gid.y;
    const uint batch   = gid.z;
    const uint Hg      = nheads / n_kv_heads;

    // Even kv chunk per split (ceil); last split may be short or empty.
    const uint chunk = (kv_len + n_splits - 1u) / n_splits;
    const uint c_lo  = split * chunk;
    const uint c_hi  = min(c_lo + chunk, kv_len);

    const size_t kv_off = (size_t)(batch * n_kv_heads + kv_head) * cache_stride * D;
    const float scale = 1.0f / sqrt(float(D));

    fvec q_reg[MAX_HG];
    for (uint g = 0; g < MAX_HG; ++g) {
        if (g < Hg) {
            const uint head = kv_head * Hg + g;
            const size_t q_off = (size_t)(batch * nheads + head) * seq * D;
            q_reg[g] = fvec(
                reinterpret_cast<const device hvec*>(Q + q_off)[lane]) * scale;
        } else {
            q_reg[g] = fvec(0.0f);
        }
    }

    float m_st[MAX_HG], s_st[MAX_HG];
    fvec  acc[MAX_HG];
    for (uint g = 0; g < MAX_HG; ++g) {
        m_st[g] = -INFINITY; s_st[g] = 0.0f; acc[g] = fvec(0.0f);
    }

    // NS simdgroups split this chunk further; barrier-free key loop.
    for (uint col = c_lo + simd; col < c_hi; col += NS) {
        const fvec kf = fvec(reinterpret_cast<const device hvec*>(K + kv_off + (size_t)col * D)[lane]);
        const fvec vf = fvec(reinterpret_cast<const device hvec*>(V + kv_off + (size_t)col * D)[lane]);
        for (uint g = 0; g < Hg; ++g) {
            const float score = simd_sum(dot(q_reg[g], kf));
            const float new_m = max(m_st[g], score);
            const float alpha = metal::fast::exp(m_st[g] - new_m);
            const float beta  = metal::fast::exp(score   - new_m);
            m_st[g] = new_m;
            s_st[g] = fma(s_st[g], alpha, beta);
            acc[g]  = fma(acc[g], alpha, beta * vf);
        }
    }

    threadgroup float m_sh[MAX_HG * NS];
    threadgroup float s_sh[MAX_HG * NS];
    threadgroup fvec  a_sh[MAX_HG * NS * 32];
    for (uint g = 0; g < Hg; ++g) {
        if (lane == 0) {
            m_sh[g * NS + simd] = m_st[g];
            s_sh[g * NS + simd] = s_st[g];
        }
        a_sh[(g * NS + simd) * 32 + lane] = acc[g];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd == 0) {
        for (uint g = 0; g < Hg; ++g) {
            float gm = -INFINITY;
            for (uint p = 0; p < NS; ++p) gm = max(gm, m_sh[g * NS + p]);
            float gs = 0.0f;
            fvec ga = fvec(0.0f);
            for (uint p = 0; p < NS; ++p) {
                const float w = metal::fast::exp(m_sh[g * NS + p] - gm);
                gs += s_sh[g * NS + p] * w;
                ga = fma(a_sh[(g * NS + p) * 32 + lane], w, ga);
            }
            const uint head = kv_head * Hg + g;
            const uint idx  = (batch * nheads + head) * SPLITS + split;
            if (lane == 0) { PM[idx] = gm; PS[idx] = gs; }
            reinterpret_cast<device fvec*>(PO + (size_t)idx * D)[lane] = ga;
        }
    }
}

// Q8_0-KV split-K decode. Identical reduction to mha_decode_split_t; the only
// change is per-key K/V come from int8 qs + fp16 scales (see deq_kv_q8).
template<uint MAX_HG, uint NS, uint SPLITS>
[[kernel, max_total_threads_per_threadgroup(NS * 32)]]
void mha_decode_split_q8_t(
    device const half* Q          [[buffer(0)]],
    device const char* KQ         [[buffer(1)]],
    device const char* VQ         [[buffer(2)]],
    device const half* KS         [[buffer(3)]],
    device const half* VS         [[buffer(4)]],
    device float*      PM         [[buffer(5)]],
    device float*      PS         [[buffer(6)]],
    device float*      PO         [[buffer(7)]],
    constant uint&    seq         [[buffer(8)]],
    constant uint&   nheads       [[buffer(9)]],
    constant uint& n_kv_heads     [[buffer(10)]],
    constant uint& kv_len         [[buffer(11)]],
    constant uint& cache_stride   [[buffer(12)]],
    constant uint& n_splits       [[buffer(13)]],
    uint3 gid  [[threadgroup_position_in_grid]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    constexpr uint D = 128;

    const uint kv_head = gid.x;
    const uint split   = gid.y;
    const uint batch   = gid.z;
    const uint Hg      = nheads / n_kv_heads;

    const uint chunk = (kv_len + n_splits - 1u) / n_splits;
    const uint c_lo  = split * chunk;
    const uint c_hi  = min(c_lo + chunk, kv_len);

    const size_t kv_q = (size_t)(batch * n_kv_heads + kv_head) * cache_stride * D;
    const size_t kv_s = (size_t)(batch * n_kv_heads + kv_head) * cache_stride * (D / 32u);
    device const char* KQp = KQ + kv_q;
    device const char* VQp = VQ + kv_q;
    device const half* KSp = KS + kv_s;
    device const half* VSp = VS + kv_s;
    const float scale = 1.0f / sqrt(float(D));

    float4 q_reg[MAX_HG];
    for (uint g = 0; g < MAX_HG; ++g) {
        if (g < Hg) {
            const uint head = kv_head * Hg + g;
            const size_t q_off = (size_t)(batch * nheads + head) * seq * D;
            q_reg[g] = float4(
                reinterpret_cast<const device half4*>(Q + q_off)[lane]) * scale;
        } else {
            q_reg[g] = float4(0.0f);
        }
    }

    float  m_st[MAX_HG], s_st[MAX_HG];
    float4 acc[MAX_HG];
    for (uint g = 0; g < MAX_HG; ++g) {
        m_st[g] = -INFINITY; s_st[g] = 0.0f; acc[g] = float4(0.0f);
    }

    for (uint col = c_lo + simd; col < c_hi; col += NS) {
        const float4 kf = float4(deq_kv_q8(KQp, KSp, col, lane));
        const float4 vf = float4(deq_kv_q8(VQp, VSp, col, lane));
        for (uint g = 0; g < Hg; ++g) {
            const float score = simd_sum(dot(q_reg[g], kf));
            const float new_m = max(m_st[g], score);
            const float alpha = metal::fast::exp(m_st[g] - new_m);
            const float beta  = metal::fast::exp(score   - new_m);
            m_st[g] = new_m;
            s_st[g] = fma(s_st[g], alpha, beta);
            acc[g]  = fma(acc[g], alpha, beta * vf);
        }
    }

    threadgroup float  m_sh[MAX_HG * NS];
    threadgroup float  s_sh[MAX_HG * NS];
    threadgroup float4 a_sh[MAX_HG * NS * 32];
    for (uint g = 0; g < Hg; ++g) {
        if (lane == 0) {
            m_sh[g * NS + simd] = m_st[g];
            s_sh[g * NS + simd] = s_st[g];
        }
        a_sh[(g * NS + simd) * 32 + lane] = acc[g];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd == 0) {
        for (uint g = 0; g < Hg; ++g) {
            float gm = -INFINITY;
            for (uint p = 0; p < NS; ++p) gm = max(gm, m_sh[g * NS + p]);
            float gs = 0.0f;
            float4 ga = float4(0.0f);
            for (uint p = 0; p < NS; ++p) {
                const float w = metal::fast::exp(m_sh[g * NS + p] - gm);
                gs += s_sh[g * NS + p] * w;
                ga = fma(a_sh[(g * NS + p) * 32 + lane], w, ga);
            }
            const uint head = kv_head * Hg + g;
            const uint idx  = (batch * nheads + head) * SPLITS + split;
            if (lane == 0) { PM[idx] = gm; PS[idx] = gs; }
            reinterpret_cast<device float4*>(PO + (size_t)idx * D)[lane] = ga;
        }
    }
}

// Combine pass: merge n_splits flash partials per (batch, head) → O.
// Grid (nheads, 1, batch); one simdgroup (32 lanes × half4) per head. SPLITS is
// the compile-time PM/PS/PO stride; n_splits (≤ SPLITS) is the host's launched
// split count and must match mha_decode_split's n_splits exactly.
template<uint SPLITS, uint D>
[[kernel, max_total_threads_per_threadgroup(32)]]
void mha_decode_combine_t(
    device const float* PM        [[buffer(0)]],
    device const float* PS        [[buffer(1)]],
    device const float* PO        [[buffer(2)]],
    device half*        O         [[buffer(3)]],
    constant uint&     seq        [[buffer(4)]],
    constant uint&    nheads      [[buffer(5)]],
    constant uint& kv_len         [[buffer(6)]],
    constant uint& n_splits       [[buffer(7)]],
    uint3 gid  [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
    constexpr uint VW = D / 32;  // per-lane head-dim block width (D=128→float4)
    using hvec = typename vec_t<VW>::h;
    using fvec = typename vec_t<VW>::f;
    const uint head  = gid.x;
    const uint batch = gid.z;

    const uint active = min(n_splits, SPLITS);
    const uint base = (batch * nheads + head) * SPLITS;
    float gm = -INFINITY;
    for (uint p = 0; p < active; ++p) gm = max(gm, PM[base + p]);

    float gs = 0.0f;
    fvec ga = fvec(0.0f);
    for (uint p = 0; p < active; ++p) {
        const float w = metal::fast::exp(PM[base + p] - gm);
        gs += PS[base + p] * w;
        const fvec po = reinterpret_cast<const device fvec*>(PO + (size_t)(base + p) * D)[lane];
        ga = fma(po, w, ga);
    }
    const float inv = gs > 0.0f ? 1.0f / gs : 0.0f;
    const size_t o_off = (size_t)(batch * nheads + head) * seq * D;
    reinterpret_cast<device hvec*>(O + o_off)[lane] = hvec(ga * inv);
}


template [[host_name("fa_causal_64")]]
[[kernel]] void fa_d64<true>(
    device const half*, device const half*, device const half*, device half*,
    constant uint&, constant uint&, constant uint&, uint3, uint, uint, uint);

template [[host_name("fa_noncausal_64")]]
[[kernel]] void fa_d64<false>(
    device const half*, device const half*, device const half*, device half*,
    constant uint&, constant uint&, constant uint&, uint3, uint, uint, uint);

template [[host_name("mha_causal")]]
[[kernel]] void fa_dN<true, 128>(
    device const half*, device const half*, device const half*, device half*,
    constant uint&, constant uint&, constant uint&, constant uint&, constant uint&,
    uint3, uint, uint, uint);

template [[host_name("mha_noncausal")]]
[[kernel]] void fa_dN<false, 128>(
    device const half*, device const half*, device const half*, device half*,
    constant uint&, constant uint&, constant uint&, constant uint&, constant uint&,
    uint3, uint, uint, uint);

// head_dim=64 decode/prefill: reads the persistent paged KV cache via
// kv_len/cache_stride (NOT the prefill-only fa_d64 that strides by seq).
template [[host_name("mha_causal_64")]]
[[kernel]] void fa_dN<true, 64>(
    device const half*, device const half*, device const half*, device half*,
    constant uint&, constant uint&, constant uint&, constant uint&, constant uint&,
    uint3, uint, uint, uint);

template [[host_name("mha_noncausal_64")]]
[[kernel]] void fa_dN<false, 64>(
    device const half*, device const half*, device const half*, device half*,
    constant uint&, constant uint&, constant uint&, constant uint&, constant uint&,
    uint3, uint, uint, uint);

// D=64 decode-only (seq==1) Br=1: no idle simdgroups. Bit-identical to
// fa_dN<true,64> at seq==1; host gates head_dim==64 && seq==1 && !split/!q8.
template [[host_name("mha_decode_d64")]]
[[kernel]] void mha_decode_dN_br1<64>(
    device const half*, device const half*, device const half*, device half*,
    constant uint&, constant uint&, constant uint&, constant uint&, constant uint&,
    uint3, uint, uint, uint);

// Prefill-only (seq>1) larger-Br variant. BR=8 → Hg*8*32 threads: 1024 at Hg=4
// (qwen3-4B/8B), 512 at Hg=2; host gates on Hg*BR*32<=1024.
template [[host_name("mha_causal_prefill")]]
[[kernel]] void fa_d128_prefill<true, 8>(
    device const half*, device const half*, device const half*, device half*,
    constant uint&, constant uint&, constant uint&, constant uint&, constant uint&,
    uint3, uint, uint, uint);

template [[host_name("mha_causal_q8")]]
[[kernel]] void fa_d128_q8<true>(
    device const half*, device const char*, device const char*,
    device const half*, device const half*, device half*,
    constant uint&, constant uint&, constant uint&, constant uint&, constant uint&,
    uint3, uint, uint, uint);

// MAX_HG=8 covers all qwen3 variants (Hg ∈ {2,4,5,8}); NS=4 → 128 threads,
// a_sh = 8*4*32 float4 = 16 KB (fits 32 KB with m_sh/s_sh headroom).
// SPLITS=8 is the compile-time cap (PM/PS/PO stride); n_kv_heads(8)*8 = 64 TGs at
// the max split count. Host sizes PM/PS/PO scratch as (batch*nheads*SPLITS)
// floats / floats / (·*D) floats and passes a runtime n_splits ≤ SPLITS chosen
// from kv_len; mha_decode_combine must use the identical n_splits.
template [[host_name("mha_decode_split")]]
[[kernel]] void mha_decode_split_t<8, 4, 8, 128>(
    device const half*, device const half*, device const half*,
    device float*, device float*, device float*,
    constant uint&, constant uint&, constant uint&, constant uint&, constant uint&,
    constant uint&, uint3, uint, uint);

template [[host_name("mha_decode_split_64")]]
[[kernel]] void mha_decode_split_t<8, 4, 8, 64>(
    device const half*, device const half*, device const half*,
    device float*, device float*, device float*,
    constant uint&, constant uint&, constant uint&, constant uint&, constant uint&,
    constant uint&, uint3, uint, uint);

template [[host_name("mha_decode_split_q8")]]
[[kernel]] void mha_decode_split_q8_t<8, 4, 8>(
    device const half*, device const char*, device const char*,
    device const half*, device const half*,
    device float*, device float*, device float*,
    constant uint&, constant uint&, constant uint&, constant uint&, constant uint&,
    constant uint&, uint3, uint, uint);

template [[host_name("mha_decode_combine")]]
[[kernel]] void mha_decode_combine_t<8, 128>(
    device const float*, device const float*, device const float*, device half*,
    constant uint&, constant uint&, constant uint&, constant uint&,
    uint3, uint);

template [[host_name("mha_decode_combine_64")]]
[[kernel]] void mha_decode_combine_t<8, 64>(
    device const float*, device const float*, device const float*, device half*,
    constant uint&, constant uint&, constant uint&, constant uint&,
    uint3, uint);

} // namespace meow::attn
