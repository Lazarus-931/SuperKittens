// Qwen3 (dense) dispatch orchestrator.

#ifndef SUPERKITTENS_QWEN_MODEL_H
#define SUPERKITTENS_QWEN_MODEL_H

#include <Metal/Metal.hpp>
#include <cstdint>

#include "../../inference/weight_store.h"
#include "../../inference/silicon/icb_recorder.h"

namespace meow {
namespace qwen {

struct LayerParams {
    uint32_t batch        = 1;
    uint32_t seq          = 1;
    uint32_t d_model      = 5120;
    uint32_t n_heads      = 64;
    uint32_t n_kv_heads   = 8;
    uint32_t head_dim     = 128;
    uint32_t n_int        = 27392;
    float    eps          = 1e-6f;
    uint32_t use_qk_norm  = 1;  // 0 for Llama-arch (Nemotron-Nano): skip per-head Q/K RMSNorm
    uint32_t rope_interleaved = 0;  // 1 = interleaved/NORM RoPE (Llama GGUF, type 0); 0 = NeoX split-half (Qwen3, type 2)
    uint32_t attn_qkv_bias = 0;  // 1 = add per-layer packed [Q|K|V] bias after QKV matvec (Qwen2/2.5)

    uint32_t layer_idx    = 0;
    uint32_t kv_buf_start = 0;
    uint32_t kv_len       = 1;
    uint32_t cache_size   = 32768;
    uint32_t write_pos    = 0;

    float    rope_freq_base   = 1000000.f;
    int32_t  rope_n_ctx_orig  = 32768;
    float    rope_freq_scale  = 1.f;
    float    rope_ext_factor  = 0.f;
    float    rope_attn_factor = 1.f;
    float    rope_beta_fast   = 32.f;
    float    rope_beta_slow   = 1.f;
};

struct LayerPSOs {
    MTL::ComputePipelineState* rmsnorm;

    MTL::ComputePipelineState* rmsnorm_t1 = nullptr;
    MTL::ComputePipelineState* gemm;              // fp16 GEMM (M>1 path)
    MTL::ComputePipelineState* gemv_m1;           // fp16 GEMV M=1 fast-path (decode)
    MTL::ComputePipelineState* gemv_swiglu_m1;    // fused gate+up+silu_mul GEMV (M=1)
    MTL::ComputePipelineState* gemv_t_m1;         // M=1 matvec w/ transposed weight (LM-head)
    MTL::ComputePipelineState* gemv_t_2dtile_m1 = nullptr;  // 2D-tile variant (preferred when non-null)
    MTL::ComputePipelineState* q8_0_matvec;       // M=1 matvec with Q8_0 weight (decode)
    MTL::ComputePipelineState* q4k_matvec = nullptr;  // M=1 matvec with Q4_K weight
    MTL::ComputePipelineState* q6k_matvec = nullptr;  // M=1 matvec with Q6_K weight
    MTL::ComputePipelineState* q3k_matvec = nullptr;  // M=1 matvec with Q3_K weight (fit-enabler)
    MTL::ComputePipelineState* q5k_matvec = nullptr;  // M=1 matvec with Q5_K weight (fit-enabler)
    // Batched (seq>1) MMA GEMM: amortizes one weight read across M rows so a
    // prefill of T tokens costs ≪ T× the M=1 matvec. Nullable; prefill falls
    // back to the per-row matvec loop when absent.
    MTL::ComputePipelineState* gemm_mma_f16  = nullptr;
    MTL::ComputePipelineState* gemm_mma_bf16 = nullptr;
    MTL::ComputePipelineState* gemm_mma_q8_0 = nullptr;
    MTL::ComputePipelineState* gemm_mma_q4k  = nullptr;
    MTL::ComputePipelineState* gemm_mma_q6k  = nullptr;
    // Small-M (seq 2..8) multi-RHS matvec: the MMA tile cost is M-independent, so
    // at small M (spec-decode verify, short prefill) it pays a fixed ~5x-a-decode
    // floor for Q8_0. These read each weight once and apply it to all M rows in
    // registers (no MMA tile floor). Bit-exact vs the matvec. Q8_0 wins M<=8;
    // Q4_K's MMA is already near-optimal at M>=4 so sm is used only for M<=2.
    MTL::ComputePipelineState* gemm_mma_q8_0_sm = nullptr;
    MTL::ComputePipelineState* gemm_mma_q4k_sm  = nullptr;
    MTL::ComputePipelineState* q8_0_swiglu_m1 = nullptr;  // fused Q8_0 gate+up+SiLU·mul (M=1)
    MTL::ComputePipelineState* q8_0_swiglu_prenorm_m1 = nullptr;  // fused residual+rmsnorm+swiglu (M=1)
    MTL::ComputePipelineState* q8_0_matvec_addres = nullptr;  // matvec + residual add (M=1)
    MTL::ComputePipelineState* split_packed;      // (T, A+B) → (T, A) + (T, B)
    MTL::ComputePipelineState* bias_add = nullptr;  // C[t,n] += bias[n] (Qwen2/2.5 QKV bias); optional
    MTL::ComputePipelineState* rope_qk;           // split-half (NeoX) RoPE on Q, K (Qwen3)
    MTL::ComputePipelineState* rope_qk_il = nullptr;  // interleaved (NORM) RoPE (Llama GGUF); used when rope_interleaved=1
    MTL::ComputePipelineState* attn;              // mha_causal (GQA); D-specific (128/64)
    MTL::ComputePipelineState* attn_prefill = nullptr;  // mha_causal_prefill (BR=8); seq>1, D=128 only
    MTL::ComputePipelineState* attn_decode_d64 = nullptr;  // mha_decode_d64 (Br=1, seq==1, D=64 only)
    // Flash-decoding split-K decode attention (long-ctx). Both non-null together.
    MTL::ComputePipelineState* attn_split   = nullptr;  // mha_decode_split (D-specific)
    MTL::ComputePipelineState* attn_combine = nullptr;  // mha_decode_combine (D-specific)
    MTL::ComputePipelineState* kv_cache_write;
    // Q8_0-KV path (gated by SK_KV_Q8). All three non-null together or disabled.
    MTL::ComputePipelineState* kv_cache_write_q8 = nullptr;
    MTL::ComputePipelineState* attn_q8           = nullptr;  // mha_causal_q8
    MTL::ComputePipelineState* attn_split_q8     = nullptr;  // mha_decode_split_q8
    MTL::ComputePipelineState* add;
    MTL::ComputePipelineState* add_rmsnorm;
    MTL::ComputePipelineState* gated_mlp;
    MTL::ComputePipelineState* silu_mul;          // elementwise SiLU(gate)*up
    MTL::ComputePipelineState* t_seq_to_head;     // (T,H,D) -> (H,T,D)
    MTL::ComputePipelineState* t_head_to_seq;     // (H,T,D) -> (T,H,D)
};

struct LayerBuffers {
    MTL::Buffer* x;

    // Per-layer concatenated weights (offsets via layer_idx)
    MTL::Buffer* w_pre_attn_norm;     // (n_layers, d_model)
    MTL::Buffer* w_qkv;               // this layer's QKV (or Q|K-only) slab
    size_t       w_qkv_inner_off = 0; // byte offset within w_qkv to the start of layer data
    MTL::Buffer* w_qkv_bias = nullptr;  // (n_layers, qkv_N) fp16 packed [Q|K|V] bias; null = no QKV bias
    MTL::Buffer* w_v = nullptr;       // separate V slab when V dtype splits off
    size_t       w_v_inner_off = 0;
    MTL::Buffer* w_q_norm;            // (n_layers, head_dim) — per-head Q-norm γ
    MTL::Buffer* w_k_norm;            // (n_layers, head_dim) — per-head K-norm γ
    MTL::Buffer* w_o;                 // this layer's O slab
    size_t       w_o_inner_off   = 0;
    MTL::Buffer* w_pre_mlp_norm;      // (n_layers, d_model)
    MTL::Buffer* w_gate;              // this layer's gate slab
    size_t       w_gate_inner_off = 0;
    MTL::Buffer* w_up;                // this layer's up slab
    size_t       w_up_inner_off   = 0;
    MTL::Buffer* w_down;              // this layer's down slab
    size_t       w_down_inner_off = 0;

    MTL::Buffer* rope_pos;            // (seq,) int32 positions
    MTL::Buffer* cos_tbl;             // (cache_size, head_dim/2) fp16
    MTL::Buffer* sin_tbl;             // (cache_size, head_dim/2) fp16

    // Per-layer KV caches. fp16 path uses k_cache/v_cache. Q8_0 path
    // (kv_q8=true) uses k_cache_q/v_cache_q (int8 (cache,D)) +
    // k_cache_s/v_cache_s (fp16 (cache,D/32) block scales).
    MTL::Buffer* k_cache;             // (cache_size, n_kv_heads, head_dim)
    MTL::Buffer* v_cache;
    MTL::Buffer* k_cache_q = nullptr;
    MTL::Buffer* v_cache_q = nullptr;
    MTL::Buffer* k_cache_s = nullptr;
    MTL::Buffer* v_cache_s = nullptr;
    bool         kv_q8     = false;

    // Scratch (reused across layers)
    MTL::Buffer* x_norm;
    MTL::Buffer* qkv_packed;          // (T, (n_heads + 2*n_kv_heads)*head_dim)
    MTL::Buffer* q;                   // (T, n_heads, head_dim)
    MTL::Buffer* kv_pack;             // (T, 2*n_kv_heads*head_dim) — intermediate
    MTL::Buffer* k_tmp;               // (T, n_kv_heads, head_dim)
    MTL::Buffer* v_tmp;               // (T, n_kv_heads, head_dim)
    MTL::Buffer* attn_out;            // (T, n_heads, head_dim)
    MTL::Buffer* o_proj;              // (T, d_model)
    MTL::Buffer* y_attn;
    MTL::Buffer* m_in;
    MTL::Buffer* mlp_out;
    MTL::Buffer* y_out;

    // MLP scratch: gate and up projections, each (T, n_int).
    MTL::Buffer* gate_buf;
    MTL::Buffer* up_buf;

    // Head-major (H, T, D) scratch for Q, K, V and attn output.
    MTL::Buffer* q_th;
    MTL::Buffer* k_th;
    MTL::Buffer* v_th;
    MTL::Buffer* attn_out_seq;

    // Flash-decoding split-K partials (decode only). Sized batch*n_heads*SPLITS.
    MTL::Buffer* attn_pm = nullptr;   // float running-max per (head, split)
    MTL::Buffer* attn_ps = nullptr;   // float running-denom
    MTL::Buffer* attn_po = nullptr;   // float (·*head_dim) unnormalized acc

    // Per-weight dtype (default FP16). Used by the M=1 fast-path to pick
    // between gemv_fp16_m1 and q8_0_matvec.
    sk::Dtype dt_qkv  = sk::Dtype::F16;
    sk::Dtype dt_v    = sk::Dtype::F16;
    sk::Dtype dt_o    = sk::Dtype::F16;
    sk::Dtype dt_gate = sk::Dtype::F16;
    sk::Dtype dt_up   = sk::Dtype::F16;
    sk::Dtype dt_down = sk::Dtype::F16;
};

// WHY: All encoder helpers below take an externally-owned encoder so a single
// computeCommandEncoder can be shared across an entire dispatch_layer. Each
// helper inserts memoryBarrierWithScope(Buffers) before its dispatch to honor
// the producer→consumer dep on the previous helper's output. Collapsing the
// per-layer encoder count from ~10 to 1 removes the dominant Apple-Silicon
// encoder-setup overhead at decode (T=1).

inline void enc_barrier(MTL::ComputeCommandEncoder* enc) {
    enc->memoryBarrier(MTL::BarrierScopeBuffers);
}

// Encode a Q8_0 matvec: B (fp16 [K]) × A (q8_0 [N,K] row-major) → C (fp16 [N]).
inline void encode_q8_0_matvec(
    MTL::ComputeCommandEncoder* enc, MTL::ComputePipelineState* pso,
    MTL::Buffer* A, size_t off_A,
    MTL::Buffer* B, size_t off_B,
    MTL::Buffer* C,
    uint32_t N, uint32_t K)
{
    enc_barrier(enc);
    enc->setComputePipelineState(pso);
    enc->setBuffer(B, off_B, 0);
    enc->setBuffer(A, off_A, 1);
    enc->setBuffer(C, 0,     2);
    enc->setBytes(&K, 4, 3);
    enc->setBytes(&N, 4, 4);
    const uint32_t rows_per_tg = 2;
    enc->dispatchThreadgroups(MTL::Size((N + rows_per_tg - 1) / rows_per_tg, 1, 1),
                              MTL::Size(128, 1, 1));
}

// Pick the M=1 quant matvec PSO for a weight dtype. All three quant kernels
// (q8_0/q4k/q6k) share the B=0,A=1,C=2,K=3,N=4 binding + NR0=2 rows/TG, 128
// threads geometry, so dispatch is uniform. Returns nullptr for unsupported.
inline MTL::ComputePipelineState* quant_matvec_pso(const LayerPSOs& P, sk::Dtype dt) {
    switch (dt) {
        case sk::Dtype::Q8_0: return P.q8_0_matvec;
        case sk::Dtype::Q4_K: return P.q4k_matvec;
        case sk::Dtype::Q6_K: return P.q6k_matvec;
        case sk::Dtype::Q3_K: return P.q3k_matvec;
        case sk::Dtype::Q5_K: return P.q5k_matvec;
        default:              return nullptr;
    }
}

// Batched MMA GEMM PSO for a weight dtype. Returns nullptr when no MMA path
// exists for the dtype (caller then falls back to the per-row matvec loop).
inline MTL::ComputePipelineState* gemm_mma_pso(const LayerPSOs& P, sk::Dtype dt) {
    switch (dt) {
        case sk::Dtype::Q8_0: return P.gemm_mma_q8_0;
        case sk::Dtype::Q4_K: return P.gemm_mma_q4k;
        case sk::Dtype::Q6_K: return P.gemm_mma_q6k;
        case sk::Dtype::F16:  return P.gemm_mma_f16;
        case sk::Dtype::BF16: return P.gemm_mma_bf16;
        default:              return nullptr;
    }
}

enum : uint32_t { GEMM_SM_MAXM = 8 };

// Small-M PSO for dtype dt, or nullptr if dt has no sm kernel (only Q8_0/Q4_K).
inline MTL::ComputePipelineState* gemm_sm_pso(const LayerPSOs& P, sk::Dtype dt) {
    switch (dt) {
        case sk::Dtype::Q8_0: return P.gemm_mma_q8_0_sm;
        case sk::Dtype::Q4_K: return P.gemm_mma_q4k_sm;
        default:              return nullptr;
    }
}

// True when the small-M multi-RHS matvec is the measured winner for (dt, M)
// on M4 (lexie bench). Q8_0: sm wins the whole 2..8 band — its MMA floor is
// 3.5-6.4x a decode regardless of M, while sm is 1.0/1.5/2.3-2.8x at M=2/4/8.
// Q4_K: MMA floor is lower (2.4-3.1x), so sm wins M<=4 (1.3-2.4x vs MMA 2.4-
// 3.1x) but MMA overtakes at M=8 (sm 4.4-5.5x). Note Q4_K M=4 on small-K
// gate/up (K=2560) is ~break-even (sm 2.89x vs MMA 2.64x); the large-K
// down/8B projections that dominate a forward favor sm at M=4, so M<=4.
inline bool gemm_sm_wins(sk::Dtype dt, uint32_t M) {
    if (M <= 1 || M > GEMM_SM_MAXM) return false;
    switch (dt) {
        case sk::Dtype::Q8_0: return true;
        case sk::Dtype::Q4_K: return M <= 4;
        default:              return false;
    }
}

// Threadgroup width (NSG*32) for the small-M kernel of dtype dt. Q8_0 sm is
// one simdgroup (occupancy-tuned); Q4_K sm needs 4 (row/sub-block split).
inline uint32_t gemm_sm_threads(sk::Dtype dt) {
    return (dt == sk::Dtype::Q8_0) ? 32u : 128u;
}

// Encode one batched MMA GEMM: A (fp16 [M,K]) × W ([N,K] quant/fp16) → C
// (fp16 [M,N], row stride ldC). BM=8 rows × BN=32 cols per threadgroup.
inline void encode_gemm_mma(
    MTL::ComputeCommandEncoder* enc, MTL::ComputePipelineState* pso,
    MTL::Buffer* A, size_t off_A,
    MTL::Buffer* W, size_t off_W,
    MTL::Buffer* C, size_t off_C,
    uint32_t M, uint32_t N, uint32_t K, uint32_t ldC)
{
    enc_barrier(enc);
    enc->setComputePipelineState(pso);
    enc->setBuffer(A, off_A, 0);
    enc->setBuffer(W, off_W, 1);
    enc->setBuffer(C, off_C, 2);
    enc->setBytes(&M, 4, 3);
    enc->setBytes(&N, 4, 4);
    enc->setBytes(&K, 4, 5);
    enc->setBytes(&ldC, 4, 6);
    // BM matches gemm_mma.metal's BM (rows/threadgroup); see the WHY there for
    // why 32 (weight-read amortization on the bandwidth-bound quant loaders).
    constexpr uint32_t BM = 32, BN = 32;
    enc->dispatchThreadgroups(
        MTL::Size((N + BN - 1) / BN, (M + BM - 1) / BM, 1),
        MTL::Size(64, 1, 1));
}

// Encode the small-M multi-RHS matvec (seq 2..8). Same bindings as gemm_mma but
// the matvec launch geometry (NR0=2 rows/TG, 128 threads); M is a uniform and
// the kernel holds all M rows in registers.
inline void encode_gemm_sm(
    MTL::ComputeCommandEncoder* enc, MTL::ComputePipelineState* pso,
    MTL::Buffer* A, size_t off_A,
    MTL::Buffer* W, size_t off_W,
    MTL::Buffer* C, size_t off_C,
    uint32_t M, uint32_t N, uint32_t K, uint32_t ldC,
    uint32_t tg_threads = 128)
{
    enc_barrier(enc);
    enc->setComputePipelineState(pso);
    enc->setBuffer(A, off_A, 0);
    enc->setBuffer(W, off_W, 1);
    enc->setBuffer(C, off_C, 2);
    enc->setBytes(&M, 4, 3);
    enc->setBytes(&N, 4, 4);
    enc->setBytes(&K, 4, 5);
    enc->setBytes(&ldC, 4, 6);
    constexpr uint32_t NR0 = 2;
    // tg_threads = NSG*32: Q8_0 sm runs one simdgroup (32) for occupancy; Q4_K
    // sm needs 4 simdgroups (128) for its row/sub-block split.
    enc->dispatchThreadgroups(MTL::Size((N + NR0 - 1) / NR0, 1, 1),
                              MTL::Size(tg_threads, 1, 1));
}

// Encode a quant matvec (M rows, looping over rows for M>1). Mirrors
// encode_q8_0_gemm but parameterized on the PSO so Q4_K/Q6_K route here too.
// ldC = output row stride in elements (defaults to N). For the split-QKV path
// the QK/V matvecs write into a [M, qkv_N] packed buffer, so ldC = qkv_N while
// N = qN+kvN (QK) or kvN (V), and off_C selects the column band.
inline void encode_quant_gemm(
    MTL::ComputeCommandEncoder* enc, MTL::ComputePipelineState* pso,
    MTL::Buffer* A, size_t off_A,
    MTL::Buffer* W, size_t off_W,
    MTL::Buffer* C, size_t off_C,
    uint32_t M, uint32_t N, uint32_t K, uint32_t ldC = 0,
    MTL::ComputePipelineState* pso_mma = nullptr,
    MTL::ComputePipelineState* pso_sm = nullptr,
    bool sm_wins = false,
    uint32_t sm_threads = 128)
{
    if (ldC == 0) ldC = N;
    // Small-M (seq 2..8 where measured best): the multi-RHS matvec reads each
    // weight once and applies it to all M rows in registers, avoiding the BM=8
    // MMA tile's M-independent fixed floor. Bit-exact vs the per-row matvec.
    if (sm_wins && pso_sm != nullptr) {
        encode_gemm_sm(enc, pso_sm, A, off_A, W, off_W, C, off_C, M, N, K, ldC,
                       sm_threads);
        return;
    }
    // Prefill (M>1): one MMA GEMM amortizes the weight read across all M rows,
    // so a T-token forward costs ≪ T× the per-row matvec. Decode (M==1) keeps
    // the matvec — its NR0=2 geometry beats the BM=8 MMA tile at one row.
    if (M > 1 && pso_mma != nullptr) {
        encode_gemm_mma(enc, pso_mma, A, off_A, W, off_W, C, off_C, M, N, K, ldC);
        return;
    }
    for (uint32_t m = 0; m < M; ++m) {
        enc_barrier(enc);
        enc->setComputePipelineState(pso);
        enc->setBuffer(A, off_A + (size_t)m * K * 2, 0);
        enc->setBuffer(W, off_W, 1);
        enc->setBuffer(C, off_C + (size_t)m * ldC * 2, 2);
        enc->setBytes(&K, 4, 3);
        enc->setBytes(&N, 4, 4);
        const uint32_t rows_per_tg = 2;
        enc->dispatchThreadgroups(MTL::Size((N + rows_per_tg - 1) / rows_per_tg, 1, 1),
                                  MTL::Size(128, 1, 1));
    }
}


inline void encode_gemm(
    MTL::ComputeCommandEncoder* enc, MTL::ComputePipelineState* pso,
    MTL::Buffer* A, size_t off_A,
    MTL::Buffer* B, size_t off_B,
    MTL::Buffer* C,
    uint32_t M, uint32_t N, uint32_t K,
    MTL::ComputePipelineState* pso_gemv_m1 = nullptr)
{
    enc_barrier(enc);
    if (M == 1 && pso_gemv_m1 != nullptr) {
        enc->setComputePipelineState(pso_gemv_m1);
        enc->setBuffer(A, off_A, 0);
        enc->setBuffer(B, off_B, 1);
        enc->setBuffer(C, 0,     2);
        enc->setBytes(&N, 4, 3);
        enc->setBytes(&K, 4, 4);
        const uint32_t BN = 128;
        enc->dispatchThreadgroups(MTL::Size((N + BN - 1) / BN, 1, 1),
                                  MTL::Size(BN, 1, 1));
        return;
    }
    enc->setComputePipelineState(pso);
    uint32_t ldA = K, ldB = N, ldC = N;
    int transA = 0, transB = 0, has_bias = 0;
    enc->setBuffer(A, off_A, 0);
    enc->setBuffer(B, off_B, 1);
    enc->setBuffer(C, 0,     2);
    enc->setBytes(&M,        4, 3); enc->setBytes(&N,        4, 4);
    enc->setBytes(&K,        4, 5); enc->setBytes(&ldA,      4, 6);
    enc->setBytes(&ldB,      4, 7); enc->setBytes(&ldC,      4, 8);
    enc->setBytes(&transA,   4, 9); enc->setBytes(&transB,   4, 10);
    enc->setBytes(&has_bias, 4, 11);
    enc->setBuffer(C, 0, 12);
    // gemm_fp16 tiles BM=32 rows; a /64 row grid skips rows 32..63 of each
    // 64-block at M>32 (same bug class fixed in deepseek_model.h).
    enc->dispatchThreadgroups(MTL::Size((N + 63) / 64, (M + 31) / 32, 1),
                              MTL::Size(64, 1, 1));
}

// WHY: Q8_0 GEMM by per-row matvec loop. Per-layer matvec call sites use this
// directly (qwen3 runtime weights are always Q8_0), skipping the dt_w branch.
inline void encode_q8_0_gemm(
    MTL::ComputeCommandEncoder* enc, MTL::ComputePipelineState* pso_q8_0,
    MTL::Buffer* A, size_t off_A,
    MTL::Buffer* W, size_t off_W,
    MTL::Buffer* C,
    uint32_t M, uint32_t N, uint32_t K)
{
    for (uint32_t m = 0; m < M; ++m) {
        const size_t off_A_row = off_A + (size_t)m * K * 2;
        enc_barrier(enc);
        enc->setComputePipelineState(pso_q8_0);
        enc->setBuffer(W, off_W, 1);
        enc->setBuffer(A, off_A_row, 0);
        enc->setBuffer(C, (size_t)m * N * 2, 2);
        enc->setBytes(&K, 4, 3);
        enc->setBytes(&N, 4, 4);
        const uint32_t rows_per_tg = 2;
        enc->dispatchThreadgroups(MTL::Size((N + rows_per_tg - 1) / rows_per_tg, 1, 1),
                                  MTL::Size(128, 1, 1));
    }
}

// WHY: fp16/Q4_K fallback for lm_head and future tied-embed paths.
inline void encode_gemm_fallback(
    MTL::ComputeCommandEncoder* enc, MTL::ComputePipelineState* pso_gemm,
    MTL::ComputePipelineState* pso_gemv_m1,
    MTL::Buffer* A, size_t off_A,
    MTL::Buffer* W, size_t off_W,
    MTL::Buffer* C,
    uint32_t M, uint32_t N, uint32_t K)
{
    encode_gemm(enc, pso_gemm, A, off_A, W, off_W, C, M, N, K, pso_gemv_m1);
}

inline void encode_rmsnorm(
    MTL::ComputeCommandEncoder* enc, MTL::ComputePipelineState* pso,
    MTL::Buffer* x, MTL::Buffer* gamma, size_t off_gamma,
    MTL::Buffer* out, uint32_t rows, uint32_t n, float eps,
    MTL::ComputePipelineState* pso_t1 = nullptr,
    bool barrier_before = true)
{
    if (barrier_before) enc_barrier(enc);
    const bool use_t1 = (pso_t1 != nullptr) && (rows == 1u);
    enc->setComputePipelineState(use_t1 ? pso_t1 : pso);
    enc->setBuffer(x,     0,         0);
    enc->setBuffer(gamma, off_gamma, 1);
    enc->setBuffer(out,   0,         2);
    enc->setBytes(&rows, 4, 3);
    enc->setBytes(&n,    4, 4);
    enc->setBytes(&eps,  4, 5);
    if (use_t1) {
        enc->dispatchThreadgroups(MTL::Size(1, rows, 1),
                                  MTL::Size(256, 1, 1));
    } else {
        enc->dispatchThreadgroups(MTL::Size(1, (rows + 3) / 4, 1),
                                  MTL::Size(128, 1, 1));
    }
}

inline void encode_split(
    MTL::ComputeCommandEncoder* enc, MTL::ComputePipelineState* pso,
    MTL::Buffer* src, MTL::Buffer* outA, MTL::Buffer* outB,
    uint32_t T, uint32_t A, uint32_t B)
{
    enc_barrier(enc);
    enc->setComputePipelineState(pso);
    enc->setBuffer(src,  0, 0);
    enc->setBuffer(outA, 0, 1);
    enc->setBuffer(outB, 0, 2);
    enc->setBytes(&T, 4, 3);
    enc->setBytes(&A, 4, 4);
    enc->setBytes(&B, 4, 5);
    enc->dispatchThreads(MTL::Size(A + B, T, 1), MTL::Size(128, 1, 1));
}

// C[t, n] += bias[n] over a [T, N] fp16 buffer (Qwen2/2.5 packed QKV bias).
// bias is bound at its per-layer byte offset so the kernel indexes [0, N).
inline void encode_bias_add(
    MTL::ComputeCommandEncoder* enc, MTL::ComputePipelineState* pso,
    MTL::Buffer* C, MTL::Buffer* bias, size_t bias_off,
    uint32_t N, uint32_t T)
{
    enc_barrier(enc);
    enc->setComputePipelineState(pso);
    enc->setBuffer(C,    0,        0);
    enc->setBuffer(bias, bias_off, 1);
    enc->setBytes(&N, 4, 2);
    enc->setBytes(&T, 4, 3);
    enc->dispatchThreads(MTL::Size(N * T, 1, 1), MTL::Size(128, 1, 1));
}

inline void encode_transpose(
    MTL::ComputeCommandEncoder* enc, MTL::ComputePipelineState* pso,
    MTL::Buffer* src, MTL::Buffer* dst,
    uint32_t T, uint32_t H, uint32_t D,
    size_t off_src = 0, size_t off_dst = 0, bool barrier_before = true)
{
    if (barrier_before) enc_barrier(enc);
    enc->setComputePipelineState(pso);
    enc->setBuffer(src, off_src, 0);
    enc->setBuffer(dst, off_dst, 1);
    enc->setBytes(&T, 4, 2);
    enc->setBytes(&H, 4, 3);
    enc->setBytes(&D, 4, 4);
    enc->dispatchThreads(MTL::Size(D, T, H), MTL::Size(32, 1, 1));
}

inline void encode_rope_qk_inplace(
    MTL::ComputeCommandEncoder* enc, MTL::ComputePipelineState* pso,
    MTL::Buffer* x,
    MTL::Buffer* cos_tbl, size_t cos_off,
    MTL::Buffer* sin_tbl, size_t sin_off,
    uint32_t seq, uint32_t n_heads, uint32_t head_dim,
    bool barrier_before = true, uint32_t batch = 1)
{
    if (barrier_before) enc_barrier(enc);
    enc->setComputePipelineState(pso);
    enc->setBuffer(x,       0,       0);
    enc->setBuffer(x,       0,       1);
    enc->setBuffer(cos_tbl, cos_off, 2);
    enc->setBuffer(sin_tbl, sin_off, 3);
    enc->setBytes(&seq,      4, 4);
    enc->setBytes(&head_dim, 4, 5);
    enc->setBytes(&n_heads,  4, 6);
    // n_rows = batch*seq. The Q/K buffer is [batch*seq, n_heads, D]; each row's
    // rotation position is (row % seq). For batch=1 this is seq (path unchanged).
    const uint32_t n_rows = batch * seq;
    enc->setBytes(&n_rows,   4, 7);
    const uint32_t hd4 = (head_dim / 2) / 4;
    const uint32_t rows_per_tg = (hd4 > 0) ? (1024u / hd4) : 1u;
    const uint32_t row_blocks = (n_rows + rows_per_tg - 1) / rows_per_tg;
    enc->dispatchThreadgroups(
        MTL::Size(n_heads, row_blocks, 1),
        MTL::Size(hd4, rows_per_tg, 1));
}

inline void dispatch_layer(
    MTL::CommandBuffer*  cmd,
    const LayerPSOs&     P,
    const LayerBuffers&  B,
    const LayerParams&   p)
{
    const uint32_t T  = p.batch * p.seq;
    const uint32_t L  = p.layer_idx;
    const uint32_t hd = p.head_dim;
    const uint32_t qN = p.n_heads * hd;
    const uint32_t kvN = p.n_kv_heads * hd;
    const uint32_t qkv_N = qN + 2 * kvN;


    const size_t off_norm     = (size_t)L * p.d_model * 2;
    const size_t off_w_qkv    = B.w_qkv_inner_off;
    const size_t off_w_q_norm = (size_t)L * hd * 2;
    const size_t off_w_k_norm = (size_t)L * hd * 2;
    const size_t off_w_o      = B.w_o_inner_off;
    const size_t off_w_gate   = B.w_gate_inner_off;
    const size_t off_w_up     = B.w_up_inner_off;
    const size_t off_w_down   = B.w_down_inner_off;

    // WHY: One encoder for the entire layer (was ~10). Each helper inserts a
    // buffer-scope memory barrier before its dispatch. Encoder-setup cost on
    // Apple Silicon is ~0.1 ms per encoder; at 36 layers × 9 saved encoders
    // that's ~32 ms/token of pure overhead removed.
    auto* enc = cmd->computeCommandEncoder();

    // 1. Pre-attn RMSNorm.
    encode_rmsnorm(enc, P.rmsnorm, B.x, B.w_pre_attn_norm, off_norm,
                   B.x_norm, T, p.d_model, p.eps, P.rmsnorm_t1);

    // 2. QKV-pack GEMM. When V is split off (Q4_K_M, where attn_v alternates
    // Q4_K/Q6_K per layer), w_qkv holds [Q|K] and w_v holds V; two matvecs feed
    // one packed [Q|K|V] output. Uniform models keep the single-slab path.
    MTL::ComputePipelineState* pso_qkv = quant_matvec_pso(P, B.dt_qkv);
    if (B.w_v != nullptr) {
        MTL::ComputePipelineState* pso_v = quant_matvec_pso(P, B.dt_v);
        if (pso_qkv && pso_v) {
            // Output is [T, qkv_N] packed; QK writes band [0:qN+kvN], V writes
            // band [qN+kvN:qkv_N]. ldC = qkv_N so per-row strides stay correct
            // during prefill (T>1).
            encode_quant_gemm(enc, pso_qkv, B.x_norm, 0, B.w_qkv, off_w_qkv,
                              B.qkv_packed, 0, T, qN + kvN, p.d_model, qkv_N,
                              gemm_mma_pso(P, B.dt_qkv),
                              gemm_sm_pso(P, B.dt_qkv), gemm_sm_wins(B.dt_qkv, T),
                              gemm_sm_threads(B.dt_qkv));
            encode_quant_gemm(enc, pso_v, B.x_norm, 0, B.w_v, B.w_v_inner_off,
                              B.qkv_packed, (size_t)(qN + kvN) * 2, T, kvN, p.d_model, qkv_N,
                              gemm_mma_pso(P, B.dt_v),
                              gemm_sm_pso(P, B.dt_v), gemm_sm_wins(B.dt_v, T),
                              gemm_sm_threads(B.dt_v));
        } else {
            encode_gemm_fallback(enc, P.gemm, P.gemv_m1, B.x_norm, 0, B.w_qkv, off_w_qkv,
                                 B.qkv_packed, T, qkv_N, p.d_model);
        }
    } else if (pso_qkv != nullptr) {
        encode_quant_gemm(enc, pso_qkv, B.x_norm, 0, B.w_qkv, off_w_qkv,
                          B.qkv_packed, 0, T, qkv_N, p.d_model, /*ldC=*/0,
                          gemm_mma_pso(P, B.dt_qkv),
                          gemm_sm_pso(P, B.dt_qkv), gemm_sm_wins(B.dt_qkv, T),
                          gemm_sm_threads(B.dt_qkv));
    } else {
        encode_gemm_fallback(enc, P.gemm, P.gemv_m1, B.x_norm, 0, B.w_qkv, off_w_qkv,
                             B.qkv_packed, T, qkv_N, p.d_model);
    }

    // 2b. QKV bias add (Qwen2/2.5). Broadcast per-layer packed [Q|K|V] bias over
    // the [T, qkv_N] matvec output before the split. Flag-gated + buffer-gated so
    // Qwen3/Llama/Mistral (attn_qkv_bias=0, w_qkv_bias=null) never reach it.
    if (p.attn_qkv_bias && B.w_qkv_bias != nullptr && P.bias_add != nullptr) {
        const size_t bias_off = (size_t)L * qkv_N * 2;
        encode_bias_add(enc, P.bias_add, B.qkv_packed, B.w_qkv_bias, bias_off,
                        qkv_N, T);
    }

    // 3. Splits.
    encode_split(enc, P.split_packed, B.qkv_packed, B.q, B.kv_pack,
                 T, qN, 2 * kvN);
    encode_split(enc, P.split_packed, B.kv_pack, B.k_tmp, B.v_tmp,
                 T, kvN, kvN);

    // 4. Per-head Q/K-norm (Qwen3 only). K-norm writes a distinct buffer from
    // Q-norm (B.k_tmp vs B.q); its dep on split-2 is honored transitively by the
    // barrier before Q-norm (Metal buffer-scope barriers are full fences), so
    // skip the redundant barrier here. Llama-arch (Nemotron-Nano, use_qk_norm=0)
    // has no per-head Q/K RMSNorm — skip both so Q/K feed RoPE unmodified.
    if (p.use_qk_norm) {
        encode_rmsnorm(enc, P.rmsnorm, B.q, B.w_q_norm, off_w_q_norm,
                       B.q, T * p.n_heads, hd, p.eps);
        encode_rmsnorm(enc, P.rmsnorm, B.k_tmp, B.w_k_norm, off_w_k_norm,
                       B.k_tmp, T * p.n_kv_heads, hd, p.eps,
                       /*pso_t1=*/nullptr, /*barrier_before=*/false);
    }

    // 5. RoPE on Q and K. RoPE-K writes a distinct buffer from RoPE-Q;
    // dep on K-norm is honored by the barrier before RoPE-Q. Llama GGUFs
    // (rope type 0, converter-permuted Q/K) need interleaved-pair rotation;
    // Qwen3 (type 2) needs split-half. Same cos/sin tables either way.
    {
        const size_t cs_off = (size_t)p.write_pos * (hd / 2) * 2;
        MTL::ComputePipelineState* rope_pso =
            (p.rope_interleaved && P.rope_qk_il) ? P.rope_qk_il : P.rope_qk;
        encode_rope_qk_inplace(enc, rope_pso, B.q,
                               B.cos_tbl, cs_off, B.sin_tbl, cs_off,
                               p.seq, p.n_heads, hd,
                               /*barrier_before=*/true, p.batch);
        encode_rope_qk_inplace(enc, rope_pso, B.k_tmp,
                               B.cos_tbl, cs_off, B.sin_tbl, cs_off,
                               p.seq, p.n_kv_heads, hd,
                               /*barrier_before=*/false, p.batch);
    }

    // Profiling-only stage gates (seq>1 prefill only; decode untouched). Diff
    // GPUPROF gpu_busy with/without to attribute TTFT to xpose vs attention.
    static const bool prof_skip_xpose = (getenv("SK_PROF_SKIP_XPOSE") != nullptr);
    static const bool prof_skip_attn  = (getenv("SK_PROF_SKIP_ATTN")  != nullptr);

    MTL::Buffer* q_in = B.q;
    MTL::Buffer* k_in = B.k_tmp;
    MTL::Buffer* v_in = B.v_tmp;
    if (p.seq > 1 && !prof_skip_xpose) {
        if (p.batch > 1) {
            // Batched prefill: lanes are independent [seq,H,D] blocks and the
            // consumers (kv_cache_write, mha_causal) index (B,H,seq,D), so each
            // lane transposes within its own slab. A single (T=batch*seq)
            // transpose would interleave lanes' rows into one head-major block,
            // which is the seq>1 batched corruption this loop fixes. Lane byte
            // offset is the same for src and dst (seq*H*D elems either way).
            const size_t q_lane  = (size_t)p.seq * p.n_heads    * hd * 2;
            const size_t kv_lane = (size_t)p.seq * p.n_kv_heads * hd * 2;
            for (uint32_t b = 0; b < p.batch; ++b) {
                const bool bar = (b == 0);
                encode_transpose(enc, P.t_seq_to_head, B.q,     B.q_th, p.seq, p.n_heads,    hd,
                                 (size_t)b * q_lane,  (size_t)b * q_lane,  bar);
                encode_transpose(enc, P.t_seq_to_head, B.k_tmp, B.k_th, p.seq, p.n_kv_heads, hd,
                                 (size_t)b * kv_lane, (size_t)b * kv_lane, bar);
                encode_transpose(enc, P.t_seq_to_head, B.v_tmp, B.v_th, p.seq, p.n_kv_heads, hd,
                                 (size_t)b * kv_lane, (size_t)b * kv_lane, bar);
            }
        } else {
            encode_transpose(enc, P.t_seq_to_head, B.q,     B.q_th, p.seq, p.n_heads,    hd);
            encode_transpose(enc, P.t_seq_to_head, B.k_tmp, B.k_th, p.seq, p.n_kv_heads, hd);
            encode_transpose(enc, P.t_seq_to_head, B.v_tmp, B.v_th, p.seq, p.n_kv_heads, hd);
        }
        q_in = B.q_th; k_in = B.k_th; v_in = B.v_th;
    } else if (p.seq > 1) {
        q_in = B.q_th; k_in = B.k_th; v_in = B.v_th;
    }

    // 6. KV cache write.
    enc_barrier(enc);
    if (B.kv_q8) {
        enc->setComputePipelineState(P.kv_cache_write_q8);
        enc->setBuffer(k_in,        0, 0);
        enc->setBuffer(v_in,        0, 1);
        enc->setBuffer(B.k_cache_q, 0, 2);
        enc->setBuffer(B.v_cache_q, 0, 3);
        enc->setBuffer(B.k_cache_s, 0, 4);
        enc->setBuffer(B.v_cache_s, 0, 5);
        enc->setBytes(&p.batch,       4, 6);
        enc->setBytes(&p.n_kv_heads,  4, 7);
        enc->setBytes(&hd,            4, 8);
        enc->setBytes(&p.seq,         4, 9);
        enc->setBytes(&p.write_pos,   4, 10);
        enc->setBytes(&p.cache_size,  4, 11);
        // One 32-lane simdgroup per (block, t, bh); nblk = hd/32 blocks per row.
        const uint32_t nblk = hd / 32u;
        enc->dispatchThreads(MTL::Size(nblk * 32u, p.seq, p.batch * p.n_kv_heads),
                             MTL::Size(32, 1, 1));
    } else {
        enc->setComputePipelineState(P.kv_cache_write);
        enc->setBuffer(k_in,     0, 0);
        enc->setBuffer(v_in,     0, 1);
        enc->setBuffer(B.k_cache, 0, 2);
        enc->setBuffer(B.v_cache, 0, 3);
        enc->setBytes(&p.batch,       4, 4);
        enc->setBytes(&p.n_kv_heads,  4, 5);
        enc->setBytes(&hd,            4, 6);
        enc->setBytes(&p.seq,         4, 7);
        enc->setBytes(&p.write_pos,   4, 8);
        enc->setBytes(&p.cache_size,  4, 9);
        const uint32_t D4 = hd / 4;
        enc->dispatchThreads(MTL::Size(D4, p.seq, p.batch * p.n_kv_heads),
                             MTL::Size(32, 4, 1));
    }

    // 7. Attention. kv-length-conditional dispatch: at decode (seq==1) with a
    // long context the flash-decoding split-K kernel reads K/V straight from
    // device (no per-tile threadgroup staging) and splits the kv range across
    // many threadgroups; at short kv mha_causal's TG staging amortizes better.
    // Crossover is Hg-dependent (measured on M4 base): Hg=2 wins unambiguously
    // from kv≈384, Hg≥4 only from kv≈1024 (4× the per-key simd_sum work delays
    // the crossover). Gates sit past the break-even so neither config regresses.
    enc_barrier(enc);
    if (!(p.seq > 1 && prof_skip_attn)) {
        const uint32_t kv_len = p.kv_len;
        const uint32_t cache_stride = p.cache_size;
        const uint32_t Hg_attn = p.n_heads / p.n_kv_heads;
        constexpr uint32_t SPLITS = 8u;  // compile-time PM/PS/PO stride
        constexpr uint32_t NS = 4u;
        const uint32_t kv_gate = (Hg_attn <= 2u) ? 384u : 1024u;
        const bool use_split = (p.seq == 1) && (kv_len >= kv_gate)
                               && (B.kv_q8 ? (P.attn_split_q8 != nullptr)
                                           : (P.attn_split != nullptr))
                               && (P.attn_combine != nullptr)
                               && (B.attn_pm != nullptr);
        if (use_split) {
            // ~256 keys/split, clamped to [1, SPLITS].
            uint32_t n_splits = (kv_len + 255u) / 256u;
            if (n_splits < 1u) n_splits = 1u;
            if (n_splits > SPLITS) n_splits = SPLITS;
            if (B.kv_q8) {
                enc->setComputePipelineState(P.attn_split_q8);
                enc->setBuffer(q_in,        0, 0);
                enc->setBuffer(B.k_cache_q, 0, 1);
                enc->setBuffer(B.v_cache_q, 0, 2);
                enc->setBuffer(B.k_cache_s, 0, 3);
                enc->setBuffer(B.v_cache_s, 0, 4);
                enc->setBuffer(B.attn_pm,  0, 5);
                enc->setBuffer(B.attn_ps,  0, 6);
                enc->setBuffer(B.attn_po,  0, 7);
                enc->setBytes(&p.seq,         4, 8);
                enc->setBytes(&p.n_heads,     4, 9);
                enc->setBytes(&p.n_kv_heads,  4, 10);
                enc->setBytes(&kv_len,        4, 11);
                enc->setBytes(&cache_stride,  4, 12);
                enc->setBytes(&n_splits,      4, 13);
            } else {
                enc->setComputePipelineState(P.attn_split);
                enc->setBuffer(q_in,       0, 0);
                enc->setBuffer(B.k_cache,  0, 1);
                enc->setBuffer(B.v_cache,  0, 2);
                enc->setBuffer(B.attn_pm,  0, 3);
                enc->setBuffer(B.attn_ps,  0, 4);
                enc->setBuffer(B.attn_po,  0, 5);
                enc->setBytes(&p.seq,         4, 6);
                enc->setBytes(&p.n_heads,     4, 7);
                enc->setBytes(&p.n_kv_heads,  4, 8);
                enc->setBytes(&kv_len,        4, 9);
                enc->setBytes(&cache_stride,  4, 10);
                enc->setBytes(&n_splits,      4, 11);
            }
            enc->dispatchThreadgroups(
                MTL::Size(p.n_kv_heads, n_splits, p.batch),
                MTL::Size(NS * 32, 1, 1));

            enc_barrier(enc);
            enc->setComputePipelineState(P.attn_combine);
            enc->setBuffer(B.attn_pm,  0, 0);
            enc->setBuffer(B.attn_ps,  0, 1);
            enc->setBuffer(B.attn_po,  0, 2);
            enc->setBuffer(B.attn_out, 0, 3);
            enc->setBytes(&p.seq,         4, 4);
            enc->setBytes(&p.n_heads,     4, 5);
            enc->setBytes(&kv_len,        4, 6);
            enc->setBytes(&n_splits,      4, 7);
            enc->dispatchThreadgroups(
                MTL::Size(p.n_heads, 1, p.batch),
                MTL::Size(32, 1, 1));
        } else if (B.kv_q8) {
            enc->setComputePipelineState(P.attn_q8);
            enc->setBuffer(q_in,        0, 0);
            enc->setBuffer(B.k_cache_q, 0, 1);
            enc->setBuffer(B.v_cache_q, 0, 2);
            enc->setBuffer(B.k_cache_s, 0, 3);
            enc->setBuffer(B.v_cache_s, 0, 4);
            enc->setBuffer(B.attn_out,  0, 5);
            enc->setBytes(&p.seq,         4, 6);
            enc->setBytes(&p.n_heads,     4, 7);
            enc->setBytes(&p.n_kv_heads,  4, 8);
            enc->setBytes(&kv_len,        4, 9);
            enc->setBytes(&cache_stride,  4, 10);
            enc->dispatchThreadgroups(
                MTL::Size(p.n_kv_heads, (p.seq + 1) / 2, p.batch),
                MTL::Size(Hg_attn * 2 * 32, 1, 1));
        } else {
            // Prefill (seq>1): BR=8 variant reuses each K/V smem tile across 8
            // query rows (vs Br=2), cutting the O(seq^2) K/V HBM re-stream ~4x.
            // Gated on Hg*BR*32<=1024 (kernel max). Decode (seq==1) keeps Br=2.
            constexpr uint32_t PBR = 8u;
            const bool use_prefill = (p.seq > 1) && (P.attn_prefill != nullptr)
                                     && (Hg_attn * PBR * 32u <= 1024u);
            // D=64 decode (seq==1): Br=1 variant uses Hg*32 threads (no idle
            // simdgroups), bit-identical to fa_dN<…,64>'s Br=2 path. Same 9
            // buffers in the same order, only the grid/tg differ.
            const bool use_d64 = (p.seq == 1) && (P.attn_decode_d64 != nullptr);
            MTL::ComputePipelineState* pso_a =
                use_d64 ? P.attn_decode_d64 : (use_prefill ? P.attn_prefill : P.attn);
            const uint32_t br = use_prefill ? PBR : (use_d64 ? 1u : 2u);
            enc->setComputePipelineState(pso_a);
            enc->setBuffer(q_in,       0, 0);
            enc->setBuffer(B.k_cache,  0, 1);
            enc->setBuffer(B.v_cache,  0, 2);
            enc->setBuffer(B.attn_out, 0, 3);
            enc->setBytes(&p.seq,         4, 4);
            enc->setBytes(&p.n_heads,     4, 5);
            enc->setBytes(&p.n_kv_heads,  4, 6);
            enc->setBytes(&kv_len,        4, 7);
            enc->setBytes(&cache_stride,  4, 8);
            enc->dispatchThreadgroups(
                MTL::Size(p.n_kv_heads, (p.seq + br - 1u) / br, p.batch),
                MTL::Size(Hg_attn * br * 32, 1, 1));
        }
    }

    MTL::Buffer* attn_o_in = B.attn_out;
    if (p.seq > 1 && !prof_skip_xpose) {
        if (p.batch > 1) {
            const size_t o_lane = (size_t)p.seq * p.n_heads * hd * 2;
            for (uint32_t b = 0; b < p.batch; ++b)
                encode_transpose(enc, P.t_head_to_seq, B.attn_out, B.attn_out_seq,
                                 p.seq, p.n_heads, hd,
                                 (size_t)b * o_lane, (size_t)b * o_lane, b == 0);
        } else {
            encode_transpose(enc, P.t_head_to_seq, B.attn_out, B.attn_out_seq,
                             p.seq, p.n_heads, hd);
        }
        attn_o_in = B.attn_out_seq;
    } else if (p.seq > 1) {
        attn_o_in = B.attn_out_seq;
    }

    // 8. O-projection.
    MTL::ComputePipelineState* pso_o = quant_matvec_pso(P, B.dt_o);
    if (pso_o != nullptr) {
        encode_quant_gemm(enc, pso_o, attn_o_in, 0, B.w_o, off_w_o,
                          B.o_proj, 0, T, p.d_model, p.n_heads * hd, /*ldC=*/0,
                          gemm_mma_pso(P, B.dt_o),
                          gemm_sm_pso(P, B.dt_o), gemm_sm_wins(B.dt_o, T),
                          gemm_sm_threads(B.dt_o));
    } else {
        encode_gemm_fallback(enc, P.gemm, P.gemv_m1, attn_o_in, 0, B.w_o, off_w_o,
                             B.o_proj, T, p.d_model, p.n_heads * hd);
    }

    // 9. Residual + pre-MLP RMSNorm. Fused with swiglu when prenorm path
    // is available (saves 1 dispatch + 1 fence per layer at decode).
    const bool mlp_is_q8 = (B.dt_gate == sk::Dtype::Q8_0 || B.dt_up == sk::Dtype::Q8_0);
    const bool mlp_both_q8 = (B.dt_gate == sk::Dtype::Q8_0 && B.dt_up == sk::Dtype::Q8_0);
    // gemv_swiglu_m1 reads weights as fp16; only use it when gate AND up are
    // genuinely fp16/bf16 (Q4_K/Q6_K must route through the quant matvec path).
    auto is_fp16ish = [](sk::Dtype d) { return d == sk::Dtype::F16 || d == sk::Dtype::BF16; };
    const bool mlp_both_fp16 = is_fp16ish(B.dt_gate) && is_fp16ish(B.dt_up);
    const bool use_prenorm_fused = (T == 1 && mlp_both_q8 && P.q8_0_swiglu_prenorm_m1 != nullptr);
    if (!use_prenorm_fused) {
        enc_barrier(enc);
        enc->setComputePipelineState(P.add_rmsnorm);
        enc->setBuffer(B.x,               0,        0);
        enc->setBuffer(B.o_proj,          0,        1);
        enc->setBuffer(B.w_pre_mlp_norm,  off_norm, 2);
        enc->setBuffer(B.y_attn,          0,        3);
        enc->setBuffer(B.m_in,            0,        4);
        enc->setBytes(&T,         4, 5);
        enc->setBytes(&p.d_model, 4, 6);
        enc->setBytes(&p.eps,     4, 7);
        enc->dispatchThreadgroups(MTL::Size(1, T, 1), MTL::Size(128, 1, 1));
    }

    if (use_prenorm_fused) {
        // WHY: 1 dispatch for residual+rmsnorm+gate+up+silu*mul. Writes
        // y_attn[K] (from TG0) and up_buf[N].
        enc_barrier(enc);
        enc->setComputePipelineState(P.q8_0_swiglu_prenorm_m1);
        enc->setBuffer(B.x,               0,        0);
        enc->setBuffer(B.o_proj,          0,        1);
        enc->setBuffer(B.w_pre_mlp_norm,  off_norm, 2);
        enc->setBuffer(B.w_gate,          off_w_gate, 3);
        enc->setBuffer(B.w_up,            off_w_up,   4);
        enc->setBuffer(B.y_attn,          0,        5);
        enc->setBuffer(B.up_buf,          0,        6);
        uint32_t K_v = p.d_model, N_v = p.n_int;
        enc->setBytes(&K_v,    4, 7);
        enc->setBytes(&N_v,    4, 8);
        enc->setBytes(&p.eps,  4, 9);
        const uint32_t rows_per_tg = 2;
        enc->dispatchThreadgroups(MTL::Size((N_v + rows_per_tg - 1) / rows_per_tg, 1, 1),
                                  MTL::Size(128, 1, 1));
    } else if (T == 1 && mlp_both_q8 && P.q8_0_swiglu_m1 != nullptr) {
        // WHY: collapse 3 dispatches (gate matvec, up matvec, silu*mul) into 1.
        enc_barrier(enc);
        enc->setComputePipelineState(P.q8_0_swiglu_m1);
        enc->setBuffer(B.m_in,   0,          0);
        enc->setBuffer(B.w_gate, off_w_gate, 1);
        enc->setBuffer(B.w_up,   off_w_up,   2);
        enc->setBuffer(B.up_buf, 0,          3);
        uint32_t K_v = p.d_model, N_v = p.n_int;
        enc->setBytes(&K_v, 4, 4);
        enc->setBytes(&N_v, 4, 5);
        const uint32_t rows_per_tg = 2;
        enc->dispatchThreadgroups(MTL::Size((N_v + rows_per_tg - 1) / rows_per_tg, 1, 1),
                                  MTL::Size(128, 1, 1));
    } else if (T == 1 && P.gemv_swiglu_m1 != nullptr && mlp_both_fp16) {
        enc_barrier(enc);
        enc->setComputePipelineState(P.gemv_swiglu_m1);
        enc->setBuffer(B.m_in,    0,          0);
        enc->setBuffer(B.w_gate,  off_w_gate, 1);
        enc->setBuffer(B.w_up,    off_w_up,   2);
        enc->setBuffer(B.up_buf,  0,          3);
        uint32_t N_v = p.n_int, K_v = p.d_model;
        enc->setBytes(&N_v, 4, 4);
        enc->setBytes(&K_v, 4, 5);
        const uint32_t BN = 128;
        enc->dispatchThreadgroups(MTL::Size((N_v + BN - 1) / BN, 1, 1),
                                  MTL::Size(BN, 1, 1));
    } else {
        MTL::ComputePipelineState* pso_gate = quant_matvec_pso(P, B.dt_gate);
        MTL::ComputePipelineState* pso_up   = quant_matvec_pso(P, B.dt_up);
        if (pso_gate != nullptr) {
            encode_quant_gemm(enc, pso_gate, B.m_in, 0, B.w_gate, off_w_gate,
                              B.gate_buf, 0, T, p.n_int, p.d_model, /*ldC=*/0,
                              gemm_mma_pso(P, B.dt_gate),
                              gemm_sm_pso(P, B.dt_gate), gemm_sm_wins(B.dt_gate, T),
                              gemm_sm_threads(B.dt_gate));
        } else {
            encode_gemm_fallback(enc, P.gemm, P.gemv_m1, B.m_in, 0, B.w_gate, off_w_gate,
                                 B.gate_buf, T, p.n_int, p.d_model);
        }
        if (pso_up != nullptr) {
            encode_quant_gemm(enc, pso_up, B.m_in, 0, B.w_up, off_w_up,
                              B.up_buf, 0, T, p.n_int, p.d_model, /*ldC=*/0,
                              gemm_mma_pso(P, B.dt_up),
                              gemm_sm_pso(P, B.dt_up), gemm_sm_wins(B.dt_up, T),
                              gemm_sm_threads(B.dt_up));
        } else {
            encode_gemm_fallback(enc, P.gemm, P.gemv_m1, B.m_in, 0, B.w_up, off_w_up,
                                 B.up_buf, T, p.n_int, p.d_model);
        }
        enc_barrier(enc);
        enc->setComputePipelineState(P.silu_mul);
        enc->setBuffer(B.gate_buf, 0, 0);
        enc->setBuffer(B.up_buf,   0, 1);
        enc->setBuffer(B.up_buf,   0, 2);
        uint32_t N_total = T * p.n_int;
        enc->setBytes(&N_total, 4, 3);
        enc->dispatchThreadgroups(MTL::Size((N_total + 255) / 256, 1, 1),
                                  MTL::Size(256, 1, 1));
    }
    // 10/11. Down-projection + final residual. Fused when down is Q8_0 and
    // the addres PSO is available: skips the standalone add_f16 dispatch and
    // the L2-drain fence between mlp_out and final residual.
    const bool use_down_addres = (T == 1
                                  && B.dt_down == sk::Dtype::Q8_0
                                  && P.q8_0_matvec_addres != nullptr);
    if (use_down_addres) {
        enc_barrier(enc);
        enc->setComputePipelineState(P.q8_0_matvec_addres);
        enc->setBuffer(B.up_buf,  0,          0);
        enc->setBuffer(B.w_down,  off_w_down, 1);
        enc->setBuffer(B.y_attn,  0,          2);
        enc->setBuffer(B.y_out,   0,          3);
        uint32_t K_v = p.n_int, N_v = p.d_model;
        enc->setBytes(&K_v, 4, 4);
        enc->setBytes(&N_v, 4, 5);
        const uint32_t rows_per_tg = 2;
        enc->dispatchThreadgroups(MTL::Size((N_v + rows_per_tg - 1) / rows_per_tg, 1, 1),
                                  MTL::Size(128, 1, 1));
    } else {
        MTL::ComputePipelineState* pso_down = quant_matvec_pso(P, B.dt_down);
        if (pso_down != nullptr) {
            encode_quant_gemm(enc, pso_down, B.up_buf, 0, B.w_down, off_w_down,
                              B.mlp_out, 0, T, p.d_model, p.n_int, /*ldC=*/0,
                              gemm_mma_pso(P, B.dt_down),
                              gemm_sm_pso(P, B.dt_down), gemm_sm_wins(B.dt_down, T),
                              gemm_sm_threads(B.dt_down));
        } else {
            encode_gemm_fallback(enc, P.gemm, P.gemv_m1, B.up_buf, 0, B.w_down, off_w_down,
                                 B.mlp_out, T, p.d_model, p.n_int);
        }
        enc_barrier(enc);
        enc->setComputePipelineState(P.add);
        enc->setBuffer(B.y_attn,  0, 0);
        enc->setBuffer(B.mlp_out, 0, 1);
        enc->setBuffer(B.y_out,   0, 2);
        uint32_t n = T * p.d_model;
        enc->setBytes(&n, 4, 3);
        uint32_t total = (n / 4u) + (n & 3u);
        enc->dispatchThreadgroups(MTL::Size((total + 127) / 128, 1, 1),
                                  MTL::Size(128, 1, 1));
    }

    enc->endEncoding();
}

// ─── Model level ─────────────────────────────────────────────────────

struct ModelParams {
    uint32_t batch        = 1;
    uint32_t seq          = 1;
    uint32_t n_layers     = 64;
    uint32_t d_model      = 5120;
    uint32_t n_heads      = 64;
    uint32_t n_kv_heads   = 8;
    uint32_t head_dim     = 128;
    uint32_t n_int        = 27392;
    uint32_t cache_max    = 32768;
    uint32_t vocab_size   = 151936;
    float    eps          = 1e-6f;
    uint32_t use_qk_norm  = 1;  // 0 for Llama-arch (Nemotron-Nano): skip per-head Q/K RMSNorm
    uint32_t rope_interleaved = 0;  // 1 = interleaved/NORM RoPE (Llama GGUF, type 0); 0 = NeoX (Qwen3, type 2)
    uint32_t attn_qkv_bias = 0;  // 1 = add per-layer packed [Q|K|V] bias after QKV matvec (Qwen2/2.5)
    uint32_t current_pos  = 0;

    // RoPE
    float    rope_freq_base   = 1000000.f;
    int32_t  rope_n_ctx_orig  = 32768;
    float    rope_freq_scale  = 1.f;
    float    rope_ext_factor  = 0.f;
    float    rope_attn_factor = 1.f;
    float    rope_beta_fast   = 32.f;
    float    rope_beta_slow   = 1.f;

    // Batched-decode: when set (seq==1, batch=N>1 lockstep-decode), the LM head
    // projects ALL T=batch rows and argmax writes output_id[0..T]. Default 0
    // keeps the single-row (last-position) decode/prefill path byte-identical.
    uint32_t decode_all_rows = 0;

    // Batched (batch=N, seq>1) chunked prefill. 0 = off (all existing paths
    // byte-identical). 1 = interior chunk: layers/KV only, skip final norm +
    // head + argmax (serving needs logits only after the full prompt). 2 =
    // final chunk: project each lane's LAST prompt row (b*seq+seq-1) -> logits
    // row b, argmax -> output_id[b]. The decode_all_rows head would project
    // all batch*seq rows (vocab×T dead work) and its argmax indexing assumes
    // seq==1, so the prefill tail is its own path.
    uint32_t batched_prefill = 0;

    // Debug knobs (default = full model, no capture)
    uint32_t layers_run     = 0;   // 0 → all n_layers; else only first N
    int32_t  capture_layer  = -1;  // if >=0, copy post-layer residual to capture_buf

    bool     kv_q8          = false;  // Q8_0 KV cache write + attention
};

struct ModelPSOs {
    LayerPSOs layer;
    MTL::ComputePipelineState* embedding_lookup;
    MTL::ComputePipelineState* argmax;
    // Optional 2-pass argmax PSOs. When both non-null AND T==1, dispatch_model
    // uses the 2-pass parallel reduce (≈3.7–5.9× faster at large V). Falls
    // back to single-pass `argmax` if either is nullptr.
    MTL::ComputePipelineState* argmax_partial = nullptr;  // fp16 partial
    MTL::ComputePipelineState* argmax_reduce  = nullptr;  // partials → out

    // Optional ICB-compatible copies of the 2-pass argmax PSOs. Same kernels,
    // built with setSupportIndirectCommandBuffers(true). When non-null and the
    // recorder + args buffers in ModelBuffers are also non-null, the tail
    // 2-pass argmax executes via a pre-recorded MTLIndirectCommandBuffer in
    // one executeCommandsInBuffer call (skips two computeCommandEncoder
    // setup/teardown cycles per decoded token).
    MTL::ComputePipelineState* argmax_partial_icb = nullptr;
    MTL::ComputePipelineState* argmax_reduce_icb  = nullptr;
};

struct LayerCache {
    MTL::Buffer* k;
    MTL::Buffer* v;
    // Q8_0-KV buffers (populated only when the handle ran with SK_KV_Q8).
    MTL::Buffer* kq = nullptr;
    MTL::Buffer* vq = nullptr;
    MTL::Buffer* ks = nullptr;
    MTL::Buffer* vs = nullptr;
};

struct ModelWeights {
    MTL::Buffer* w_embed;
    MTL::Buffer* w_pre_attn_norm;
    std::vector<MTL::Buffer*> w_qkv;        // size n_layers
    std::vector<size_t>       w_qkv_off;    // size n_layers
    MTL::Buffer* w_qkv_bias = nullptr;      // (n_layers, qkv_N) fp16 packed [Q|K|V]; null = no QKV bias
    // V-proj split: when V's GGUF dtype differs from Q/K (Q4_K_M: Q/K=Q4_K,
    // V=Q6_K, distinct block sizes) the QKV slab holds only [Q|K] and V lives
    // in its own per-layer buffer. Empty otherwise (uniform-dtype fast path).
    std::vector<MTL::Buffer*> w_v;
    std::vector<size_t>       w_v_off;
    MTL::Buffer* w_q_norm;        // (n_layers, head_dim)
    MTL::Buffer* w_k_norm;        // (n_layers, head_dim)
    std::vector<MTL::Buffer*> w_o;
    std::vector<size_t>       w_o_off;
    MTL::Buffer* w_pre_mlp_norm;
    MTL::Buffer* w_final_norm;
    std::vector<MTL::Buffer*> w_gate;
    std::vector<size_t>       w_gate_off;
    std::vector<MTL::Buffer*> w_up;
    std::vector<size_t>       w_up_off;
    std::vector<MTL::Buffer*> w_down;
    std::vector<size_t>       w_down_off;
    MTL::Buffer* w_lm_head;  // null when tied
    size_t       off_w_lm_head = 0;  // byte offset into w_lm_head (for mmap-backed weights)
    const LayerCache* layer_caches;

    // Per-projection dtypes (default FP16). Set by loader. dt_qkv is the dtype
    // of the Q+K slab. dt_v / dt_down are PER-LAYER: Q4_K_M bumps every other
    // layer's attn_v + ffn_down to Q6_K, so they vary across the stack. When
    // w_v is populated, V always lives in its own buffer (dt per dt_v_layer).
    sk::Dtype dt_qkv     = sk::Dtype::F16;
    sk::Dtype dt_o       = sk::Dtype::F16;
    sk::Dtype dt_gate    = sk::Dtype::F16;
    sk::Dtype dt_up      = sk::Dtype::F16;
    sk::Dtype dt_lm_head = sk::Dtype::F16;
    std::vector<sk::Dtype> dt_v_layer;     // size n_layers when w_v split; else empty
    std::vector<sk::Dtype> dt_down_layer;  // size n_layers
};

struct ModelBuffers {
    MTL::Buffer* input_ids;
    MTL::Buffer* output_id;
    MTL::Buffer* x_a;
    MTL::Buffer* x_b;
    MTL::Buffer* logits;
    MTL::Buffer* rope_pos;
    MTL::Buffer* cos_tbl;
    MTL::Buffer* sin_tbl;

    MTL::Buffer* x_norm;
    MTL::Buffer* qkv_packed;
    MTL::Buffer* q;
    MTL::Buffer* kv_pack;
    MTL::Buffer* k_tmp;
    MTL::Buffer* v_tmp;
    MTL::Buffer* attn_out;
    MTL::Buffer* o_proj;
    MTL::Buffer* y_attn;
    MTL::Buffer* m_in;
    MTL::Buffer* mlp_out;
    MTL::Buffer* capture;  // optional snapshot buffer (T, d_model) fp16

    // MLP scratch: gate and up projections, each (T, n_int).
    MTL::Buffer* gate_buf;
    MTL::Buffer* up_buf;

    // Head-major (H, T, D) scratch for Q, K, V and a seq-major attn output.
    MTL::Buffer* q_th;
    MTL::Buffer* k_th;
    MTL::Buffer* v_th;
    MTL::Buffer* attn_out_seq;

    // Flash-decoding split-K partials (decode only). Sized batch*n_heads*SPLITS.
    MTL::Buffer* attn_pm = nullptr;
    MTL::Buffer* attn_ps = nullptr;
    MTL::Buffer* attn_po = nullptr;

    // 2-pass argmax scratch (n_blocks partials). Sized at handle-creation
    // time as ceil(vocab_size / 16384). May be nullptr when 2-pass disabled.
    MTL::Buffer* argmax_val_buf = nullptr;
    MTL::Buffer* argmax_idx_buf = nullptr;

    // ICB-tail wiring (decode T=1 only). All three are non-null together or
    // all null. argmax_args holds two uint32_t scalars laid out as
    //   [0..4)  : vocab_size  (read by argmax_partial at buffer(3))
    //   [4..8)  : n_blocks    (read by argmax_reduce  at buffer(3))
    // setBytes is forbidden inside ICB-recorded commands, so kernel scalars
    // come from this real MTL::Buffer instead. Values are written once at
    // sk_qwen_create() time (vocab_size and n_blocks are model-static).
    MTL::Buffer*                argmax_args = nullptr;
    sk::silicon::IcbRecorder*   argmax_icb  = nullptr;
};

inline void dispatch_model(
    MTL::CommandBuffer* cmd,
    const ModelPSOs&    P,
    const ModelWeights& W,
    ModelBuffers&       B,
    const ModelParams&  M)
{
    const uint32_t T = M.batch * M.seq;

    // A. Embedding lookup → x_a
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.embedding_lookup);
        enc->setBuffer(W.w_embed,   0, 0);
        enc->setBuffer(B.input_ids, 0, 1);
        enc->setBuffer(B.x_a,       0, 2);
        enc->setBytes(&T,            4, 3);
        enc->setBytes(&M.d_model,    4, 4);
        enc->setBytes(&M.vocab_size, 4, 5);
        const uint32_t D4 = M.d_model / 4;
        enc->dispatchThreadgroups(MTL::Size((D4 + 127) / 128, T, 1),
                                  MTL::Size(128, 1, 1));
        enc->endEncoding();
    }

    // B. Layer stack (ping-pong x_a ↔ x_b)
    MTL::Buffer* cur = B.x_a;
    MTL::Buffer* nxt = B.x_b;

    const uint32_t n_run = (M.layers_run > 0 && M.layers_run < M.n_layers)
                           ? M.layers_run : M.n_layers;
    for (uint32_t L = 0; L < n_run; ++L) {
        // KV-cache addressing
        const uint32_t total_after   = M.current_pos + M.seq;
        const uint32_t kv_len        = (total_after < M.cache_max) ? total_after : M.cache_max;
        const uint32_t logical_first = total_after - kv_len;
        const uint32_t kv_buf_start  = logical_first % M.cache_max;

        LayerParams lp;
        lp.batch        = M.batch;
        lp.seq          = M.seq;
        lp.d_model      = M.d_model;
        lp.n_heads      = M.n_heads;
        lp.n_kv_heads   = M.n_kv_heads;
        lp.head_dim     = M.head_dim;
        lp.n_int        = M.n_int;
        lp.eps          = M.eps;
        lp.use_qk_norm  = M.use_qk_norm;
        lp.rope_interleaved = M.rope_interleaved;
        lp.attn_qkv_bias = M.attn_qkv_bias;
        lp.layer_idx    = L;
        lp.kv_buf_start = kv_buf_start;
        lp.kv_len       = kv_len;
        lp.cache_size   = M.cache_max;
        lp.write_pos    = M.current_pos;
        lp.rope_freq_base   = M.rope_freq_base;
        lp.rope_n_ctx_orig  = M.rope_n_ctx_orig;
        lp.rope_freq_scale  = M.rope_freq_scale;
        lp.rope_ext_factor  = M.rope_ext_factor;
        lp.rope_attn_factor = M.rope_attn_factor;
        lp.rope_beta_fast   = M.rope_beta_fast;
        lp.rope_beta_slow   = M.rope_beta_slow;

        LayerBuffers lb{};
        lb.x = cur;
        lb.w_pre_attn_norm = W.w_pre_attn_norm;
        lb.w_qkv             = W.w_qkv[L];
        lb.w_qkv_inner_off   = W.w_qkv_off[L];
        lb.w_qkv_bias        = W.w_qkv_bias;
        if (!W.w_v.empty()) {
            lb.w_v             = W.w_v[L];
            lb.w_v_inner_off   = W.w_v_off[L];
        }
        lb.dt_v   = W.dt_v_layer.empty()    ? W.dt_qkv : W.dt_v_layer[L];
        lb.dt_down= W.dt_down_layer.empty() ? sk::Dtype::F16 : W.dt_down_layer[L];
        lb.w_q_norm          = W.w_q_norm;
        lb.w_k_norm          = W.w_k_norm;
        lb.w_o               = W.w_o[L];
        lb.w_o_inner_off     = W.w_o_off[L];
        lb.w_pre_mlp_norm    = W.w_pre_mlp_norm;
        lb.w_gate            = W.w_gate[L];
        lb.w_gate_inner_off  = W.w_gate_off[L];
        lb.w_up              = W.w_up[L];
        lb.w_up_inner_off    = W.w_up_off[L];
        lb.w_down            = W.w_down[L];
        lb.w_down_inner_off  = W.w_down_off[L];
        lb.rope_pos        = B.rope_pos;
        lb.cos_tbl         = B.cos_tbl;
        lb.sin_tbl         = B.sin_tbl;
        lb.k_cache         = W.layer_caches[L].k;
        lb.v_cache         = W.layer_caches[L].v;
        lb.k_cache_q       = W.layer_caches[L].kq;
        lb.v_cache_q       = W.layer_caches[L].vq;
        lb.k_cache_s       = W.layer_caches[L].ks;
        lb.v_cache_s       = W.layer_caches[L].vs;
        lb.kv_q8           = M.kv_q8;
        lb.x_norm          = B.x_norm;
        lb.qkv_packed      = B.qkv_packed;
        lb.q               = B.q;
        lb.kv_pack         = B.kv_pack;
        lb.k_tmp           = B.k_tmp;
        lb.v_tmp           = B.v_tmp;
        lb.attn_out        = B.attn_out;
        lb.o_proj          = B.o_proj;
        lb.y_attn          = B.y_attn;
        lb.m_in            = B.m_in;
        lb.mlp_out         = B.mlp_out;
        lb.y_out           = nxt;
        lb.gate_buf        = B.gate_buf;
        lb.up_buf          = B.up_buf;
        lb.q_th            = B.q_th;
        lb.k_th            = B.k_th;
        lb.v_th            = B.v_th;
        lb.attn_out_seq    = B.attn_out_seq;
        lb.attn_pm         = B.attn_pm;
        lb.attn_ps         = B.attn_ps;
        lb.attn_po         = B.attn_po;
        lb.dt_qkv          = W.dt_qkv;
        lb.dt_o            = W.dt_o;
        lb.dt_gate         = W.dt_gate;
        lb.dt_up           = W.dt_up;

        dispatch_layer(cmd, P.layer, lb, lp);

        // Optional: snapshot this layer's residual output into capture buffer.
        if (B.capture && M.capture_layer >= 0 && (int32_t)L == M.capture_layer) {
            auto* benc = cmd->blitCommandEncoder();
            benc->copyFromBuffer(nxt, 0, B.capture, 0,
                                 (size_t)T * M.d_model * 2);
            benc->endEncoding();
        }

        MTL::Buffer* tmp = cur; cur = nxt; nxt = tmp;
    }

    // Interior batched-prefill chunk: KV is written, no logits consumer yet.
    if (M.batched_prefill == 1u) return;

    // C. Final RMSNorm — own encoder (single helper, not worth fusing here).
    {
        auto* enc = cmd->computeCommandEncoder();
        encode_rmsnorm(enc, P.layer.rmsnorm, cur, W.w_final_norm, 0,
                       nxt, T, M.d_model, M.eps, P.layer.rmsnorm_t1);
        enc->endEncoding();
    }

    // Final batched-prefill chunk: per-lane last-row head + argmax (see the
    // batched_prefill field WHY). Rows are independent (disjoint outputs, the
    // shared head weight is read-only) so the lane matvecs need no barriers;
    // the encoder boundary orders head -> argmax.
    if (M.batched_prefill == 2u) {
        MTL::Buffer* w_head = W.w_lm_head ? W.w_lm_head : W.w_embed;
        const size_t off_head = W.w_lm_head ? W.off_w_lm_head : 0;
        MTL::ComputePipelineState* qpso =
            W.w_lm_head ? quant_matvec_pso(P.layer, W.dt_lm_head) : nullptr;
        const uint32_t K_v = M.d_model, N_v = M.vocab_size;
        {
            auto* enc = cmd->computeCommandEncoder();
            for (uint32_t b = 0; b < M.batch; ++b) {
                const size_t off_A = ((size_t)b * M.seq + M.seq - 1u) * K_v * 2;
                const size_t off_C = (size_t)b * N_v * 2;
                if (qpso) {
                    enc->setComputePipelineState(qpso);
                    enc->setBuffer(nxt,      off_A,    0);
                    enc->setBuffer(w_head,   off_head, 1);
                    enc->setBuffer(B.logits, off_C,    2);
                    enc->setBytes(&K_v, 4, 3);
                    enc->setBytes(&N_v, 4, 4);
                    enc->dispatchThreadgroups(MTL::Size((N_v + 1) / 2, 1, 1),
                                              MTL::Size(128, 1, 1));
                } else if (P.layer.gemv_t_m1 != nullptr) {
                    const bool use_2d = (P.layer.gemv_t_2dtile_m1 != nullptr);
                    enc->setComputePipelineState(use_2d ? P.layer.gemv_t_2dtile_m1
                                                        : P.layer.gemv_t_m1);
                    enc->setBuffer(nxt,      off_A,    0);
                    enc->setBuffer(w_head,   off_head, 1);
                    enc->setBuffer(B.logits, off_C,    2);
                    enc->setBytes(&N_v, 4, 3);
                    enc->setBytes(&K_v, 4, 4);
                    if (use_2d) {
                        const uint32_t OUT_ROWS_PER_TG = 64;
                        enc->dispatchThreadgroups(
                            MTL::Size((N_v + OUT_ROWS_PER_TG - 1) / OUT_ROWS_PER_TG, 1, 1),
                            MTL::Size(32, 16, 1));
                    } else {
                        const uint32_t BN = 128;
                        enc->dispatchThreadgroups(MTL::Size((N_v + BN - 1) / BN, 1, 1),
                                                  MTL::Size(BN, 1, 1));
                    }
                } else {
                    const uint32_t M_v = 1u;
                    uint32_t ldA = K_v, ldB = K_v, ldC = N_v;
                    int transA = 0, transB = 1, has_bias = 0;
                    enc->setComputePipelineState(P.layer.gemm);
                    enc->setBuffer(nxt,      off_A,    0);
                    enc->setBuffer(w_head,   off_head, 1);
                    enc->setBuffer(B.logits, off_C,    2);
                    enc->setBytes(&M_v,      4, 3); enc->setBytes(&N_v,      4, 4);
                    enc->setBytes(&K_v,      4, 5); enc->setBytes(&ldA,      4, 6);
                    enc->setBytes(&ldB,      4, 7); enc->setBytes(&ldC,      4, 8);
                    enc->setBytes(&transA,   4, 9); enc->setBytes(&transB,   4, 10);
                    enc->setBytes(&has_bias, 4, 11);
                    enc->setBuffer(B.logits, off_C, 12);
                    enc->dispatchThreadgroups(MTL::Size((N_v + 63) / 64, 1, 1),
                                              MTL::Size(64, 1, 1));
                }
            }
            enc->endEncoding();
        }
        {
            auto* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(P.argmax);
            for (uint32_t b = 0; b < M.batch; ++b) {
                enc->setBuffer(B.logits,    (size_t)b * N_v * 2, 0);
                enc->setBuffer(B.output_id, (size_t)b * sizeof(int32_t), 1);
                enc->setBytes(&N_v, 4, 2);
                enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1024, 1, 1));
            }
            enc->endEncoding();
        }
        return;
    }

    // D. LM-head GEMM (tied → reuse embedding; untied → use w_lm_head). Both (V,D) row-major → transB=1.
    //
    // Generation reads ONLY the last position's logits. At prefill (T>1) the
    // head projects just row T-1; projecting all T rows would be vocab×d_model
    // of dead work AND a T-wide argmax would write output_id[1..T-1] past the
    // batch=1 buffer (OOB — the recurring logits-sizing bug). The result is
    // written to logits row T-1 so get_last_logits (reads logits row
    // last_seq-1 == T-1) and the argmax below both land on it.
    // Batched-decode (decode_all_rows): project every one of the T=batch rows so
    // each request gets its own logits row. The M=1 / prefill default projects
    // only row T-1 (row_lo=T-1). One matvec per row reuses the same head weight
    // read across rows just like the per-row decode it mirrors.
    const uint32_t head_row_lo = M.decode_all_rows ? 0u : (T - 1u);
    // Batched-decode LM-head amortization: a per-row loop re-reads the (large)
    // head weight T times — for a tied small model the head is ~25% of weight
    // bytes, so the serial loop is the throughput-scaling bottleneck. When an MMA
    // GEMM exists for the head dtype, run ONE [T,K]·[N,K] GEMM so the head weight
    // is read once and applied to all T rows (the same bandwidth-amortization the
    // layer projections get from gemm_sm). Falls back to the per-row loop below
    // when no MMA path is available for the dtype.
    bool head_batched = false;
    if (M.decode_all_rows && T > 1u) {
        MTL::Buffer* w_head = W.w_lm_head ? W.w_lm_head : W.w_embed;
        const size_t off_head = W.w_lm_head ? W.off_w_lm_head : 0;
        MTL::ComputePipelineState* head_mma =
            (W.dt_lm_head == sk::Dtype::Q8_0 && W.w_lm_head) ? P.layer.gemm_mma_q8_0
            : (W.dt_lm_head == sk::Dtype::Q4_K && W.w_lm_head) ? P.layer.gemm_mma_q4k
            : (W.dt_lm_head == sk::Dtype::F16 || !W.w_lm_head) ? P.layer.gemm_mma_f16
            : nullptr;
        MTL::ComputePipelineState* head_sm =
            (W.dt_lm_head == sk::Dtype::Q8_0 && W.w_lm_head) ? P.layer.gemm_mma_q8_0_sm
            : nullptr;
        const uint32_t K_v = M.d_model, N_v = M.vocab_size;
        if (head_sm && gemm_sm_wins(W.dt_lm_head, T)) {
            auto* enc = cmd->computeCommandEncoder();
            encode_gemm_sm(enc, head_sm, nxt, 0, w_head, off_head,
                           B.logits, 0, T, N_v, K_v, N_v,
                           gemm_sm_threads(W.dt_lm_head));
            enc->endEncoding();
            head_batched = true;
        } else if (head_mma) {
            auto* enc = cmd->computeCommandEncoder();
            encode_gemm_mma(enc, head_mma, nxt, 0, w_head, off_head,
                            B.logits, 0, T, N_v, K_v, N_v);
            enc->endEncoding();
            head_batched = true;
        }
    }
    for (uint32_t last = head_row_lo; !head_batched && last < T; ++last) {
        const uint32_t M_v = 1u, K_v = M.d_model, N_v = M.vocab_size;
        const size_t   off_A = (size_t)last * K_v * 2;
        const size_t   off_C = (size_t)last * N_v * 2;
        MTL::Buffer* w_head = W.w_lm_head ? W.w_lm_head : W.w_embed;

        if (W.dt_lm_head == sk::Dtype::Q8_0 && W.w_lm_head &&
            P.layer.q8_0_matvec != nullptr) {
            auto* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(P.layer.q8_0_matvec);
            enc->setBuffer(nxt,           off_A,           0);
            enc->setBuffer(W.w_lm_head,   W.off_w_lm_head, 1);
            enc->setBuffer(B.logits,      off_C,           2);
            enc->setBytes(&K_v, 4, 3);
            enc->setBytes(&N_v, 4, 4);
            enc->dispatchThreadgroups(MTL::Size((N_v + 1) / 2, 1, 1), MTL::Size(128, 1, 1));
            enc->endEncoding();
        } else
        if (W.dt_lm_head == sk::Dtype::Q4_K && W.w_lm_head &&
            P.layer.q4k_matvec != nullptr) {
            // SK_QWEN_Q4K_HEAD: head requantized Q8_0→Q4_K at load (weights.c++).
            // q4k_matvec shares the q8_0_matvec binding (B=0,A=1,C=2,K=3,N=4) and
            // NR0=2 geometry, so this is the Q8 branch with the Q4_K PSO. Half the
            // head bytes on the bandwidth-bound largest decode matvec.
            auto* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(P.layer.q4k_matvec);
            enc->setBuffer(nxt,           off_A,           0);
            enc->setBuffer(W.w_lm_head,   W.off_w_lm_head, 1);
            enc->setBuffer(B.logits,      off_C,           2);
            enc->setBytes(&K_v, 4, 3);
            enc->setBytes(&N_v, 4, 4);
            enc->dispatchThreadgroups(MTL::Size((N_v + 1) / 2, 1, 1), MTL::Size(128, 1, 1));
            enc->endEncoding();
        } else
        if (W.dt_lm_head == sk::Dtype::Q6_K && W.w_lm_head &&
            P.layer.q6k_matvec != nullptr) {
            // Q4_K_M's output.weight is Q6_K; matvec straight from the quant
            // bytes (vocab·d_model) instead of a host-dequanted fp16 head.
            auto* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(P.layer.q6k_matvec);
            enc->setBuffer(nxt,           off_A,           0);
            enc->setBuffer(W.w_lm_head,   W.off_w_lm_head, 1);
            enc->setBuffer(B.logits,      off_C,           2);
            enc->setBytes(&K_v, 4, 3);
            enc->setBytes(&N_v, 4, 4);
            enc->dispatchThreadgroups(MTL::Size((N_v + 1) / 2, 1, 1), MTL::Size(128, 1, 1));
            enc->endEncoding();
        } else
        if (P.layer.gemv_t_m1 != nullptr) {
            // M=1 transposed-weight matvec (head input is the single last row).
            // Prefer 2D-tile variant when registered; same buffer signature.
            const size_t off_head = (w_head == W.w_lm_head) ? W.off_w_lm_head : 0;
            auto* enc = cmd->computeCommandEncoder();
            const bool use_2d = (P.layer.gemv_t_2dtile_m1 != nullptr);
            enc->setComputePipelineState(use_2d ? P.layer.gemv_t_2dtile_m1
                                                : P.layer.gemv_t_m1);
            enc->setBuffer(nxt,      off_A,    0);
            enc->setBuffer(w_head,   off_head, 1);
            enc->setBuffer(B.logits, off_C,    2);
            enc->setBytes(&N_v, 4, 3);
            enc->setBytes(&K_v, 4, 4);
            if (use_2d) {
                // SG_ROWS=16, TOR=4 → 64 outputs / TG; TG = (32, 16, 1) = 512 threads.
                const uint32_t OUT_ROWS_PER_TG = 64;
                enc->dispatchThreadgroups(
                    MTL::Size((N_v + OUT_ROWS_PER_TG - 1) / OUT_ROWS_PER_TG, 1, 1),
                    MTL::Size(32, 16, 1));
            } else {
                const uint32_t BN = 128;
                enc->dispatchThreadgroups(MTL::Size((N_v + BN - 1) / BN, 1, 1),
                                          MTL::Size(BN, 1, 1));
            }
            enc->endEncoding();
        } else {
            const size_t off_head = (w_head == W.w_lm_head) ? W.off_w_lm_head : 0;
            auto* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(P.layer.gemm);
            uint32_t ldA = K_v, ldB = K_v, ldC = N_v;
            int transA = 0, transB = 1, has_bias = 0;
            enc->setBuffer(nxt,        off_A,    0);
            enc->setBuffer(w_head,     off_head, 1);
            enc->setBuffer(B.logits,   off_C,    2);
            enc->setBytes(&M_v,      4, 3); enc->setBytes(&N_v,      4, 4);
            enc->setBytes(&K_v,      4, 5); enc->setBytes(&ldA,      4, 6);
            enc->setBytes(&ldB,      4, 7); enc->setBytes(&ldC,      4, 8);
            enc->setBytes(&transA,   4, 9); enc->setBytes(&transB,   4, 10);
            enc->setBytes(&has_bias, 4, 11);
            enc->setBuffer(B.logits, off_C, 12);
            enc->dispatchThreadgroups(MTL::Size((N_v + 63) / 64, 1, 1),
                                      MTL::Size(64, 1, 1));
            enc->endEncoding();
        }
    }

    // E. Argmax → output_id
    //
    // Fast path: T==1 (decode) + 2-pass PSOs available → parallel block-reduce
    // argmax (≈3.7× faster at V=151936 on M4). Otherwise fall back to the
    // single-TG `argmax` PSO that handles arbitrary T.
    const bool can_2pass = (T == 1u)
                        && P.argmax_partial && P.argmax_reduce
                        && B.argmax_val_buf && B.argmax_idx_buf;
    const bool can_icb_tail = can_2pass
                        && P.argmax_partial_icb && P.argmax_reduce_icb
                        && B.argmax_args && B.argmax_icb;
    if (can_icb_tail) {
        // Pre-recorded ICB: two dispatches (argmax_partial → argmax_reduce)
        // in a single executeCommandsInBuffer call. Producer/consumer
        // ordering relies on IcbRecorder::record's default barrier_before=true
        // on slot 1 (which reads val/idx written by slot 0).
        auto* enc = cmd->computeCommandEncoder();
        B.argmax_icb->execute(enc, 0, 2);
        // logits and output_id are encoder-touched resources for this graph;
        // the recorder's tracked list already includes them via mark_resource
        // at record time.
        enc->endEncoding();
    } else if (can_2pass) {
        constexpr uint32_t ELTS_PER_TG = 16384u;
        const uint32_t n_blocks = (M.vocab_size + ELTS_PER_TG - 1u) / ELTS_PER_TG;
        // Pass 1: per-tile partials.
        {
            auto* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(P.argmax_partial);
            enc->setBuffer(B.logits,         0, 0);
            enc->setBuffer(B.argmax_val_buf, 0, 1);
            enc->setBuffer(B.argmax_idx_buf, 0, 2);
            enc->setBytes(&M.vocab_size,     4, 3);
            enc->dispatchThreadgroups(MTL::Size(n_blocks, 1, 1),
                                      MTL::Size(1024, 1, 1));
            enc->endEncoding();
        }
        // Pass 2: reduce partials → out[0].
        {
            auto* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(P.argmax_reduce);
            enc->setBuffer(B.argmax_val_buf, 0, 0);
            enc->setBuffer(B.argmax_idx_buf, 0, 1);
            enc->setBuffer(B.output_id,      0, 2);
            enc->setBytes(&n_blocks,         4, 3);
            enc->dispatchThreadgroups(MTL::Size(1, 1, 1),
                                      MTL::Size(1024, 1, 1));
            enc->endEncoding();
        }
    } else if (M.decode_all_rows && M.seq == 1u) {
        // Batched decode: argmax each request's logits row r → output_id[r].
        // One TG per row (the single-TG argmax PSO), all in one encoder.
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.argmax);
        for (uint32_t r = 0; r < T; ++r) {
            enc->setBuffer(B.logits,    (size_t)r * M.vocab_size * 2, 0);
            enc->setBuffer(B.output_id, (size_t)r * sizeof(int32_t), 1);
            enc->setBytes(&M.vocab_size, 4, 2);
            enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1024, 1, 1));
        }
        enc->endEncoding();
    } else {
        // Section D wrote only the last position's logits (row T-1). Argmax that
        // one row → output_id[0]. The logits base is offset to row T-1 so the
        // kernel's row=0 lands on it; dispatching T threadgroups would also
        // write output_id[1..T-1] past the batch=1 buffer (OOB).
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.argmax);
        enc->setBuffer(B.logits,    (size_t)(T - 1u) * M.vocab_size * 2, 0);
        enc->setBuffer(B.output_id, 0, 1);
        enc->setBytes(&M.vocab_size, 4, 2);
        enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1024, 1, 1));
        enc->endEncoding();
    }
}

// Pipeline-parallel layer-range dispatch (additive; dispatch_model above is
// byte-identical and unchanged). Runs a sub-window of the layer stack so the
// forward can be split across the layer dimension and hop the residual stream
// between hosts.
//   do_embed  : embed input_ids -> x_a before the loop (true only when start==0).
//               When false, the caller must have loaded the incoming hidden
//               state into B.x_a already (resume from a prior stage's output).
//   start/end : run layers [start, end). KV for each layer L is written at
//               write_pos=M.current_pos exactly as in dispatch_model.
//   do_tail   : final RMSNorm + LM head + argmax -> output_id (true only on the
//               last stage). When false, the residual after layer end-1 is left
//               in B.x_a (the loop always lands the result back in x_a so the
//               host copy-out / next-stage hand-in has a fixed buffer).
// On return, when do_tail==false the post-window residual stream (T*d_model fp16)
// is in B.x_a; when do_tail==true output_id holds the greedy next token.
inline void dispatch_layer_range(
    MTL::CommandBuffer* cmd,
    const ModelPSOs&    P,
    const ModelWeights& W,
    ModelBuffers&       B,
    const ModelParams&  M,
    uint32_t            start_layer,
    uint32_t            end_layer,
    bool                do_embed,
    bool                do_tail)
{
    const uint32_t T = M.batch * M.seq;

    if (do_embed) {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.embedding_lookup);
        enc->setBuffer(W.w_embed,   0, 0);
        enc->setBuffer(B.input_ids, 0, 1);
        enc->setBuffer(B.x_a,       0, 2);
        enc->setBytes(&T,            4, 3);
        enc->setBytes(&M.d_model,    4, 4);
        enc->setBytes(&M.vocab_size, 4, 5);
        const uint32_t D4 = M.d_model / 4;
        enc->dispatchThreadgroups(MTL::Size((D4 + 127) / 128, T, 1),
                                  MTL::Size(128, 1, 1));
        enc->endEncoding();
    }

    // Layer window: the incoming residual is in x_a (embed wrote it, or the
    // caller loaded the previous stage's hidden there). Ping-pong x_a<->x_b.
    MTL::Buffer* cur = B.x_a;
    MTL::Buffer* nxt = B.x_b;

    if (end_layer > M.n_layers) end_layer = M.n_layers;
    for (uint32_t L = start_layer; L < end_layer; ++L) {
        const uint32_t total_after   = M.current_pos + M.seq;
        const uint32_t kv_len        = (total_after < M.cache_max) ? total_after : M.cache_max;
        const uint32_t logical_first = total_after - kv_len;
        const uint32_t kv_buf_start  = logical_first % M.cache_max;

        LayerParams lp;
        lp.batch        = M.batch;
        lp.seq          = M.seq;
        lp.d_model      = M.d_model;
        lp.n_heads      = M.n_heads;
        lp.n_kv_heads   = M.n_kv_heads;
        lp.head_dim     = M.head_dim;
        lp.n_int        = M.n_int;
        lp.eps          = M.eps;
        lp.use_qk_norm  = M.use_qk_norm;
        lp.rope_interleaved = M.rope_interleaved;
        lp.layer_idx    = L;
        lp.kv_buf_start = kv_buf_start;
        lp.kv_len       = kv_len;
        lp.cache_size   = M.cache_max;
        lp.write_pos    = M.current_pos;
        lp.rope_freq_base   = M.rope_freq_base;
        lp.rope_n_ctx_orig  = M.rope_n_ctx_orig;
        lp.rope_freq_scale  = M.rope_freq_scale;
        lp.rope_ext_factor  = M.rope_ext_factor;
        lp.rope_attn_factor = M.rope_attn_factor;
        lp.rope_beta_fast   = M.rope_beta_fast;
        lp.rope_beta_slow   = M.rope_beta_slow;

        LayerBuffers lb{};
        lb.x = cur;
        lb.w_pre_attn_norm = W.w_pre_attn_norm;
        lb.w_qkv             = W.w_qkv[L];
        lb.w_qkv_inner_off   = W.w_qkv_off[L];
        if (!W.w_v.empty()) {
            lb.w_v             = W.w_v[L];
            lb.w_v_inner_off   = W.w_v_off[L];
        }
        lb.dt_v   = W.dt_v_layer.empty()    ? W.dt_qkv : W.dt_v_layer[L];
        lb.dt_down= W.dt_down_layer.empty() ? sk::Dtype::F16 : W.dt_down_layer[L];
        lb.w_q_norm          = W.w_q_norm;
        lb.w_k_norm          = W.w_k_norm;
        lb.w_o               = W.w_o[L];
        lb.w_o_inner_off     = W.w_o_off[L];
        lb.w_pre_mlp_norm    = W.w_pre_mlp_norm;
        lb.w_gate            = W.w_gate[L];
        lb.w_gate_inner_off  = W.w_gate_off[L];
        lb.w_up              = W.w_up[L];
        lb.w_up_inner_off    = W.w_up_off[L];
        lb.w_down            = W.w_down[L];
        lb.w_down_inner_off  = W.w_down_off[L];
        lb.rope_pos        = B.rope_pos;
        lb.cos_tbl         = B.cos_tbl;
        lb.sin_tbl         = B.sin_tbl;
        lb.k_cache         = W.layer_caches[L].k;
        lb.v_cache         = W.layer_caches[L].v;
        lb.k_cache_q       = W.layer_caches[L].kq;
        lb.v_cache_q       = W.layer_caches[L].vq;
        lb.k_cache_s       = W.layer_caches[L].ks;
        lb.v_cache_s       = W.layer_caches[L].vs;
        lb.kv_q8           = M.kv_q8;
        lb.x_norm          = B.x_norm;
        lb.qkv_packed      = B.qkv_packed;
        lb.q               = B.q;
        lb.kv_pack         = B.kv_pack;
        lb.k_tmp           = B.k_tmp;
        lb.v_tmp           = B.v_tmp;
        lb.attn_out        = B.attn_out;
        lb.o_proj          = B.o_proj;
        lb.y_attn          = B.y_attn;
        lb.m_in            = B.m_in;
        lb.mlp_out         = B.mlp_out;
        lb.y_out           = nxt;
        lb.gate_buf        = B.gate_buf;
        lb.up_buf          = B.up_buf;
        lb.q_th            = B.q_th;
        lb.k_th            = B.k_th;
        lb.v_th            = B.v_th;
        lb.attn_out_seq    = B.attn_out_seq;
        lb.attn_pm         = B.attn_pm;
        lb.attn_ps         = B.attn_ps;
        lb.attn_po         = B.attn_po;
        lb.dt_qkv          = W.dt_qkv;
        lb.dt_o            = W.dt_o;
        lb.dt_gate         = W.dt_gate;
        lb.dt_up           = W.dt_up;

        dispatch_layer(cmd, P.layer, lb, lp);

        MTL::Buffer* tmp = cur; cur = nxt; nxt = tmp;
    }

    // Land the post-window residual back in x_a so a non-tail stage exposes its
    // output at a fixed buffer for host copy-out (and a resume stage re-enters
    // here with its hidden in x_a). An odd layer count leaves cur==x_b; one blit
    // moves it to x_a. (When do_tail, the tail reads `cur` directly below.)
    if (!do_tail) {
        if (cur != B.x_a) {
            auto* benc = cmd->blitCommandEncoder();
            benc->copyFromBuffer(cur, 0, B.x_a, 0, (size_t)T * M.d_model * 2);
            benc->endEncoding();
        }
        return;
    }

    // Tail: final RMSNorm -> LM head (last row) -> argmax. Mirrors dispatch_model
    // sections C/D/E exactly; `cur` holds the final residual, `nxt` is scratch.
    {
        auto* enc = cmd->computeCommandEncoder();
        encode_rmsnorm(enc, P.layer.rmsnorm, cur, W.w_final_norm, 0,
                       nxt, T, M.d_model, M.eps, P.layer.rmsnorm_t1);
        enc->endEncoding();
    }
    {
        const uint32_t last = T - 1u;
        const uint32_t M_v = 1u, K_v = M.d_model, N_v = M.vocab_size;
        const size_t   off_A = (size_t)last * K_v * 2;
        const size_t   off_C = (size_t)last * N_v * 2;
        MTL::Buffer* w_head = W.w_lm_head ? W.w_lm_head : W.w_embed;

        if (W.dt_lm_head == sk::Dtype::Q8_0 && W.w_lm_head &&
            P.layer.q8_0_matvec != nullptr) {
            auto* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(P.layer.q8_0_matvec);
            enc->setBuffer(nxt,           off_A,           0);
            enc->setBuffer(W.w_lm_head,   W.off_w_lm_head, 1);
            enc->setBuffer(B.logits,      off_C,           2);
            enc->setBytes(&K_v, 4, 3);
            enc->setBytes(&N_v, 4, 4);
            enc->dispatchThreadgroups(MTL::Size((N_v + 1) / 2, 1, 1), MTL::Size(128, 1, 1));
            enc->endEncoding();
        } else
        if (W.dt_lm_head == sk::Dtype::Q6_K && W.w_lm_head &&
            P.layer.q6k_matvec != nullptr) {
            auto* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(P.layer.q6k_matvec);
            enc->setBuffer(nxt,           off_A,           0);
            enc->setBuffer(W.w_lm_head,   W.off_w_lm_head, 1);
            enc->setBuffer(B.logits,      off_C,           2);
            enc->setBytes(&K_v, 4, 3);
            enc->setBytes(&N_v, 4, 4);
            enc->dispatchThreadgroups(MTL::Size((N_v + 1) / 2, 1, 1), MTL::Size(128, 1, 1));
            enc->endEncoding();
        } else
        if (P.layer.gemv_t_m1 != nullptr) {
            const size_t off_head = (w_head == W.w_lm_head) ? W.off_w_lm_head : 0;
            auto* enc = cmd->computeCommandEncoder();
            const bool use_2d = (P.layer.gemv_t_2dtile_m1 != nullptr);
            enc->setComputePipelineState(use_2d ? P.layer.gemv_t_2dtile_m1
                                                : P.layer.gemv_t_m1);
            enc->setBuffer(nxt,      off_A,    0);
            enc->setBuffer(w_head,   off_head, 1);
            enc->setBuffer(B.logits, off_C,    2);
            enc->setBytes(&N_v, 4, 3);
            enc->setBytes(&K_v, 4, 4);
            if (use_2d) {
                const uint32_t OUT_ROWS_PER_TG = 64;
                enc->dispatchThreadgroups(
                    MTL::Size((N_v + OUT_ROWS_PER_TG - 1) / OUT_ROWS_PER_TG, 1, 1),
                    MTL::Size(32, 16, 1));
            } else {
                const uint32_t BN = 128;
                enc->dispatchThreadgroups(MTL::Size((N_v + BN - 1) / BN, 1, 1),
                                          MTL::Size(BN, 1, 1));
            }
            enc->endEncoding();
        } else {
            const size_t off_head = (w_head == W.w_lm_head) ? W.off_w_lm_head : 0;
            auto* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(P.layer.gemm);
            uint32_t ldA = K_v, ldB = K_v, ldC = N_v;
            int transA = 0, transB = 1, has_bias = 0;
            enc->setBuffer(nxt,        off_A,    0);
            enc->setBuffer(w_head,     off_head, 1);
            enc->setBuffer(B.logits,   off_C,    2);
            enc->setBytes(&M_v,      4, 3); enc->setBytes(&N_v,      4, 4);
            enc->setBytes(&K_v,      4, 5); enc->setBytes(&ldA,      4, 6);
            enc->setBytes(&ldB,      4, 7); enc->setBytes(&ldC,      4, 8);
            enc->setBytes(&transA,   4, 9); enc->setBytes(&transB,   4, 10);
            enc->setBytes(&has_bias, 4, 11);
            enc->setBuffer(B.logits, off_C, 12);
            enc->dispatchThreadgroups(MTL::Size((N_v + 63) / 64, 1, 1),
                                      MTL::Size(64, 1, 1));
            enc->endEncoding();
        }
    }
    {
        const bool can_2pass = (T == 1u)
                            && P.argmax_partial && P.argmax_reduce
                            && B.argmax_val_buf && B.argmax_idx_buf;
        const bool can_icb_tail = can_2pass
                            && P.argmax_partial_icb && P.argmax_reduce_icb
                            && B.argmax_args && B.argmax_icb;
        if (can_icb_tail) {
            auto* enc = cmd->computeCommandEncoder();
            B.argmax_icb->execute(enc, 0, 2);
            enc->endEncoding();
        } else if (can_2pass) {
            constexpr uint32_t ELTS_PER_TG = 16384u;
            const uint32_t n_blocks = (M.vocab_size + ELTS_PER_TG - 1u) / ELTS_PER_TG;
            {
                auto* enc = cmd->computeCommandEncoder();
                enc->setComputePipelineState(P.argmax_partial);
                enc->setBuffer(B.logits,         0, 0);
                enc->setBuffer(B.argmax_val_buf, 0, 1);
                enc->setBuffer(B.argmax_idx_buf, 0, 2);
                enc->setBytes(&M.vocab_size,     4, 3);
                enc->dispatchThreadgroups(MTL::Size(n_blocks, 1, 1),
                                          MTL::Size(1024, 1, 1));
                enc->endEncoding();
            }
            {
                auto* enc = cmd->computeCommandEncoder();
                enc->setComputePipelineState(P.argmax_reduce);
                enc->setBuffer(B.argmax_val_buf, 0, 0);
                enc->setBuffer(B.argmax_idx_buf, 0, 1);
                enc->setBuffer(B.output_id,      0, 2);
                enc->setBytes(&n_blocks,         4, 3);
                enc->dispatchThreadgroups(MTL::Size(1, 1, 1),
                                          MTL::Size(1024, 1, 1));
                enc->endEncoding();
            }
        } else {
            auto* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(P.argmax);
            enc->setBuffer(B.logits,    (size_t)(T - 1u) * M.vocab_size * 2, 0);
            enc->setBuffer(B.output_id, 0, 1);
            enc->setBytes(&M.vocab_size, 4, 2);
            enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1024, 1, 1));
            enc->endEncoding();
        }
    }
}

} // namespace qwen
} // namespace meow

#endif
