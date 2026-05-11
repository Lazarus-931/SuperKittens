//
//  deepseek_model.h — DeepSeek V4 Flash dispatch orchestrator.
//
//  Mirrors gemma4_model.h structure: dispatch_layer + dispatch_model,
//  header-only. Implements the MLA (Multi-head Latent Attention) form of
//  attention used by DeepSeek V2/V3/V4: Q and KV are projected through a
//  low-rank latent space, only the latent + the RoPE-rotated K positional
//  half are stored in the KV cache. This is what lets DS4 hit 1M context.
//
//  What this header implements (using existing wired kernels):
//    1. pre-attn RMSNorm                        ← rmsnorm
//    2. Q-down  (x → q_a, low-rank)             ← gemm_fp16
//    3. RMSNorm on q_a                           ← rmsnorm
//    4. Q-up    (q_a → q, full per-head)        ← gemm_fp16
//    5. KV-down (x → kv_a packed [c_kv | k_pe]) ← gemm_fp16
//    6. RMSNorm on c_kv                         ← rmsnorm
//    7. p-RoPE on k_pe (pos-encoded half of K)  ← kernel_dsv4_rope_tail_f32
//    8. p-RoPE on Q's pe-half                    ← kernel_dsv4_rope_tail_f32
//    9. Cache write: c_kv + k_pe → cache         ← kv_cache_write
//   10. K-up    (c_kv → k_no_pe per-head)       ← gemm_fp16   [TODO inside attn]
//   11. V-up    (c_kv → v per-head)             ← gemm_fp16   [TODO inside attn]
//   12. Flash attention                          ← kernel_flash_attn_ext_vec_*
//   13. Output projection + residual            ← gemm_fp16 + add
//   14. pre-mlp RMSNorm                         ← rmsnorm
//   15. Shared expert (always-on dense FFN)     ← gated_mlp
//   16. Routed expert MoE FFN                   ← moe_ffn (router + swiglu_pair + down_scatter)
//
//  The routed-expert path is fully wired. Steps 10–12 (K/V-up + flash_attn)
//  are stubbed: PSOs are loaded but the args struct + dispatch are pending
//  the flash_attn_ext_vec launcher. dispatch_attn marks them as `// TODO:`
//  and lays out the buffer plumbing.
//

#ifndef SUPERKITTENS_DEEPSEEK_MODEL_H
#define SUPERKITTENS_DEEPSEEK_MODEL_H

#include <Metal/Metal.hpp>
#include <cstdint>

#include "../../kernels/moe/moe_ffn.h"
#include <cmath>

// Pull math into the deepseek namespace cleanly so the inline FA scale stays
// readable. `std::sqrt` works since we include <cmath>.
namespace meow { namespace deepseek {
inline float metal_sqrt_safe(float x) { return std::sqrt(x); }
}}

namespace meow {
namespace deepseek {

// ─────────────────────────────────────────────────────────────────────
//  Layer level
// ─────────────────────────────────────────────────────────────────────

struct LayerParams {
    // Shape (DS4 V4 Flash typical numbers as defaults)
    uint32_t batch          = 1;
    uint32_t seq            = 1;

    uint32_t d_model        = 7168;
    uint32_t n_heads        = 128;
    uint32_t qk_nope_dim    = 128;        // K's non-rotated half
    uint32_t qk_rope_dim    = 64;         // K's RoPE-rotated half (also Q's)
    uint32_t v_head_dim     = 128;        // V's head dim (≠ K's full dim in MLA)
    // dk = qk_nope_dim + qk_rope_dim;    derived
    uint32_t q_lora_rank    = 1536;
    uint32_t kv_lora_rank   = 512;        // compressed KV dim — what gets cached
    uint32_t n_int          = 2048;       // per-routed-expert FFN intermediate
    uint32_t shared_n_int   = 2048;       // shared expert FFN intermediate
    uint32_t n_expert       = 256;
    uint32_t top_k          = 8;
    float    eps            = 1e-6f;
    MoeQuant moe_quant      = MoeQuant::FP16;

    // Per-call (varies per layer + decode position)
    uint32_t layer_idx      = 0;
    uint32_t kv_buf_start   = 0;
    uint32_t kv_len         = 1;
    uint32_t cache_size     = 8192;
    uint32_t write_pos      = 0;

    // RoPE
    int32_t  rope_n_ctx_orig = 4096;
    float    rope_freq_base  = 10000.f;
    float    rope_freq_scale = 1.f;
    float    rope_ext_factor = 0.f;
    float    rope_attn_factor = 1.f;
    float    rope_beta_fast  = 32.f;
    float    rope_beta_slow  = 1.f;
};

struct LayerPSOs {
    MTL::ComputePipelineState* rmsnorm;
    MTL::ComputePipelineState* gemm;                  // SK fp16 GEMM
    MTL::ComputePipelineState* rope_tail;             // kernel_dsv4_rope_tail_f32
    MTL::ComputePipelineState* flash_attn_vec;        // kernel_flash_attn_ext_vec_*
    MTL::ComputePipelineState* cast_h2f;              // fp16 → fp32 elt-wise
    MTL::ComputePipelineState* cast_f2h;              // fp32 → fp16 elt-wise
    MTL::ComputePipelineState* causal_mask_fill;      // per-forward mask gen
    MTL::ComputePipelineState* kv_up_pair;            // fused MLA K-up + V-up
    MTL::ComputePipelineState* split_packed;              // splits kv_a_packed → c_kv + k_pe
    MTL::ComputePipelineState* kv_cache_write;
    MTL::ComputePipelineState* add;
    MTL::ComputePipelineState* add_rmsnorm;           // fused residual+RMSNorm
    MTL::ComputePipelineState* gated_mlp;             // shared expert path

    MoeFfnPSOs moe;
};

struct LayerBuffers {
    // Per-token stream
    MTL::Buffer* x;

    // ── MLA weights (per-layer concatenated) ──
    MTL::Buffer* w_pre_attn_norm;     // (n_layers, d_model)
    MTL::Buffer* w_q_a;               // (n_layers, d_model, q_lora_rank)
    MTL::Buffer* w_q_a_norm;          // (n_layers, q_lora_rank) — γ for the post-down norm
    MTL::Buffer* w_q_b;               // (n_layers, q_lora_rank, n_heads*(qk_nope_dim+qk_rope_dim))
    MTL::Buffer* w_kv_a;              // (n_layers, d_model, kv_lora_rank + qk_rope_dim)
    MTL::Buffer* w_kv_a_norm;         // (n_layers, kv_lora_rank)
    MTL::Buffer* w_kv_b;              // (n_layers, kv_lora_rank, n_heads*(qk_nope_dim + v_head_dim))
    MTL::Buffer* w_o;                 // (n_layers, n_heads*v_head_dim, d_model)
    MTL::Buffer* w_pre_mlp_norm;      // (n_layers, d_model)

    // Shared expert weights
    MTL::Buffer* w_shared_gate;       // (n_layers, d_model, shared_n_int)
    MTL::Buffer* w_shared_up;         // (n_layers, d_model, shared_n_int)
    MTL::Buffer* w_shared_down;       // (n_layers, shared_n_int, d_model)

    // Routed-expert weights (per-expert; layer-strided)
    MTL::Buffer* w_router;            // (n_layers, d_model, n_expert)
    MTL::Buffer* w_gate;              // (n_layers, n_expert, n_int, d_model)
    MTL::Buffer* w_up;                // same
    MTL::Buffer* w_down;              // (n_layers, n_expert, d_model, n_int)

    // Position buffer for RoPE
    MTL::Buffer* rope_pos;

    // ── KV cache (compressed: c_kv + k_pe stored separately) ──
    MTL::Buffer* c_kv_cache;          // (cache_size, kv_lora_rank)         — compressed
    MTL::Buffer* k_pe_cache;          // (cache_size, qk_rope_dim)          — RoPE'd K-positional

    // Scratch (reused across layers)
    MTL::Buffer* x_norm;
    MTL::Buffer* q_a;                 // (T, q_lora_rank) post-down, post-norm
    MTL::Buffer* q_packed;            // (T, n_heads * (qk_nope_dim + qk_rope_dim)) fp16
    MTL::Buffer* q_packed_f32;        // same shape, fp32 — RoPE + flash_attn input
    MTL::Buffer* k_pe_f32;            // (T, qk_rope_dim) fp32 — RoPE input/output
    MTL::Buffer* attn_out_f32;        // (T, n_heads, v_head_dim) fp32 — flash_attn output
    MTL::Buffer* causal_mask;         // (seq_max, cache_max) fp16 — regenerated per forward
    MTL::Buffer* kv_a_packed;         // (T, kv_lora_rank + qk_rope_dim)
    MTL::Buffer* c_kv;                // (T, kv_lora_rank)  — post-split, post-norm
    MTL::Buffer* k_pe;                // (T, qk_rope_dim)   — post-split, post-RoPE
    MTL::Buffer* k_no_pe;             // (T, n_heads, qk_nope_dim) — K-up output
    MTL::Buffer* v;                   // (T, n_heads, v_head_dim)
    MTL::Buffer* attn_out;            // (T, n_heads, v_head_dim)
    MTL::Buffer* o_proj;              // (T, d_model)
    MTL::Buffer* y_attn;              // (T, d_model) — post-attn residual

    MTL::Buffer* m_in;                // pre-MoE-norm
    MTL::Buffer* shared_mid;          // (T, shared_n_int)
    MTL::Buffer* shared_out;          // (T, d_model)
    MTL::Buffer* moe_top_idx;
    MTL::Buffer* moe_top_score;
    MTL::Buffer* moe_hidden;
    MTL::Buffer* y_out;               // (T, d_model) — final layer output
};

// Helper: encode a single fp16 GEMM (NN, no bias).
inline void encode_gemm(
    MTL::CommandBuffer* cmd, MTL::ComputePipelineState* pso,
    MTL::Buffer* A, size_t off_A,
    MTL::Buffer* B, size_t off_B,
    MTL::Buffer* C,
    uint32_t M, uint32_t N, uint32_t K)
{
    auto* enc = cmd->computeCommandEncoder();
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
    enc->dispatchThreadgroups(MTL::Size((N + 63) / 64, (M + 63) / 64, 1),
                              MTL::Size(64, 1, 1));
    enc->endEncoding();
}

// Element-wise cast. Used to bridge fp16 GEMMs ↔ fp32 RoPE/flash_attn.
inline void encode_cast(
    MTL::CommandBuffer* cmd, MTL::ComputePipelineState* pso,
    MTL::Buffer* src, MTL::Buffer* dst, uint32_t n)
{
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(src, 0, 0);
    enc->setBuffer(dst, 0, 1);
    enc->setBytes(&n,   4, 2);
    enc->dispatchThreadgroups(MTL::Size((n + 127) / 128, 1, 1),
                              MTL::Size(128, 1, 1));
    enc->endEncoding();
}

inline void encode_rmsnorm(
    MTL::CommandBuffer* cmd, MTL::ComputePipelineState* pso,
    MTL::Buffer* x, MTL::Buffer* gamma, size_t off_gamma,
    MTL::Buffer* out, uint32_t rows, uint32_t n, float eps)
{
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(x,     0,         0);
    enc->setBuffer(gamma, off_gamma, 1);
    enc->setBuffer(out,   0,         2);
    enc->setBytes(&rows, 4, 3);
    enc->setBytes(&n,    4, 4);
    enc->setBytes(&eps,  4, 5);
    enc->dispatchThreadgroups(MTL::Size(1, (rows + 3) / 4, 1),
                              MTL::Size(128, 1, 1));
    enc->endEncoding();
}

// ─── MLA attention: skeleton with all GEMMs wired ───────────────────
inline void dispatch_attn(
    MTL::CommandBuffer*  cmd,
    const LayerPSOs&     P,
    const LayerBuffers&  B,
    const LayerParams&   p)
{
    const uint32_t T   = p.batch * p.seq;
    const uint32_t L   = p.layer_idx;
    const uint32_t dk  = p.qk_nope_dim + p.qk_rope_dim;

    // Per-layer byte offsets (fp16 = 2 bytes/scalar).
    const size_t off_norm        = (size_t)L * p.d_model * 2;
    const size_t off_w_q_a       = (size_t)L * p.d_model * p.q_lora_rank * 2;
    const size_t off_w_q_a_norm  = (size_t)L * p.q_lora_rank * 2;
    const size_t off_w_q_b       = (size_t)L * p.q_lora_rank * p.n_heads * dk * 2;
    const size_t off_w_kv_a      = (size_t)L * p.d_model * (p.kv_lora_rank + p.qk_rope_dim) * 2;
    const size_t off_w_kv_a_norm = (size_t)L * p.kv_lora_rank * 2;
    const size_t off_w_o         = (size_t)L * p.n_heads * p.v_head_dim * p.d_model * 2;

    // 1. Pre-attn RMSNorm
    encode_rmsnorm(cmd, P.rmsnorm, B.x, B.w_pre_attn_norm, off_norm,
                   B.x_norm, T, p.d_model, p.eps);

    // 2. Q-down: x_norm (T, d_model) @ W_q_a (d_model, q_lora_rank) → q_a (T, q_lora_rank)
    encode_gemm(cmd, P.gemm, B.x_norm, 0, B.w_q_a, off_w_q_a, B.q_a,
                T, p.q_lora_rank, p.d_model);

    // 3. RMSNorm on q_a (in-place stash).
    encode_rmsnorm(cmd, P.rmsnorm, B.q_a, B.w_q_a_norm, off_w_q_a_norm,
                   B.q_a, T, p.q_lora_rank, p.eps);

    // 4. Q-up: q_a (T, q_lora_rank) @ W_q_b → q_packed (T, n_heads * dk)
    encode_gemm(cmd, P.gemm, B.q_a, 0, B.w_q_b, off_w_q_b, B.q_packed,
                T, p.n_heads * dk, p.q_lora_rank);

    // 5. KV-down: x_norm @ W_kv_a → kv_a_packed (T, kv_lora_rank + qk_rope_dim)
    encode_gemm(cmd, P.gemm, B.x_norm, 0, B.w_kv_a, off_w_kv_a, B.kv_a_packed,
                T, p.kv_lora_rank + p.qk_rope_dim, p.d_model);

    // 6a. Split kv_a_packed → c_kv (T, kv_lora_rank) + k_pe (T, qk_rope_dim).
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.split_packed);
        enc->setBuffer(B.kv_a_packed, 0, 0);
        enc->setBuffer(B.c_kv,        0, 1);
        enc->setBuffer(B.k_pe,        0, 2);
        enc->setBytes(&T,             4, 3);
        enc->setBytes(&p.kv_lora_rank, 4, 4);
        enc->setBytes(&p.qk_rope_dim,  4, 5);
        const uint32_t tot = p.kv_lora_rank + p.qk_rope_dim;
        enc->dispatchThreads(MTL::Size(tot, T, 1), MTL::Size(128, 1, 1));
        enc->endEncoding();
    }

    // 6b. RMSNorm c_kv in-place against w_kv_a_norm.
    encode_rmsnorm(cmd, P.rmsnorm, B.c_kv, B.w_kv_a_norm, off_w_kv_a_norm,
                   B.c_kv, T, p.kv_lora_rank, p.eps);

    // 7+8. p-RoPE on Q and K positional halves.
    //
    //   Q layout: q_packed (T, n_heads, dk) where dk = qk_nope_dim + qk_rope_dim.
    //             RoPE rotates the trailing qk_rope_dim per head; ne00 = dk,
    //             ne01 = T, ne02 = n_heads, ne03 = batch.
    //   K layout: k_pe (T, qk_rope_dim) — entire row IS the RoPE'd half.
    //             We rotate it as if ne00 = qk_rope_dim and n_dims = qk_rope_dim
    //             (rotate everything; no pass-through prefix).
    //
    // NOTE: dsv4_rope_tail_f32 expects float32 input. Q/K here are fp16. This
    //       is the one remaining type-mismatch in the dispatch — a fp16 variant
    //       of dsv4_rope_tail (or a half→float convert + RoPE + float→half) is
    //       the next concrete fix. We dispatch with the kernel as-is; output
    //       buffer alignment is correct, just dtype will be reinterpreted —
    //       caller must arrange for fp32 buffers or layer in a convert.
    {
        // Pack args for Q.
        // Mirror the ArgsRopeTail layout from rope_tail.c++.
        struct alignas(8) ArgsRopeTail {
            int64_t ne00, ne01, ne02, ne03;
            uint64_t nb00, nb01, nb02, nb03;
            uint64_t nb0,  nb1,  nb2,  nb3;
            int32_t n_dims, mode, n_ctx_orig, inverse;
            float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
            bool src2;
            char _pad[7];
        };
        ArgsRopeTail aq{};
        aq.ne00 = dk; aq.ne01 = p.seq; aq.ne02 = p.n_heads; aq.ne03 = p.batch;
        aq.nb00 = sizeof(float);
        aq.nb01 = aq.nb00 * dk;
        aq.nb02 = aq.nb01 * p.seq;
        aq.nb03 = aq.nb02 * p.n_heads;
        aq.nb0 = aq.nb00; aq.nb1 = aq.nb01; aq.nb2 = aq.nb02; aq.nb3 = aq.nb03;
        aq.n_dims     = (int32_t)p.qk_rope_dim;
        aq.mode       = 2;                              // NEOX
        aq.n_ctx_orig = p.rope_n_ctx_orig;
        aq.inverse    = 0;
        aq.freq_base  = p.rope_freq_base;
        aq.freq_scale = p.rope_freq_scale;
        aq.ext_factor = p.rope_ext_factor;
        aq.attn_factor = p.rope_attn_factor;
        aq.beta_fast  = p.rope_beta_fast;
        aq.beta_slow  = p.rope_beta_slow;
        aq.src2 = false;

        // Cast Q fp16 → fp32 (RoPE kernel reads/writes fp32).
        encode_cast(cmd, P.cast_h2f, B.q_packed, B.q_packed_f32,
                    T * p.n_heads * dk);

        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.rope_tail);
        enc->setBytes(&aq, sizeof(aq), 0);
        enc->setBuffer(B.q_packed_f32, 0, 1);   // src0 fp32
        enc->setBuffer(B.rope_pos,     0, 2);   // src1 = positions
        enc->setBuffer(B.q_packed_f32, 0, 3);   // src2 unused
        enc->setBuffer(B.q_packed_f32, 0, 4);   // dst fp32 (in-place)
        enc->dispatchThreadgroups(MTL::Size(p.seq, p.n_heads, p.batch),
                                  MTL::Size(256, 1, 1));
        enc->endEncoding();

        // k_pe: ne00 = qk_rope_dim, n_dims = qk_rope_dim (rotate all),
        //       ne01 = T, ne02 = 1, ne03 = 1 (K has no heads dim in this slot).
        ArgsRopeTail ak = aq;
        ak.ne00 = p.qk_rope_dim;
        ak.ne01 = p.seq * p.batch; ak.ne02 = 1; ak.ne03 = 1;
        ak.nb00 = sizeof(float);
        ak.nb01 = ak.nb00 * p.qk_rope_dim;
        ak.nb02 = ak.nb01 * ak.ne01;
        ak.nb03 = ak.nb02;
        ak.nb0 = ak.nb00; ak.nb1 = ak.nb01; ak.nb2 = ak.nb02; ak.nb3 = ak.nb03;
        ak.n_dims = (int32_t)p.qk_rope_dim;

        // Cast k_pe fp16 → fp32, RoPE in fp32, cast back to fp16.
        encode_cast(cmd, P.cast_h2f, B.k_pe, B.k_pe_f32,
                    T * p.qk_rope_dim);

        auto* enc2 = cmd->computeCommandEncoder();
        enc2->setComputePipelineState(P.rope_tail);
        enc2->setBytes(&ak, sizeof(ak), 0);
        enc2->setBuffer(B.k_pe_f32, 0, 1);
        enc2->setBuffer(B.rope_pos, 0, 2);
        enc2->setBuffer(B.k_pe_f32, 0, 3);
        enc2->setBuffer(B.k_pe_f32, 0, 4);
        enc2->dispatchThreadgroups(MTL::Size(p.seq * p.batch, 1, 1),
                                   MTL::Size(256, 1, 1));
        enc2->endEncoding();

        encode_cast(cmd, P.cast_f2h, B.k_pe_f32, B.k_pe,
                    T * p.qk_rope_dim);
    }

    // 9. Cache write: append c_kv + k_pe to the layer's compressed cache.
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.kv_cache_write);
        enc->setBuffer(B.kv_a_packed, 0, 0);
        enc->setBuffer(B.kv_a_packed, 0, 1);   // (no separate V — V is decompressed)
        enc->setBuffer(B.c_kv_cache,  0, 2);
        enc->setBuffer(B.k_pe_cache,  0, 3);
        const uint32_t one = 1;
        enc->setBytes(&p.batch,        4, 4);
        enc->setBytes(&one,            4, 5);
        enc->setBytes(&p.kv_lora_rank, 4, 6);
        enc->setBytes(&p.seq,          4, 7);
        enc->setBytes(&p.write_pos,    4, 8);
        enc->setBytes(&p.cache_size,   4, 9);
        const uint32_t D4 = p.kv_lora_rank / 4;
        enc->dispatchThreads(MTL::Size(D4, p.seq, p.batch),
                             MTL::Size(32, 4, 1));
        enc->endEncoding();
    }

    // 10+11. K-up + V-up — fused via kv_up_pair (decode-optimized, T=1 wins
    //        ~1.2× over two separate GEMMs). At prefill (T≥2) it currently
    //        loses; caller can detect T>1 and fall back to two encode_gemm
    //        calls. For now we always use the fused path — DS4 decode is
    //        the dominant scenario.
    //
    //        w_kv_b layout: [kv_lora_rank, n_heads*qk_nope_dim + n_heads*v_head_dim].
    //        W_k_up occupies the first (R * n_heads*qk_nope_dim) elts,
    //        W_v_up occupies the next (R * n_heads*v_head_dim).
    {
        const uint32_t k_out = p.n_heads * p.qk_nope_dim;
        const uint32_t v_out = p.n_heads * p.v_head_dim;

        const size_t layer_slab = (size_t)p.kv_lora_rank *
                                  (k_out + v_out) * 2;
        const size_t off_w_kv_b = (size_t)L * layer_slab;
        const size_t off_w_k_up = off_w_kv_b;
        const size_t off_w_v_up = off_w_kv_b +
                                  (size_t)p.kv_lora_rank * k_out * 2;

        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.kv_up_pair);
        enc->setBuffer(B.kv_a_packed, 0,           0);  // c_kv (first R cols)
        enc->setBuffer(B.w_kv_b,      off_w_k_up,  1);
        enc->setBuffer(B.w_kv_b,      off_w_v_up,  2);
        enc->setBuffer(B.k_no_pe,     0,           3);
        enc->setBuffer(B.v,           0,           4);
        enc->setBytes(&T,             4, 5);
        enc->setBytes(&p.kv_lora_rank, 4, 6);
        enc->setBytes(&k_out,         4, 7);
        enc->setBytes(&v_out,         4, 8);
        // v2 tile-MMA: BM=32, BN=64, 128 threads. Wins from T=1 through prefill.
        const uint32_t BM = 32, BN = 64;
        const uint32_t max_out = (k_out > v_out) ? k_out : v_out;
        enc->dispatchThreadgroups(
            MTL::Size((max_out + BN - 1) / BN, (T + BM - 1) / BM, 1),
            MTL::Size(128, 1, 1));
        enc->endEncoding();
    }

    // 12. Flash attention. Q is (T, n_heads, dk). K = [k_no_pe | k_pe] full.
    //     V = v (full). All bytes from current step; cached K/V handled by
    //     the kv_cache_write step above. The PSO must be resolved by the
    //     caller with function constants set: has_mask=true (causal mask
    //     bound at slot 4), has_sinks/has_bias/has_scap/has_kvpad=false,
    //     nsg=4, nwg=1, ns10=dk*sizeof(half), ns20=v_head_dim*sizeof(half).
    {
        #pragma pack(push, 8)
        struct ArgsFAVec {
            int32_t  ne01, ne02, ne03; char _p1[4];
            uint64_t nb01, nb02, nb03;
            int32_t  ne11, ne_12_2, ne_12_3, ns10;
            uint64_t nb11, nb12, nb13;
            int32_t  ns20; char _p2[4];
            uint64_t nb21, nb22, nb23;
            int32_t  ne31, ne32, ne33; char _p3[4];
            uint64_t nb31, nb32, nb33;
            int32_t  ne1, ne2, ne3;
            float    scale, max_bias, m0, m1;
            int32_t  n_head_log2;
            float    logit_softcap;
        };
        #pragma pack(pop)
        static_assert(sizeof(ArgsFAVec) == 192, "FA args mismatch");

        ArgsFAVec a{};
        a.ne01 = (int32_t)p.seq;     a.ne02 = (int32_t)p.n_heads; a.ne03 = (int32_t)p.batch;
        a.nb01 = (uint64_t)dk * sizeof(float);   // Q is fp32 in DS4 path
        a.nb02 = a.nb01 * p.seq;
        a.nb03 = a.nb02 * p.n_heads;
        a.ne11 = (int32_t)p.kv_len;  a.ne_12_2 = (int32_t)p.n_heads; a.ne_12_3 = (int32_t)p.batch;
        a.nb11 = (uint64_t)dk * sizeof(uint16_t);
        a.nb12 = a.nb11 * p.kv_len;
        a.nb13 = a.nb12 * p.n_heads;
        a.ns10 = (int32_t)a.nb11;
        a.nb21 = (uint64_t)p.v_head_dim * sizeof(uint16_t);
        a.nb22 = a.nb21 * p.kv_len;
        a.nb23 = a.nb22 * p.n_heads;
        a.ns20 = (int32_t)a.nb21;
        a.ne31 = (int32_t)p.seq; a.ne32 = 1; a.ne33 = 1;
        a.nb31 = (uint64_t)p.kv_len * sizeof(uint16_t);
        a.nb32 = a.nb31 * p.seq;
        a.nb33 = a.nb32;
        a.ne1 = (int32_t)p.n_heads; a.ne2 = (int32_t)p.seq; a.ne3 = (int32_t)p.batch;
        a.scale = 1.f / metal_sqrt_safe((float)dk);
        a.max_bias = 0.f; a.m0 = 1.f; a.m1 = 1.f;
        a.n_head_log2 = 0; a.logit_softcap = 0.f;

        // K and V here come from the kv caches; flash_attn reads them as the
        // dense decompressed full K, V. With kv_cache_write storing already-
        // up-projected K and V (the simple "cache full" path), B.k_cache and
        // B.v_cache are the right inputs. If MLA compressed cache is used
        // later, decompression must happen here first OR via absorption.
        // Fill causal mask for (q_seq, kv_len) with the current write_pos as
        // q_offset. Run before flash_attn, in the same command buffer.
        {
            auto* mask_enc = cmd->computeCommandEncoder();
            mask_enc->setComputePipelineState(P.causal_mask_fill);
            mask_enc->setBuffer(B.causal_mask, 0, 0);
            mask_enc->setBytes(&p.seq,         4, 1);
            mask_enc->setBytes(&p.kv_len,      4, 2);
            mask_enc->setBytes(&p.write_pos,   4, 3);
            mask_enc->dispatchThreads(MTL::Size(p.kv_len, p.seq, 1),
                                       MTL::Size(32, 4, 1));
            mask_enc->endEncoding();
        }

        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.flash_attn_vec);
        enc->setBytes(&a, sizeof(a), 0);
        enc->setBuffer(B.q_packed_f32, 0, 1);   // Q is fp32 in ds4 path
        enc->setBuffer(B.k_no_pe,       0, 2);   // K (fp16)
        enc->setBuffer(B.v,             0, 3);   // V (fp16)
        enc->setBuffer(B.causal_mask,   0, 4);   // causal mask (fp16, -inf above)
        enc->setBuffer(B.q_packed_f32,  0, 5);   // sinks (unused)
        enc->setBuffer(B.q_packed_f32,  0, 6);   // pad   (unused)
        enc->setBuffer(B.attn_out_f32,  0, 7);   // dst — flash_attn writes fp32

        enc->setThreadgroupMemoryLength(32 * 1024, 0);
        enc->dispatchThreadgroups(
            MTL::Size(p.seq, p.n_heads, p.batch),
            MTL::Size(32 * 4, 1, 1));   // nsg=4
        enc->endEncoding();

        // Cast attn_out fp32 → fp16 so the O-proj GEMM can read it.
        encode_cast(cmd, P.cast_f2h, B.attn_out_f32, B.attn_out,
                    T * p.n_heads * p.v_head_dim);
    }

    // 13. Output projection. Residual is folded into the next op
    //     (fused add_rmsnorm in dispatch_layer below — saves one HBM
    //     round-trip of d_model).
    encode_gemm(cmd, P.gemm, B.attn_out, 0, B.w_o, off_w_o, B.o_proj,
                T, p.d_model, p.n_heads * p.v_head_dim);
}

// ─── Shared expert (always-on dense FFN) ────────────────────────────
// DS4 routes top_k experts AND adds the output of one always-on shared
// expert. We compute the shared expert via the existing fused gated_mlp
// kernel and stash its output for the down_scatter step to combine.
inline void dispatch_shared_expert(
    MTL::CommandBuffer*  cmd,
    const LayerPSOs&     P,
    const LayerBuffers&  B,
    const LayerParams&   p)
{
    const uint32_t T = p.batch * p.seq;
    const uint32_t L = p.layer_idx;

    const size_t off_w_gate = (size_t)L * p.d_model * p.shared_n_int * 2;
    const size_t off_w_down = (size_t)L * p.shared_n_int * p.d_model * 2;

    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(P.gated_mlp);
    enc->setBuffer(B.m_in,          0,          0);
    enc->setBuffer(B.w_shared_gate, off_w_gate, 1);
    enc->setBuffer(B.w_shared_up,   off_w_gate, 2);
    enc->setBuffer(B.w_shared_down, off_w_down, 3);
    enc->setBuffer(B.shared_out,    0,          4);
    uint32_t M = T;
    uint32_t N_v = p.d_model;
    uint32_t K_v = p.d_model;
    enc->setBytes(&M,             4, 5);
    enc->setBytes(&N_v,           4, 6);
    enc->setBytes(&K_v,           4, 7);
    enc->setBytes(&p.shared_n_int, 4, 8);
    enc->dispatchThreadgroups(MTL::Size((N_v + 63) / 64, (M + 63) / 64, 1),
                              MTL::Size(128, 1, 1));
    enc->endEncoding();
}

// ─── Full layer ─────────────────────────────────────────────────────
inline void dispatch_layer(
    MTL::CommandBuffer*  cmd,
    const LayerPSOs&     P,
    const LayerBuffers&  B,
    const LayerParams&   p)
{
    dispatch_attn(cmd, P, B, p);

    const uint32_t T = p.batch * p.seq;
    const uint32_t L = p.layer_idx;
    const size_t   off_norm = (size_t)L * p.d_model * 2;

    // Fused: y_attn = x + o_proj;  m_in = RMSNorm(y_attn, w_pre_mlp_norm).
    // Replaces two kernels (add_f16 + rmsnorm) with one — 1.57–2.34× faster.
    {
        auto* enc = cmd->computeCommandEncoder();
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
        enc->endEncoding();
    }

    // Shared expert runs in parallel with router dispatch. Its output is
    // added inside down_scatter as part of the residual term.
    dispatch_shared_expert(cmd, P, B, p);

    // MoE FFN: router → swiglu_pair → down_scatter (residual baked in).
    // We pass shared_out + y_attn as the "residual" so down_scatter writes
    // y_out = y_attn + shared_out + sum_routed.
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.add);
        enc->setBuffer(B.y_attn,     0, 0);
        enc->setBuffer(B.shared_out, 0, 1);
        enc->setBuffer(B.shared_out, 0, 2);   // overwrite shared_out = y_attn + shared_out
        uint32_t n = T * p.d_model;
        enc->setBytes(&n, 4, 3);
        uint32_t total = (n / 4u) + (n & 3u);
        enc->dispatchThreadgroups(MTL::Size((total + 127) / 128, 1, 1),
                                  MTL::Size(128, 1, 1));
        enc->endEncoding();
    }

    MoeFfnBuffers MB{};
    MB.x         = B.m_in;            // routing input (post-norm)
    MB.w_router  = B.w_router;
    MB.w_gate    = B.w_gate;
    MB.w_up      = B.w_up;
    MB.w_down    = B.w_down;
    MB.top_idx   = B.moe_top_idx;
    MB.top_score = B.moe_top_score;
    MB.hidden    = B.moe_hidden;
    MB.residual  = B.shared_out;      // = y_attn + shared_out (precomputed above)
    MB.out       = B.y_out;

    MoeFfnParams mp{T, p.d_model, p.n_int, p.n_expert, p.top_k, p.moe_quant};
    dispatch_moe_ffn(cmd, P.moe, MB, mp);
}

// ─────────────────────────────────────────────────────────────────────
//  Model level
// ─────────────────────────────────────────────────────────────────────

struct ModelParams {
    uint32_t batch          = 1;
    uint32_t seq            = 1;
    uint32_t n_layers       = 60;
    uint32_t d_model        = 7168;
    uint32_t n_int          = 2048;
    uint32_t shared_n_int   = 2048;
    uint32_t n_heads        = 128;
    uint32_t qk_nope_dim    = 128;
    uint32_t qk_rope_dim    = 64;
    uint32_t v_head_dim     = 128;
    uint32_t q_lora_rank    = 1536;
    uint32_t kv_lora_rank   = 512;
    uint32_t n_expert       = 256;
    uint32_t top_k          = 8;
    uint32_t cache_max      = 8192;
    MoeQuant moe_quant      = MoeQuant::FP16;
    uint32_t vocab_size     = 129280;
    float    eps            = 1e-6f;
    uint32_t current_pos    = 0;

    // RoPE
    int32_t  rope_n_ctx_orig = 4096;
    float    rope_freq_base  = 10000.f;
    float    rope_freq_scale = 1.f;
    float    rope_ext_factor = 0.f;
    float    rope_attn_factor = 1.f;
    float    rope_beta_fast  = 32.f;
    float    rope_beta_slow  = 1.f;
};

struct ModelPSOs {
    LayerPSOs layer;
    MTL::ComputePipelineState* embedding_lookup;
    MTL::ComputePipelineState* argmax;
};

struct LayerCache {
    MTL::Buffer* c_kv;        // (cache_max, kv_lora_rank)
    MTL::Buffer* k_pe;        // (cache_max, qk_rope_dim)
};

struct ModelWeights {
    MTL::Buffer* w_embed;
    MTL::Buffer* w_pre_attn_norm;
    MTL::Buffer* w_q_a;
    MTL::Buffer* w_q_a_norm;
    MTL::Buffer* w_q_b;
    MTL::Buffer* w_kv_a;
    MTL::Buffer* w_kv_a_norm;
    MTL::Buffer* w_kv_b;
    MTL::Buffer* w_o;
    MTL::Buffer* w_pre_mlp_norm;
    MTL::Buffer* w_final_norm;
    MTL::Buffer* w_shared_gate;
    MTL::Buffer* w_shared_up;
    MTL::Buffer* w_shared_down;
    MTL::Buffer* w_router;
    MTL::Buffer* w_gate;
    MTL::Buffer* w_up;
    MTL::Buffer* w_down;
    const LayerCache* layer_caches;
};

struct ModelBuffers {
    MTL::Buffer* input_ids;
    MTL::Buffer* output_id;
    MTL::Buffer* x_a;
    MTL::Buffer* x_b;
    MTL::Buffer* logits;
    MTL::Buffer* rope_pos;

    // Layer scratch
    MTL::Buffer* x_norm;
    MTL::Buffer* q_a;
    MTL::Buffer* q_packed;
    MTL::Buffer* q_packed_f32;
    MTL::Buffer* k_pe_f32;
    MTL::Buffer* attn_out_f32;
    MTL::Buffer* causal_mask;
    MTL::Buffer* kv_a_packed;
    MTL::Buffer* c_kv;
    MTL::Buffer* k_pe;
    MTL::Buffer* k_no_pe;
    MTL::Buffer* v;
    MTL::Buffer* attn_out;
    MTL::Buffer* o_proj;
    MTL::Buffer* y_attn;
    MTL::Buffer* m_in;
    MTL::Buffer* shared_mid;
    MTL::Buffer* shared_out;
    MTL::Buffer* moe_top_idx;
    MTL::Buffer* moe_top_score;
    MTL::Buffer* moe_hidden;
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

    // B. Layer stack (ping-pong x_a ↔ x_b).
    MTL::Buffer* cur = B.x_a;
    MTL::Buffer* nxt = B.x_b;

    for (uint32_t L = 0; L < M.n_layers; ++L) {
        // KV-cache addressing for THIS step.
        const uint32_t total_after = M.current_pos + M.seq;
        const uint32_t kv_len      = (total_after < M.cache_max) ? total_after : M.cache_max;
        const uint32_t logical_first = total_after - kv_len;
        const uint32_t kv_buf_start  = logical_first % M.cache_max;

        LayerParams lp;
        lp.batch          = M.batch;
        lp.seq            = M.seq;
        lp.d_model        = M.d_model;
        lp.n_heads        = M.n_heads;
        lp.qk_nope_dim    = M.qk_nope_dim;
        lp.qk_rope_dim    = M.qk_rope_dim;
        lp.v_head_dim     = M.v_head_dim;
        lp.q_lora_rank    = M.q_lora_rank;
        lp.kv_lora_rank   = M.kv_lora_rank;
        lp.n_int          = M.n_int;
        lp.shared_n_int   = M.shared_n_int;
        lp.n_expert       = M.n_expert;
        lp.top_k          = M.top_k;
        lp.eps            = M.eps;
        lp.moe_quant      = M.moe_quant;
        lp.layer_idx      = L;
        lp.kv_buf_start   = kv_buf_start;
        lp.kv_len         = kv_len;
        lp.cache_size     = M.cache_max;
        lp.write_pos      = M.current_pos;
        lp.rope_n_ctx_orig = M.rope_n_ctx_orig;
        lp.rope_freq_base  = M.rope_freq_base;
        lp.rope_freq_scale = M.rope_freq_scale;
        lp.rope_ext_factor = M.rope_ext_factor;
        lp.rope_attn_factor = M.rope_attn_factor;
        lp.rope_beta_fast  = M.rope_beta_fast;
        lp.rope_beta_slow  = M.rope_beta_slow;

        LayerBuffers lb{};
        lb.x = cur;
        lb.w_pre_attn_norm = W.w_pre_attn_norm;
        lb.w_q_a           = W.w_q_a;
        lb.w_q_a_norm      = W.w_q_a_norm;
        lb.w_q_b           = W.w_q_b;
        lb.w_kv_a          = W.w_kv_a;
        lb.w_kv_a_norm     = W.w_kv_a_norm;
        lb.w_kv_b          = W.w_kv_b;
        lb.w_o             = W.w_o;
        lb.w_pre_mlp_norm  = W.w_pre_mlp_norm;
        lb.w_shared_gate   = W.w_shared_gate;
        lb.w_shared_up     = W.w_shared_up;
        lb.w_shared_down   = W.w_shared_down;
        lb.w_router        = W.w_router;
        lb.w_gate          = W.w_gate;
        lb.w_up            = W.w_up;
        lb.w_down          = W.w_down;
        lb.rope_pos        = B.rope_pos;
        lb.c_kv_cache      = W.layer_caches[L].c_kv;
        lb.k_pe_cache      = W.layer_caches[L].k_pe;

        lb.x_norm        = B.x_norm;
        lb.q_a           = B.q_a;
        lb.q_packed      = B.q_packed;
        lb.q_packed_f32  = B.q_packed_f32;
        lb.k_pe_f32      = B.k_pe_f32;
        lb.attn_out_f32  = B.attn_out_f32;
        lb.causal_mask   = B.causal_mask;
        lb.kv_a_packed   = B.kv_a_packed;
        lb.c_kv          = B.c_kv;
        lb.k_pe          = B.k_pe;
        lb.k_no_pe       = B.k_no_pe;
        lb.v             = B.v;
        lb.attn_out      = B.attn_out;
        lb.o_proj        = B.o_proj;
        lb.y_attn        = B.y_attn;
        lb.m_in          = B.m_in;
        lb.shared_mid    = B.shared_mid;
        lb.shared_out    = B.shared_out;
        lb.moe_top_idx   = B.moe_top_idx;
        lb.moe_top_score = B.moe_top_score;
        lb.moe_hidden    = B.moe_hidden;
        lb.y_out         = nxt;

        dispatch_layer(cmd, P.layer, lb, lp);

        MTL::Buffer* tmp = cur; cur = nxt; nxt = tmp;
    }

    // C. Final RMSNorm
    encode_rmsnorm(cmd, P.layer.rmsnorm, cur, W.w_final_norm, 0,
                   nxt, T, M.d_model, M.eps);

    // D. LM-head GEMM (tied with input embedding, so transB=1 against w_embed).
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.layer.gemm);
        const uint32_t M_v = T;
        const uint32_t K_v = M.d_model;
        const uint32_t N_v = M.vocab_size;
        uint32_t ldA = K_v, ldB = K_v, ldC = N_v;
        int transA = 0, transB = 1, has_bias = 0;
        enc->setBuffer(nxt,        0, 0);
        enc->setBuffer(W.w_embed,  0, 1);
        enc->setBuffer(B.logits,   0, 2);
        enc->setBytes(&M_v,      4, 3); enc->setBytes(&N_v,      4, 4);
        enc->setBytes(&K_v,      4, 5); enc->setBytes(&ldA,      4, 6);
        enc->setBytes(&ldB,      4, 7); enc->setBytes(&ldC,      4, 8);
        enc->setBytes(&transA,   4, 9); enc->setBytes(&transB,   4, 10);
        enc->setBytes(&has_bias, 4, 11);
        enc->setBuffer(B.logits, 0, 12);
        enc->dispatchThreadgroups(MTL::Size((N_v + 63) / 64, (M_v + 63) / 64, 1),
                                  MTL::Size(64, 1, 1));
        enc->endEncoding();
    }

    // E. Argmax → output_id
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.argmax);
        enc->setBuffer(B.logits,    0, 0);
        enc->setBuffer(B.output_id, 0, 1);
        enc->setBytes(&M.vocab_size, 4, 2);
        enc->dispatchThreadgroups(MTL::Size(T, 1, 1), MTL::Size(1024, 1, 1));
        enc->endEncoding();
    }
}

} // namespace deepseek
} // namespace meow

#endif
