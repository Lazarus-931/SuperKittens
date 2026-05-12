//  qwen_model.h — Qwen3-32B (dense) dispatch orchestrator.

#ifndef SUPERKITTENS_QWEN_MODEL_H
#define SUPERKITTENS_QWEN_MODEL_H

#include <Metal/Metal.hpp>
#include <cstdint>

namespace meow {
namespace qwen {

// ─── Layer level ─────────────────────────────────────────────────────

struct LayerParams {
    // Shape
    uint32_t batch        = 1;
    uint32_t seq          = 1;
    uint32_t d_model      = 5120;
    uint32_t n_heads      = 64;
    uint32_t n_kv_heads   = 8;
    uint32_t head_dim     = 128;
    uint32_t n_int        = 27392;
    float    eps          = 1e-6f;

    // Per-call (vary per layer + decode position)
    uint32_t layer_idx    = 0;
    uint32_t kv_buf_start = 0;
    uint32_t kv_len       = 1;
    uint32_t cache_size   = 32768;
    uint32_t write_pos    = 0;

    // RoPE (Qwen3: θ = 1,000,000; YaRN-corrected cos/sin tables baked host-side)
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
    MTL::ComputePipelineState* gemm;              // fp16 GEMM (M=1 → gemv fast-path)
    MTL::ComputePipelineState* split_packed;      // (T, A+B) → (T, A) + (T, B)
    MTL::ComputePipelineState* rope_qk;           // split-half RoPE on Q, K
    MTL::ComputePipelineState* attn;              // mha_causal (d=128, GQA)
    MTL::ComputePipelineState* kv_cache_write;
    MTL::ComputePipelineState* add;
    MTL::ComputePipelineState* add_rmsnorm;
    MTL::ComputePipelineState* gated_mlp;
};

struct LayerBuffers {
    MTL::Buffer* x;

    // Per-layer concatenated weights (offsets via layer_idx)
    MTL::Buffer* w_pre_attn_norm;     // (n_layers, d_model)
    MTL::Buffer* w_qkv;               // (n_layers, d_model, (n_heads + 2*n_kv_heads)*head_dim)
    MTL::Buffer* w_q_norm;            // (n_layers, head_dim) — per-head Q-norm γ
    MTL::Buffer* w_k_norm;            // (n_layers, head_dim) — per-head K-norm γ
    MTL::Buffer* w_o;                 // (n_layers, n_heads*head_dim, d_model)
    MTL::Buffer* w_pre_mlp_norm;      // (n_layers, d_model)
    MTL::Buffer* w_gate;              // (n_layers, d_model, n_int)
    MTL::Buffer* w_up;                // (n_layers, d_model, n_int)
    MTL::Buffer* w_down;              // (n_layers, n_int, d_model)

    MTL::Buffer* rope_pos;            // (seq,) int32 positions

    // Per-layer KV caches
    MTL::Buffer* k_cache;             // (cache_size, n_kv_heads, head_dim)
    MTL::Buffer* v_cache;

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
};

// ─── Inline encode helpers (mirror gemma4_model.h pattern) ───────────

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

// split_packed: (T, A+B) fp16 → (T, A) + (T, B). One thread per (t, c).
inline void encode_split(
    MTL::CommandBuffer* cmd, MTL::ComputePipelineState* pso,
    MTL::Buffer* src, MTL::Buffer* outA, MTL::Buffer* outB,
    uint32_t T, uint32_t A, uint32_t B)
{
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(src,  0, 0);
    enc->setBuffer(outA, 0, 1);
    enc->setBuffer(outB, 0, 2);
    enc->setBytes(&T, 4, 3);
    enc->setBytes(&A, 4, 4);
    enc->setBytes(&B, 4, 5);
    enc->dispatchThreads(MTL::Size(A + B, T, 1), MTL::Size(128, 1, 1));
    enc->endEncoding();
}

// rope_qk: in-place rotation on the Q OR K buffer. Existing kernel takes
// (out_q, out_k, pos, cos, sin, T, n_heads, head_dim). For Qwen3 we already
// have separate Q and K buffers post-split, so we call it twice — once with
// the Q buffer aliased to both q and k arg slots (degenerate "rotate Q only"
// pattern), then with K. This matches `kernels/rotary/rotary.metal::rope_qk`.
//
// NOTE: cos/sin tables come from caller-side YaRN-corrected bake (no kernel
// change needed; Qwen3 uses split-half rotation == what rope_qk already does).
inline void encode_rope_qk_inplace(
    MTL::CommandBuffer* cmd, MTL::ComputePipelineState* pso,
    MTL::Buffer* x, MTL::Buffer* pos,
    MTL::Buffer* cos_tbl, MTL::Buffer* sin_tbl,
    uint32_t seq, uint32_t n_heads, uint32_t head_dim)
{
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    // Standard rope_qk signature: (q, k, cos, sin, seq, head_dim, n_heads).
    // Pass x as both q and k slots — the kernel processes each independently.
    enc->setBuffer(x,       0, 0);
    enc->setBuffer(x,       0, 1);
    enc->setBuffer(cos_tbl, 0, 2);
    enc->setBuffer(sin_tbl, 0, 3);
    enc->setBytes(&seq,      4, 4);
    enc->setBytes(&head_dim, 4, 5);
    enc->setBytes(&n_heads,  4, 6);
    enc->dispatchThreadgroups(MTL::Size(n_heads, 1, 1), MTL::Size(1024, 1, 1));
    enc->endEncoding();
}

// ─── Single layer dispatch ───────────────────────────────────────────
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

    // Per-layer byte offsets.
    const size_t off_norm     = (size_t)L * p.d_model * 2;
    const size_t off_w_qkv    = (size_t)L * p.d_model * qkv_N * 2;
    const size_t off_w_q_norm = (size_t)L * hd * 2;
    const size_t off_w_k_norm = (size_t)L * hd * 2;
    const size_t off_w_o      = (size_t)L * p.n_heads * hd * p.d_model * 2;
    const size_t off_w_gate   = (size_t)L * p.d_model * p.n_int * 2;
    const size_t off_w_down   = (size_t)L * p.n_int * p.d_model * 2;

    // 1. Pre-attn RMSNorm.
    encode_rmsnorm(cmd, P.rmsnorm, B.x, B.w_pre_attn_norm, off_norm,
                   B.x_norm, T, p.d_model, p.eps);

    // 2. QKV-pack GEMM: x_norm → qkv_packed (T × (qN + 2*kvN)).
    encode_gemm(cmd, P.gemm, B.x_norm, 0, B.w_qkv, off_w_qkv,
                B.qkv_packed, T, qkv_N, p.d_model);

    // 3a. Split qkv_packed → q (T, qN) + kv_pack (T, 2*kvN).
    encode_split(cmd, P.split_packed, B.qkv_packed, B.q, B.kv_pack,
                 T, qN, 2 * kvN);
    // 3b. Split kv_pack → k_tmp (T, kvN) + v_tmp (T, kvN).
    encode_split(cmd, P.split_packed, B.kv_pack, B.k_tmp, B.v_tmp,
                 T, kvN, kvN);

    // 4. Per-head Q-norm and K-norm. Each row of length hd is one head's
    //    vector. Q has T*n_heads such rows; K has T*n_kv_heads.
    encode_rmsnorm(cmd, P.rmsnorm, B.q, B.w_q_norm, off_w_q_norm,
                   B.q, T * p.n_heads, hd, p.eps);
    encode_rmsnorm(cmd, P.rmsnorm, B.k_tmp, B.w_k_norm, off_w_k_norm,
                   B.k_tmp, T * p.n_kv_heads, hd, p.eps);

    // 5. RoPE on Q and K (in-place). cos/sin tables are caller-baked with
    //    Qwen's θ=1M and YaRN correction (no kernel change needed).
    // TODO: launcher needs to allocate + fill cos_tbl, sin_tbl buffers and
    //       pass them through ModelBuffers. For now skip the encode — the
    //       structural step is here.
    // encode_rope_qk_inplace(cmd, P.rope_qk, B.q,     B.rope_pos, cos_tbl, sin_tbl, p.seq, p.n_heads,    hd);
    // encode_rope_qk_inplace(cmd, P.rope_qk, B.k_tmp, B.rope_pos, cos_tbl, sin_tbl, p.seq, p.n_kv_heads, hd);

    // 6. KV cache write: k_tmp, v_tmp → k_cache, v_cache at write_pos.
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.kv_cache_write);
        enc->setBuffer(B.k_tmp,   0, 0);
        enc->setBuffer(B.v_tmp,   0, 1);
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
        enc->endEncoding();
    }

    // 7. Flash attention (mha_causal — GQA-aware, d=128).
    //    Reads Q (post-RoPE) and the full K/V caches up to write_pos+seq.
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.attn);
        enc->setBuffer(B.q,        0, 0);
        enc->setBuffer(B.k_cache,  0, 1);
        enc->setBuffer(B.v_cache,  0, 2);
        enc->setBuffer(B.attn_out, 0, 3);
        const uint32_t kv_len = p.kv_len;
        const uint32_t cache_stride = p.cache_size;
        enc->setBytes(&p.seq,         4, 4);
        enc->setBytes(&p.n_heads,     4, 5);
        enc->setBytes(&p.n_kv_heads,  4, 6);
        enc->setBytes(&kv_len,        4, 7);
        enc->setBytes(&cache_stride,  4, 8);
        // mha_causal dispatch: GQA-aware. Grid (n_kv_heads, ceil(seq/2), batch),
        // TG Hg*2*32 where Hg = n_heads/n_kv_heads.
        const uint32_t Hg_attn = p.n_heads / p.n_kv_heads;
        enc->dispatchThreadgroups(
            MTL::Size(p.n_kv_heads, (p.seq + 1) / 2, p.batch),
            MTL::Size(Hg_attn * 2 * 32, 1, 1));
        enc->endEncoding();
    }

    // 8. O-projection GEMM: attn_out → o_proj.
    encode_gemm(cmd, P.gemm, B.attn_out, 0, B.w_o, off_w_o, B.o_proj,
                T, p.d_model, p.n_heads * hd);

    // 9. Fused residual + pre-MLP RMSNorm.
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

    // 10. Dense SwiGLU MLP (fused gate + up + SiLU·mul + down).
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.gated_mlp);
        enc->setBuffer(B.m_in,    0,          0);
        enc->setBuffer(B.w_gate,  off_w_gate, 1);
        enc->setBuffer(B.w_up,    off_w_gate, 2);   // same per-layer slab size as w_gate
        enc->setBuffer(B.w_down,  off_w_down, 3);
        enc->setBuffer(B.mlp_out, 0,          4);
        uint32_t M_v = T, N_v = p.d_model, K_v = p.d_model;
        enc->setBytes(&M_v,     4, 5);
        enc->setBytes(&N_v,     4, 6);
        enc->setBytes(&K_v,     4, 7);
        enc->setBytes(&p.n_int, 4, 8);
        enc->dispatchThreadgroups(MTL::Size((N_v + 63) / 64, (M_v + 63) / 64, 1),
                                  MTL::Size(128, 1, 1));
        enc->endEncoding();
    }

    // 11. Final residual: y_out = y_attn + mlp_out.
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.add);
        enc->setBuffer(B.y_attn,  0, 0);
        enc->setBuffer(B.mlp_out, 0, 1);
        enc->setBuffer(B.y_out,   0, 2);
        uint32_t n = T * p.d_model;
        enc->setBytes(&n, 4, 3);
        uint32_t total = (n / 4u) + (n & 3u);
        enc->dispatchThreadgroups(MTL::Size((total + 127) / 128, 1, 1),
                                  MTL::Size(128, 1, 1));
        enc->endEncoding();
    }
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
    uint32_t current_pos  = 0;

    // RoPE
    float    rope_freq_base   = 1000000.f;
    int32_t  rope_n_ctx_orig  = 32768;
    float    rope_freq_scale  = 1.f;
    float    rope_ext_factor  = 0.f;
    float    rope_attn_factor = 1.f;
    float    rope_beta_fast   = 32.f;
    float    rope_beta_slow   = 1.f;
};

struct ModelPSOs {
    LayerPSOs layer;
    MTL::ComputePipelineState* embedding_lookup;
    MTL::ComputePipelineState* argmax;
};

struct LayerCache {
    MTL::Buffer* k;
    MTL::Buffer* v;
};

struct ModelWeights {
    MTL::Buffer* w_embed;
    MTL::Buffer* w_pre_attn_norm;
    MTL::Buffer* w_qkv;
    MTL::Buffer* w_q_norm;        // (n_layers, head_dim)
    MTL::Buffer* w_k_norm;        // (n_layers, head_dim)
    MTL::Buffer* w_o;
    MTL::Buffer* w_pre_mlp_norm;
    MTL::Buffer* w_final_norm;
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

    for (uint32_t L = 0; L < M.n_layers; ++L) {
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
        lb.w_qkv           = W.w_qkv;
        lb.w_q_norm        = W.w_q_norm;
        lb.w_k_norm        = W.w_k_norm;
        lb.w_o             = W.w_o;
        lb.w_pre_mlp_norm  = W.w_pre_mlp_norm;
        lb.w_gate          = W.w_gate;
        lb.w_up            = W.w_up;
        lb.w_down          = W.w_down;
        lb.rope_pos        = B.rope_pos;
        lb.k_cache         = W.layer_caches[L].k;
        lb.v_cache         = W.layer_caches[L].v;
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

        dispatch_layer(cmd, P.layer, lb, lp);

        MTL::Buffer* tmp = cur; cur = nxt; nxt = tmp;
    }

    // C. Final RMSNorm
    encode_rmsnorm(cmd, P.layer.rmsnorm, cur, W.w_final_norm, 0,
                   nxt, T, M.d_model, M.eps);

    // D. LM-head GEMM (tied with embedding via transB=1)
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.layer.gemm);
        const uint32_t M_v = T, K_v = M.d_model, N_v = M.vocab_size;
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

} // namespace qwen
} // namespace meow

#endif
