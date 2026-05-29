// DeepSeek V4 Flash orchestrator. MLA attention + MoE FFN.

#ifndef SUPERKITTENS_DEEPSEEK_MODEL_H
#define SUPERKITTENS_DEEPSEEK_MODEL_H

#include <Metal/Metal.hpp>
#include <cstdint>
#include <cmath>

#include "../../kernels/moe/moe_ffn.h"

namespace meow { namespace deepseek {
inline float metal_sqrt_safe(float x) { return std::sqrt(x); }
}}

namespace meow {
namespace deepseek {

struct LayerParams {
    uint32_t batch          = 1;
    uint32_t seq            = 1;
    uint32_t d_model        = 7168;
    uint32_t n_heads        = 128;
    uint32_t qk_nope_dim    = 128;
    uint32_t qk_rope_dim    = 64;
    uint32_t v_head_dim     = 128;
    uint32_t q_lora_rank    = 1536;
    uint32_t kv_lora_rank   = 512;
    uint32_t n_int          = 2048;
    uint32_t shared_n_int   = 2048;
    uint32_t n_expert       = 256;
    uint32_t top_k          = 8;
    float    eps            = 1e-6f;
    MoeQuant moe_quant      = MoeQuant::FP16;

    uint32_t layer_idx      = 0;
    uint32_t kv_buf_start   = 0;
    uint32_t kv_len         = 1;
    uint32_t cache_size     = 8192;
    uint32_t write_pos      = 0;

    int32_t  rope_n_ctx_orig = 4096;
    float    rope_freq_base  = 10000.f;
    float    rope_freq_scale = 1.f;
    float    rope_ext_factor = 0.f;
    float    rope_attn_factor = 1.f;
    float    rope_beta_fast  = 32.f;
    float    rope_beta_slow  = 1.f;

    // V3/V4 additions (patch A). modeling_deepseek_v3.py refs in comments.
    bool     has_q_lora        = true;     // V2-Lite: false (config.q_lora_rank is None)
    bool     is_moe_layer      = true;     // false for L < first_k_dense_replace
    bool     rope_interleave   = true;     // V3: true (apply_rotary_pos_emb_interleave, modeling:444)
    float    yarn_mscale       = 1.0f;     // pre-squared; attn scale *= mscale^2 (modeling:412-414)
    uint32_t n_group           = 8;        // 0 disables grouping (V2-Lite)
    uint32_t topk_group        = 4;
    float    routed_scaling    = 2.5f;     // V2-Lite: 1.0
    bool     norm_topk_prob    = true;     // V2-Lite: false
    bool     router_has_bias   = true;     // V2-Lite: false (no e_score_correction_bias)
    uint32_t first_k_dense_replace = 3;    // V3: 3 dense layers, V2-Lite: 1
};

// V3 YaRN mscale: mscale = 0.1 * mscale_all_dim * log(factor) + 1
// (modeling_deepseek_v3.py:364-368). Returns 1.0 for factor <= 1.
inline float ds_compute_yarn_mscale(float factor, float mscale_all_dim) {
    if (factor <= 1.0f) return 1.0f;
    return 0.1f * mscale_all_dim * std::log(factor) + 1.0f;
}

struct LayerPSOs {
    MTL::ComputePipelineState* rmsnorm;
    MTL::ComputePipelineState* rmsnorm_t1 = nullptr;  // optional T=1 fast path
    MTL::ComputePipelineState* gemm;
    MTL::ComputePipelineState* rope_tail;
    MTL::ComputePipelineState* rope_interleave;   // V3 GPT-J-style pair RoPE
    MTL::ComputePipelineState* router_v3;          // V3 sigmoid+bias+group+topk router
    MTL::ComputePipelineState* router_v2;          // V2-Lite softmax+topk router (shared moe_router)
    MTL::ComputePipelineState* flash_attn_vec;
    MTL::ComputePipelineState* mla_decode_v2;      // V2-Lite per-head MLA decode (dk=192, dv=128)
    MTL::ComputePipelineState* mla_kv_write;       // assemble dk=192 K + dv=128 V into per-head cache
    MTL::ComputePipelineState* moe_mv_gate;        // mul_mv_id_q4_K (routed gate/up/down)
    MTL::ComputePipelineState* moe_swiglu_f32;     // deepseek_moe_swiglu_f32
    MTL::ComputePipelineState* moe_scatter_add;    // deepseek_moe_scatter_add_f32
    MTL::ComputePipelineState* cast_h2f;
    MTL::ComputePipelineState* cast_f2h;
    MTL::ComputePipelineState* causal_mask_fill;
    MTL::ComputePipelineState* kv_up_pair;
    MTL::ComputePipelineState* split_packed;
    MTL::ComputePipelineState* kv_cache_write;
    MTL::ComputePipelineState* add;
    MTL::ComputePipelineState* add_rmsnorm;
    MTL::ComputePipelineState* gated_mlp;

    MoeFfnPSOs moe;
};

struct LayerBuffers {
    MTL::Buffer* x;

    MTL::Buffer* w_pre_attn_norm;
    MTL::Buffer* w_q_a;
    MTL::Buffer* w_q_a_norm;
    MTL::Buffer* w_q_b;
    MTL::Buffer* w_kv_a;
    MTL::Buffer* w_kv_a_norm;
    MTL::Buffer* w_kv_b;
    MTL::Buffer* w_o;
    MTL::Buffer* w_pre_mlp_norm;

    MTL::Buffer* w_shared_gate;
    MTL::Buffer* w_shared_up;
    MTL::Buffer* w_shared_down;

    MTL::Buffer* w_router;
    MTL::Buffer* router_bias;   // V3 e_score_correction_bias (fp32, per layer, len n_expert). May be null on V2.
    MTL::Buffer* w_gate;
    MTL::Buffer* w_up;
    MTL::Buffer* w_down;

    MTL::Buffer* rope_pos;

    MTL::Buffer* c_kv_cache;
    MTL::Buffer* k_pe_cache;

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
    MTL::Buffer* moe_x_f32;       // routing input cast to fp32 (mul_mv_id reads fp32)
    MTL::Buffer* moe_gate_f32;    // [top_k, n_int] fp32
    MTL::Buffer* moe_up_f32;      // [top_k, n_int] fp32
    MTL::Buffer* moe_mid_f32;     // [top_k, n_int] fp32
    MTL::Buffer* moe_down_f32;    // [top_k, d_model] fp32
    MTL::Buffer* y_out;
};

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
    MTL::Buffer* out, uint32_t rows, uint32_t n, float eps,
    MTL::ComputePipelineState* pso_t1 = nullptr)
{
    const bool use_t1 = (pso_t1 != nullptr) && (rows == 1u);
    auto* enc = cmd->computeCommandEncoder();
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
    enc->endEncoding();
}

inline void dispatch_attn(
    MTL::CommandBuffer*  cmd,
    const LayerPSOs&     P,
    const LayerBuffers&  B,
    const LayerParams&   p)
{
    const uint32_t T   = p.batch * p.seq;
    const uint32_t L   = p.layer_idx;
    const uint32_t dk  = p.qk_nope_dim + p.qk_rope_dim;

    const size_t off_norm        = (size_t)L * p.d_model * 2;
    const size_t off_w_q_a       = (size_t)L * p.d_model * p.q_lora_rank * 2;
    const size_t off_w_q_a_norm  = (size_t)L * p.q_lora_rank * 2;
    const size_t off_w_q_b       = (size_t)L * p.q_lora_rank * p.n_heads * dk * 2;
    const size_t off_w_kv_a      = (size_t)L * p.d_model * (p.kv_lora_rank + p.qk_rope_dim) * 2;
    const size_t off_w_kv_a_norm = (size_t)L * p.kv_lora_rank * 2;
    const size_t off_w_o         = (size_t)L * p.n_heads * p.v_head_dim * p.d_model * 2;

    encode_rmsnorm(cmd, P.rmsnorm, B.x, B.w_pre_attn_norm, off_norm,
                   B.x_norm, T, p.d_model, p.eps, P.rmsnorm_t1);

    if (p.has_q_lora) {
        encode_gemm(cmd, P.gemm, B.x_norm, 0, B.w_q_a, off_w_q_a, B.q_a,
                    T, p.q_lora_rank, p.d_model);
        encode_rmsnorm(cmd, P.rmsnorm, B.q_a, B.w_q_a_norm, off_w_q_a_norm,
                       B.q_a, T, p.q_lora_rank, p.eps, P.rmsnorm_t1);
        encode_gemm(cmd, P.gemm, B.q_a, 0, B.w_q_b, off_w_q_b, B.q_packed,
                    T, p.n_heads * dk, p.q_lora_rank);
    } else {
        // Patch H: V2-Lite has no q_lora; q_proj is a single dense (d_model -> n_heads*dk).
        // Loader writes the q_proj weight directly into w_q_b (configuration_deepseek_v2:q_lora_rank=None).
        const size_t off_w_q_proj = (size_t)L * p.d_model * p.n_heads * dk * 2;
        encode_gemm(cmd, P.gemm, B.x_norm, 0, B.w_q_b, off_w_q_proj, B.q_packed,
                    T, p.n_heads * dk, p.d_model);
    }

    encode_gemm(cmd, P.gemm, B.x_norm, 0, B.w_kv_a, off_w_kv_a, B.kv_a_packed,
                T, p.kv_lora_rank + p.qk_rope_dim, p.d_model);

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

    encode_rmsnorm(cmd, P.rmsnorm, B.c_kv, B.w_kv_a_norm, off_w_kv_a_norm,
                   B.c_kv, T, p.kv_lora_rank, p.eps, P.rmsnorm_t1);

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

        // Patch C: V3 uses interleaved (pair) RoPE on the rotated half of Q.
        // modeling_deepseek_v3.py:apply_rotary_pos_emb_interleave (~:444).
        // V2 path: rope_tail (NeoX/half-split mode=2).
        struct alignas(8) ArgsRopeInterleave {
            int32_t  ne00, ne01, ne02, ne03;
            uint64_t nb01, nb02, nb03;
            int32_t  n_dims;
            int32_t  n_ctx_orig;
            float    freq_base, freq_scale, ext_factor, attn_factor;
            float    beta_fast, beta_slow, mscale;
        };
        auto* enc = cmd->computeCommandEncoder();
        if (p.rope_interleave) {
            ArgsRopeInterleave iq{};
            iq.ne00 = dk; iq.ne01 = p.seq; iq.ne02 = p.n_heads; iq.ne03 = p.batch;
            iq.nb01 = sizeof(float) * dk;
            iq.nb02 = iq.nb01 * p.seq;
            iq.nb03 = iq.nb02 * p.n_heads;
            iq.n_dims     = (int32_t)p.qk_rope_dim;
            iq.n_ctx_orig = p.rope_n_ctx_orig;
            iq.freq_base  = p.rope_freq_base;
            iq.freq_scale = p.rope_freq_scale;
            iq.ext_factor = p.rope_ext_factor;
            iq.attn_factor = p.rope_attn_factor;
            iq.beta_fast  = p.rope_beta_fast;
            iq.beta_slow  = p.rope_beta_slow;
            iq.mscale     = 1.0f;
            enc->setComputePipelineState(P.rope_interleave);
            enc->setBytes(&iq, sizeof(iq), 0);
            enc->setBuffer(B.q_packed_f32, 0, 1);
            enc->setBuffer(B.rope_pos,     0, 2);
            enc->setBuffer(B.q_packed_f32, 0, 4);
        } else {
            enc->setComputePipelineState(P.rope_tail);
            enc->setBytes(&aq, sizeof(aq), 0);
            enc->setBuffer(B.q_packed_f32, 0, 1);
            enc->setBuffer(B.rope_pos,     0, 2);
            enc->setBuffer(B.q_packed_f32, 0, 3);
            enc->setBuffer(B.q_packed_f32, 0, 4);
        }
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
        if (p.rope_interleave) {
            ArgsRopeInterleave ik{};
            ik.ne00 = p.qk_rope_dim;
            ik.ne01 = p.seq * p.batch; ik.ne02 = 1; ik.ne03 = 1;
            ik.nb01 = sizeof(float) * p.qk_rope_dim;
            ik.nb02 = ik.nb01 * ik.ne01;
            ik.nb03 = ik.nb02;
            ik.n_dims     = (int32_t)p.qk_rope_dim;
            ik.n_ctx_orig = p.rope_n_ctx_orig;
            ik.freq_base  = p.rope_freq_base;
            ik.freq_scale = p.rope_freq_scale;
            ik.ext_factor = p.rope_ext_factor;
            ik.attn_factor = p.rope_attn_factor;
            ik.beta_fast  = p.rope_beta_fast;
            ik.beta_slow  = p.rope_beta_slow;
            ik.mscale     = 1.0f;
            enc2->setComputePipelineState(P.rope_interleave);
            enc2->setBytes(&ik, sizeof(ik), 0);
            enc2->setBuffer(B.k_pe_f32, 0, 1);
            enc2->setBuffer(B.rope_pos, 0, 2);
            enc2->setBuffer(B.k_pe_f32, 0, 4);
        } else {
            enc2->setComputePipelineState(P.rope_tail);
            enc2->setBytes(&ak, sizeof(ak), 0);
            enc2->setBuffer(B.k_pe_f32, 0, 1);
            enc2->setBuffer(B.rope_pos, 0, 2);
            enc2->setBuffer(B.k_pe_f32, 0, 3);
            enc2->setBuffer(B.k_pe_f32, 0, 4);
        }
        enc2->dispatchThreadgroups(MTL::Size(p.seq * p.batch, 1, 1),
                                   MTL::Size(256, 1, 1));
        enc2->endEncoding();

        encode_cast(cmd, P.cast_f2h, B.k_pe_f32, B.k_pe,
                    T * p.qk_rope_dim);
    }

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
        enc->setBuffer(B.c_kv,        0,           0);  // normalized compressed-KV
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

    {
        // Assemble per-head K (dk=192: nope ++ shared rope) and V (dv=128) for
        // this decode step into the per-head cache at write_pos. k_no_pe/v are
        // the kv_up_pair outputs; k_pe is the shared rotated key (fp16).
        {
            auto* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(P.mla_kv_write);
            enc->setBuffer(B.k_no_pe,    0, 0);
            enc->setBuffer(B.k_pe,       0, 1);
            enc->setBuffer(B.v,          0, 2);
            enc->setBuffer(B.c_kv_cache, 0, 3);   // K cache [H, cache, 192]
            enc->setBuffer(B.k_pe_cache, 0, 4);   // V cache [H, cache, 128]
            enc->setBytes(&T,             4, 5);
            enc->setBytes(&p.n_heads,     4, 6);
            enc->setBytes(&p.qk_nope_dim, 4, 7);
            enc->setBytes(&p.qk_rope_dim, 4, 8);
            enc->setBytes(&p.v_head_dim,  4, 9);
            enc->setBytes(&p.cache_size,  4, 10);
            enc->setBytes(&p.write_pos,   4, 11);
            enc->dispatchThreads(MTL::Size(dk, p.n_heads, T),
                                 MTL::Size(dk < 256u ? dk : 256u, 1, 1));
            enc->endEncoding();
        }

        // Causal mask over (q_seq, kv_len).
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

        // kernel_mla_decode_v2 arg struct (== flash_attn_ext_vec layout). Q is
        // fp32 [seq, head, dk]; K/V are the per-head caches [head, kv, dk|dv].
        #pragma pack(push, 8)
        struct ArgsMLA {
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
        static_assert(sizeof(ArgsMLA) == 192, "MLA args mismatch");

        ArgsMLA a{};
        a.ne01 = (int32_t)p.seq;     a.ne02 = (int32_t)p.n_heads; a.ne03 = (int32_t)p.batch;
        a.nb01 = (uint64_t)dk * sizeof(float);   // Q fp32 row stride
        a.nb02 = a.nb01 * p.seq;
        a.nb03 = a.nb02 * p.n_heads;
        a.ne11 = (int32_t)p.kv_len;  a.ne_12_2 = (int32_t)p.n_heads; a.ne_12_3 = (int32_t)p.batch;
        // K cache: [head, cache_max, dk]. Row stride dk*2; head stride cache_max*dk*2.
        a.nb11 = (uint64_t)dk * sizeof(uint16_t);
        a.nb12 = (uint64_t)p.cache_size * dk * sizeof(uint16_t);
        a.nb13 = a.nb12 * p.n_heads;
        a.ns10 = (int32_t)a.nb11;
        // V cache: [head, cache_max, dv].
        a.nb21 = (uint64_t)p.v_head_dim * sizeof(uint16_t);
        a.nb22 = (uint64_t)p.cache_size * p.v_head_dim * sizeof(uint16_t);
        a.nb23 = a.nb22 * p.n_heads;
        a.ns20 = (int32_t)a.nb21;
        a.ne31 = (int32_t)p.seq; a.ne32 = 1; a.ne33 = 1;
        a.nb31 = (uint64_t)p.kv_len * sizeof(uint16_t);
        a.nb32 = a.nb31 * p.seq;
        a.nb33 = a.nb32;
        a.ne1 = (int32_t)p.n_heads; a.ne2 = (int32_t)p.seq; a.ne3 = (int32_t)p.batch;
        a.scale = (1.f / metal_sqrt_safe((float)dk)) * p.yarn_mscale * p.yarn_mscale;
        a.max_bias = 0.f; a.m0 = 1.f; a.m1 = 1.f;
        a.n_head_log2 = 0; a.logit_softcap = 0.f;

        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.mla_decode_v2);
        enc->setBytes(&a, sizeof(a), 0);
        enc->setBuffer(B.q_packed_f32, 0, 1);   // Q (fp32, RoPE-applied)
        enc->setBuffer(B.c_kv_cache,    0, 2);   // K cache (fp16)
        enc->setBuffer(B.k_pe_cache,    0, 3);   // V cache (fp16)
        enc->setBuffer(B.causal_mask,   0, 4);   // mask (fp16)
        enc->setBuffer(B.q_packed_f32,  0, 5);   // sinks (unused)
        enc->setBuffer(B.q_packed_f32,  0, 6);   // pad   (unused)
        enc->setBuffer(B.attn_out_f32,  0, 7);   // dst (fp32 [batch, seq, head, dv])
        enc->dispatchThreadgroups(
            MTL::Size(p.seq, p.n_heads, p.batch),
            MTL::Size(32, 1, 1));   // one simdgroup per (q,head,batch)
        enc->endEncoding();

        // Cast attn_out fp32 → fp16 for the O-proj GEMM.
        encode_cast(cmd, P.cast_f2h, B.attn_out_f32, B.attn_out,
                    T * p.n_heads * p.v_head_dim);
    }

    encode_gemm(cmd, P.gemm, B.attn_out, 0, B.w_o, off_w_o, B.o_proj,
                T, p.d_model, p.n_heads * p.v_head_dim);
}

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

    dispatch_shared_expert(cmd, P, B, p);

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

    // Patch E: layers 0..first_k_dense_replace-1 use plain dense MLP (no MoE).
    // The shared-expert dispatch above already computed dense MLP into
    // shared_out (loader puts dense gate/up/down into w_shared_* with width
    // config.intermediate_size). Skip MoE; emit y_out = shared_out.
    // configuration_deepseek_v3.py:90 first_k_dense_replace=3.
    if (!p.is_moe_layer) {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.add);
        // Zero-add trick: y_out = shared_out + 0; reuse shared_out buffer slot 0,1.
        // Simpler: a single copy via add(shared_out, shared_out, y_out) / 2 is wrong.
        // Use add of shared_out and shared_out then divide? No — emit
        // y_out = shared_out via memcpy-equivalent: bind shared_out twice to add
        // then halve? No. The add kernel signature is (a, b, out, n) so we use
        // a single add encoding with B.shared_out as one input and a zero buffer
        // would be cleaner. Instead, just chain: enc->setBuffer copies via
        // add(y_attn=0?). Cleanest: use blitCommandEncoder for the copy.
        enc->endEncoding();
        auto* blit = cmd->blitCommandEncoder();
        blit->copyFromBuffer(B.shared_out, 0, B.y_out, 0,
                             (size_t)T * p.d_model * 2);
        blit->endEncoding();
        return;
    }

    // ── V2-Lite MoE: shared moe_router → per-expert mul_mv_id_q4_K (gate/up) →
    //    SwiGLU → mul_mv_id_q4_K (down) → weighted scatter-add into residual. ──
    // moe_router: x fp16 (m_in), W fp16 (w_router, [D,N]) → top_idx, top_score.
    {
        const size_t router_off = (size_t)L * p.d_model * p.n_expert * 2;
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.router_v2);
        enc->setBuffer(B.m_in,          0,          0);
        enc->setBuffer(B.w_router,      router_off, 1);
        enc->setBuffer(B.moe_top_idx,   0,          2);
        enc->setBuffer(B.moe_top_score, 0,          3);
        enc->setBytes(&T,         4, 4);
        enc->setBytes(&p.d_model, 4, 5);
        enc->setBytes(&p.n_expert,4, 6);
        enc->setBytes(&p.top_k,   4, 7);
        enc->dispatchThreadgroups(MTL::Size(T, 1, 1), MTL::Size(256, 1, 1));
        enc->endEncoding();
    }

    // Routing input cast fp16 → fp32 (mul_mv_id reads fp32 activations).
    encode_cast(cmd, P.cast_h2f, B.m_in, B.moe_x_f32, T * p.d_model);

    // Q4_K per-expert matvec: ne00=in_dim, ne01=out_rows, ne02=n_expert.
    // src0 stride: block=144 B / 256 weights. ids = top_idx (T*top_k ints).
    struct ArgsMulMvId {
        int32_t  nei0; int32_t nei1; uint64_t nbi1;
        int32_t  ne00; int32_t ne01; int32_t ne02; char _p0[4];
        uint64_t nb00; uint64_t nb01; uint64_t nb02;
        int32_t  ne10; int32_t ne11; int32_t ne12; int32_t ne13;
        uint64_t nb10; uint64_t nb11; uint64_t nb12;
        int32_t  ne0; int32_t ne1; uint64_t nb1; int32_t nr0; char _p1[4];
    };
    const uint32_t Q4K_BLK = 144;
    const uint32_t NR0_Q4K = 2;
    auto encode_mv_id = [&](MTL::Buffer* w, size_t w_off, MTL::Buffer* src1,
                            MTL::Buffer* dst, uint32_t in_dim, uint32_t out_rows) {
        const uint64_t row_blk = (uint64_t)(in_dim / 256) * Q4K_BLK;   // nb01
        const uint64_t slab    = (uint64_t)out_rows * row_blk;          // nb02
        ArgsMulMvId a{};
        a.nei0 = (int32_t)p.top_k; a.nei1 = (int32_t)T;
        a.nbi1 = (uint64_t)p.top_k * sizeof(int32_t);
        a.ne00 = (int32_t)in_dim; a.ne01 = (int32_t)out_rows; a.ne02 = (int32_t)p.n_expert;
        a.nb00 = Q4K_BLK; a.nb01 = row_blk; a.nb02 = slab;
        a.ne10 = (int32_t)in_dim; a.ne11 = 1; a.ne12 = 1; a.ne13 = 1;
        a.nb10 = sizeof(float); a.nb11 = (uint64_t)in_dim * sizeof(float); a.nb12 = a.nb11;
        a.ne0 = (int32_t)out_rows; a.ne1 = 1; a.nb1 = (uint64_t)out_rows * sizeof(float);
        a.nr0 = (int32_t)NR0_Q4K;
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.moe_mv_gate);
        enc->setBytes(&a, sizeof(a), 0);
        enc->setBuffer(w,    w_off, 1);
        enc->setBuffer(src1, 0,     2);
        enc->setBuffer(dst,  0,     3);
        enc->setBuffer(B.moe_top_idx, 0, 4);
        enc->setThreadgroupMemoryLength(NR0_Q4K * 32 * sizeof(float), 0);
        // q4_K covers NSG*NR0 rows per threadgroup.x (first_row=(x*NSG+sg)*nr0).
        const uint32_t rows_per_tg = 4u * NR0_Q4K;
        enc->dispatchThreadgroups(
            MTL::Size((out_rows + rows_per_tg - 1) / rows_per_tg, 1, T * p.top_k),
            MTL::Size(4 * 32, 1, 1));   // NSG=4
        enc->endEncoding();
    };

    // Per-layer Q4_K byte strides (144 B / 256 weights).
    const size_t gate_layer = (size_t)p.n_expert * p.n_int * (p.d_model / 256) * Q4K_BLK;
    const size_t down_layer = (size_t)p.n_expert * p.d_model * (p.n_int / 256) * Q4K_BLK;
    encode_mv_id(B.w_gate, (size_t)L * gate_layer, B.moe_x_f32, B.moe_gate_f32,
                 p.d_model, p.n_int);
    encode_mv_id(B.w_up,   (size_t)L * gate_layer, B.moe_x_f32, B.moe_up_f32,
                 p.d_model, p.n_int);

    // SwiGLU mid = silu(gate)*up over [top_k, n_int].
    {
        const uint32_t n = T * p.top_k * p.n_int;
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.moe_swiglu_f32);
        enc->setBuffer(B.moe_gate_f32, 0, 0);
        enc->setBuffer(B.moe_up_f32,   0, 1);
        enc->setBuffer(B.moe_mid_f32,  0, 2);
        enc->setBytes(&n, 4, 3);
        enc->dispatchThreadgroups(MTL::Size((n + 255) / 256, 1, 1),
                                  MTL::Size(256, 1, 1));
        enc->endEncoding();
    }

    encode_mv_id(B.w_down, (size_t)L * down_layer, B.moe_mid_f32, B.moe_down_f32,
                 p.n_int, p.d_model);

    // Weighted scatter-add: y_out = residual + scale * Σ score[s] * down[s].
    {
        const float scale = p.routed_scaling;
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.moe_scatter_add);
        enc->setBuffer(B.moe_down_f32,  0, 0);
        enc->setBuffer(B.moe_top_score, 0, 1);
        enc->setBuffer(B.shared_out,    0, 2);   // residual = y_attn + shared
        enc->setBuffer(B.y_out,         0, 3);
        enc->setBytes(&p.d_model, 4, 4);
        enc->setBytes(&p.top_k,   4, 5);
        enc->setBytes(&scale,     4, 6);
        enc->dispatchThreadgroups(MTL::Size((p.d_model + 255) / 256, 1, 1),
                                  MTL::Size(256, 1, 1));
        enc->endEncoding();
    }
}


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

    int32_t  rope_n_ctx_orig = 4096;
    float    rope_freq_base  = 10000.f;
    float    rope_freq_scale = 1.f;
    float    rope_ext_factor = 0.f;
    float    rope_attn_factor = 1.f;
    float    rope_beta_fast  = 32.f;
    float    rope_beta_slow  = 1.f;

    // V3 additions.
    bool     has_q_lora              = true;
    bool     router_has_bias         = true;
    bool     rope_interleave         = true;
    bool     norm_topk_prob          = true;
    uint32_t n_group                 = 8;
    uint32_t topk_group              = 4;
    float    routed_scaling          = 2.5f;
    float    mscale_all_dim          = 1.0f;
    float    rope_scaling_factor     = 1.0f;
    uint32_t first_k_dense_replace   = 3;
};

struct ModelPSOs {
    LayerPSOs layer;
    MTL::ComputePipelineState* embedding_lookup;
    MTL::ComputePipelineState* argmax;
    // Optional 2-pass fp16 argmax PSOs (≈3.8× at V=102400).
    MTL::ComputePipelineState* argmax_partial = nullptr;
    MTL::ComputePipelineState* argmax_reduce  = nullptr;
};

struct LayerCache {
    MTL::Buffer* c_kv;        // (cache_max, kv_lora_rank)
    MTL::Buffer* k_pe;        // (cache_max, qk_rope_dim)
};

struct ModelWeights {
    MTL::Buffer* w_embed;
    MTL::Buffer* w_lm_head;            // patch F: untied LM head; loader may alias to w_embed.
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
    MTL::Buffer* router_bias;          // patch G: per-layer e_score_correction_bias concatenated (n_layers * n_expert fp32). May be null on V2.
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
    MTL::Buffer* moe_x_f32;
    MTL::Buffer* moe_gate_f32;
    MTL::Buffer* moe_up_f32;
    MTL::Buffer* moe_mid_f32;
    MTL::Buffer* moe_down_f32;

    // 2-pass argmax scratch (ceil(vocab_size/16384) partials each).
    MTL::Buffer* argmax_val_buf = nullptr;
    MTL::Buffer* argmax_idx_buf = nullptr;
};

inline void dispatch_model(
    MTL::CommandBuffer* cmd,
    const ModelPSOs&    P,
    const ModelWeights& W,
    ModelBuffers&       B,
    const ModelParams&  M)
{
    const uint32_t T = M.batch * M.seq;

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
        // V3 plumbing.
        lp.has_q_lora      = M.has_q_lora;
        lp.is_moe_layer    = (L >= M.first_k_dense_replace);
        lp.rope_interleave = M.rope_interleave;
        lp.yarn_mscale     = ds_compute_yarn_mscale(M.rope_scaling_factor, M.mscale_all_dim);
        lp.n_group         = M.n_group;
        lp.topk_group      = M.topk_group;
        lp.routed_scaling  = M.routed_scaling;
        lp.norm_topk_prob  = M.norm_topk_prob;
        lp.router_has_bias = M.router_has_bias;
        lp.first_k_dense_replace = M.first_k_dense_replace;

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
        lb.router_bias     = W.router_bias;
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
        lb.moe_x_f32     = B.moe_x_f32;
        lb.moe_gate_f32  = B.moe_gate_f32;
        lb.moe_up_f32    = B.moe_up_f32;
        lb.moe_mid_f32   = B.moe_mid_f32;
        lb.moe_down_f32  = B.moe_down_f32;
        lb.y_out         = nxt;

        dispatch_layer(cmd, P.layer, lb, lp);

        MTL::Buffer* tmp = cur; cur = nxt; nxt = tmp;
    }

    encode_rmsnorm(cmd, P.layer.rmsnorm, cur, W.w_final_norm, 0,
                   nxt, T, M.d_model, M.eps, P.layer.rmsnorm_t1);

    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.layer.gemm);
        const uint32_t M_v = T;
        const uint32_t K_v = M.d_model;
        const uint32_t N_v = M.vocab_size;
        uint32_t ldA = K_v, ldB = K_v, ldC = N_v;
        int transA = 0, transB = 1, has_bias = 0;
        enc->setBuffer(nxt,        0, 0);
        // Patch F: untied LM head; loader aliases w_lm_head -> w_embed for V2-tied case.
        enc->setBuffer(W.w_lm_head ? W.w_lm_head : W.w_embed, 0, 1);
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

    {
        const bool can_2pass = (T == 1u)
                            && P.argmax_partial && P.argmax_reduce
                            && B.argmax_val_buf && B.argmax_idx_buf;
        if (can_2pass) {
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
            enc->setBuffer(B.logits,    0, 0);
            enc->setBuffer(B.output_id, 0, 1);
            enc->setBytes(&M.vocab_size, 4, 2);
            enc->dispatchThreadgroups(MTL::Size(T, 1, 1), MTL::Size(1024, 1, 1));
            enc->endEncoding();
        }
    }
}

} // namespace deepseek
} // namespace meow

#endif
