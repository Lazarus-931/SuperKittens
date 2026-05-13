//
//  gemma4_model.h — Gemma 4 dispatch orchestrator.
//
//  dispatch_layer  : one transformer layer (LOCAL d=256 SWA or GLOBAL d=512).
//  dispatch_model  : embedding → N×layer → norm → LM head → argmax.
//
//  Buffers caller-allocated; PSOs caller-resolved. Header-only.
//

#ifndef SUPERKITTENS_GEMMA4_MODEL_H
#define SUPERKITTENS_GEMMA4_MODEL_H

#include <Metal/Metal.hpp>
#include <cstdint>

namespace meow {
namespace gemma4 {

// ──────────────────────────────────────────────────────────────────────
//  Layer-level
// ──────────────────────────────────────────────────────────────────────

struct LayerParams {
    // Shape
    uint32_t batch          = 1;
    uint32_t seq            = 1;       // Q rows for this dispatch (decode=1)
    uint32_t n_heads        = 16;
    uint32_t n_kv_heads     = 4;       // for THIS layer (local or global value)
    uint32_t n_kv_heads_max = 16;      // largest across layer-types (for slab strides)
    uint32_t head_dim       = 256;     // 256 LOCAL, 512 GLOBAL
    uint32_t head_dim_max   = 512;     // for slab strides on shared buffers
    uint32_t window         = 4096;
    bool     is_global      = false;
    uint32_t prope_p_pairs  = 64;
    uint32_t d_model        = 4608;
    uint32_t n_int          = 12288;
    uint32_t ple_dim        = 256;
    uint32_t rot_dims       = 0;       // 0 = rotate full head_dim (default)
    float    eps            = 1e-5f;

    // Per-call (varies per layer + decode position)
    uint32_t layer_idx      = 0;
    uint32_t kv_buf_start   = 0;       // logical-K[0] offset in this layer's cache
    uint32_t kv_len         = 1;       // total K positions to attend over
    uint32_t cache_size     = 4096;    // physical cache buffer dim for this layer-type
    uint32_t write_pos      = 0;       // where to write new K/V into cache (logical pos)

    // Per-layer MLP offsets (in elements, not bytes).
    size_t   off_w_gate_e   = 0;       // (dm*sum_{l<L} n_int_l)
    size_t   off_w_down_e   = 0;       // (sum_{l<L} n_int_l * dm)

    // KV-sharing
    bool     is_kv_shared   = false;   // if true, skip kv_cache_write and reuse source's cache
};

struct LayerPSOs {
    MTL::ComputePipelineState* rmsnorm;
    MTL::ComputePipelineState* gemm;
    MTL::ComputePipelineState* qkv_norm;
    MTL::ComputePipelineState* rope;            // standard (local)
    MTL::ComputePipelineState* prope;           // p-RoPE (global, legacy/unused for Gemma4 full-attn)
    MTL::ComputePipelineState* rope_partial;    // partial RoPE for full_attention (rot_dims arg)
    MTL::ComputePipelineState* attn_local;
    MTL::ComputePipelineState* attn_global;
    MTL::ComputePipelineState* gated_mlp_gelu;
    MTL::ComputePipelineState* add;
    MTL::ComputePipelineState* add_rmsnorm;
    MTL::ComputePipelineState* kv_cache_write;
    MTL::ComputePipelineState* ple_gate_act;
    MTL::ComputePipelineState* ple_inject;
};

struct LayerBuffers {
    MTL::Buffer* x;                  // input

    // Concatenated per-layer weights (all layers in one buffer; offsets via layer_idx).
    MTL::Buffer* w_pre_attn_norm;    // (n_layers, d_model)
    MTL::Buffer* w_post_attn_norm;   // (n_layers, d_model)
    MTL::Buffer* w_pre_feedforward_layernorm;  // (n_layers, d_model)
    MTL::Buffer* w_post_feedforward_layernorm; // (n_layers, d_model)
    MTL::Buffer* w_qkv;              // (n_layers, d_model, qkv_out_max)
    MTL::Buffer* gamma_q;            // (n_layers, head_dim_max)
    MTL::Buffer* gamma_k;            // (n_layers, head_dim_max)
    MTL::Buffer* w_out;              // (n_layers, n_heads*head_dim_max, d_model)
    MTL::Buffer* w_gate;             // (n_layers, d_model, n_int)
    MTL::Buffer* w_up;               // (n_layers, d_model, n_int)
    MTL::Buffer* w_down;             // (n_layers, n_int, d_model)

    // PLE pipeline (E-models)
    MTL::Buffer* w_per_layer_input_gate;        // (n_layers, PLE_dim, d_model)
    MTL::Buffer* w_per_layer_projection;        // (n_layers, d_model, PLE_dim)
    MTL::Buffer* w_post_per_layer_input_norm;   // (n_layers, d_model)
    MTL::Buffer* w_layer_scalar;                // (n_layers,) fp32
    MTL::Buffer* per_layer_inputs;              // (T, n_layers, PLE_dim)

    // RoPE tables (shared, no layer offset)
    MTL::Buffer* cos;
    MTL::Buffer* sin;

    // K/V cache for THIS layer (caller selected the right buffer per layer-type).
    MTL::Buffer* k_cache;            // (batch, n_kv_heads, cache_size, head_dim)
    MTL::Buffer* v_cache;

    // Scratch (reused across layers)
    MTL::Buffer* x_norm;
    MTL::Buffer* qkv_packed;
    MTL::Buffer* q_norm;             // Q goes here; K/V go to cache directly
    MTL::Buffer* k_tmp;              // tmp K before cache write (need rotated K)
    MTL::Buffer* v_tmp;              // tmp V before cache write
    MTL::Buffer* attn_out;
    MTL::Buffer* o_proj;
    MTL::Buffer* y_attn;
    MTL::Buffer* m_in;
    MTL::Buffer* m_out;
    MTL::Buffer* y_out;

    // PLE scratch
    MTL::Buffer* ple_gate_out;       // (T, PLE_dim)
    MTL::Buffer* ple_gated;          // (T, PLE_dim)
    MTL::Buffer* ple_proj_back;      // (T, d_model)
};

inline void dispatch_layer(
    MTL::CommandBuffer*  cmd,
    const LayerPSOs&     P,
    const LayerBuffers&  B,
    const LayerParams&   p)
{
    using NS::UInteger;

    const uint32_t L = p.layer_idx;
    const uint32_t qkv_out = (p.n_heads + 2u * p.n_kv_heads_max) * p.head_dim_max;
    const uint32_t qhd_max = p.n_heads * p.head_dim_max;

    // Per-layer byte offsets.
    const size_t off_norm        = (size_t)L * p.d_model * 2;
    const size_t off_qkv         = (size_t)L * p.d_model * qkv_out * 2;
    const size_t off_gamma       = (size_t)L * p.head_dim_max * 2;
    const size_t off_w_out       = (size_t)L * qhd_max * p.d_model * 2;
    const size_t off_w_gate      = p.off_w_gate_e * 2;
    const size_t off_w_down      = p.off_w_down_e * 2;

    // 1. Pre-attn RMSNorm
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.rmsnorm);
        enc->setBuffer(B.x,               0,        0);
        enc->setBuffer(B.w_pre_attn_norm, off_norm, 1);
        enc->setBuffer(B.x_norm,          0,        2);
        uint32_t rows = p.batch * p.seq;
        enc->setBytes(&rows,      4, 3);
        enc->setBytes(&p.d_model, 4, 4);
        enc->setBytes(&p.eps,     4, 5);
        enc->dispatchThreadgroups(MTL::Size(1, (rows + 3) / 4, 1),
                                  MTL::Size(128, 1, 1));
        enc->endEncoding();
    }

    // 2. QKV projection
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.gemm);
        const uint32_t M = p.batch * p.seq;
        const uint32_t K_v = p.d_model;
        const uint32_t N_v = (p.n_heads + 2 * p.n_kv_heads) * p.head_dim;
        uint32_t ldA = K_v, ldB = N_v, ldC = N_v;
        int transA = 0, transB = 0, has_bias = 0;
        enc->setBuffer(B.x_norm,    0,       0);
        enc->setBuffer(B.w_qkv,     off_qkv, 1);
        enc->setBuffer(B.qkv_packed,0,       2);
        enc->setBytes(&M,        4, 3);
        enc->setBytes(&N_v,      4, 4);
        enc->setBytes(&K_v,      4, 5);
        enc->setBytes(&ldA,      4, 6);
        enc->setBytes(&ldB,      4, 7);
        enc->setBytes(&ldC,      4, 8);
        enc->setBytes(&transA,   4, 9);
        enc->setBytes(&transB,   4, 10);
        enc->setBytes(&has_bias, 4, 11);
        enc->setBuffer(B.qkv_packed, 0, 12);
        enc->dispatchThreadgroups(MTL::Size((N_v + 63) / 64, (M + 63) / 64, 1),
                                  MTL::Size(64, 1, 1));
        enc->endEncoding();
    }

    // 3. QKV split + per-head RMSNorm (V no γ)
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.qkv_norm);
        enc->setBuffer(B.qkv_packed, 0,         0);
        enc->setBuffer(B.gamma_q,    off_gamma, 1);
        enc->setBuffer(B.gamma_k,    off_gamma, 2);
        enc->setBuffer(B.q_norm,     0,         3);
        enc->setBuffer(B.k_tmp,      0,         4);
        enc->setBuffer(B.v_tmp,      0,         5);
        uint32_t T = p.batch * p.seq;
        enc->setBytes(&T,            4, 6);
        enc->setBytes(&p.n_heads,    4, 7);
        enc->setBytes(&p.n_kv_heads, 4, 8);
        enc->setBytes(&p.head_dim,   4, 9);
        enc->setBytes(&p.eps,        4, 10);
        const uint32_t slots = p.n_heads + 2u * p.n_kv_heads;
        enc->dispatchThreadgroups(MTL::Size(slots, T, 1),
                                  MTL::Size(p.head_dim, 1, 1));
        enc->endEncoding();
    }

    // 4. RoPE / p-RoPE on Q and the in-flight K (k_tmp), in-place.
    {
        auto* enc = cmd->computeCommandEncoder();
        if (p.is_global) {
            // Gemma4 full_attention uses standard partial RoPE (HF
            // modeling_gemma4.py:1229,1245 with partial_rotary_factor=0.25),
            // NOT p-RoPE. rot_dims = head_dim * 0.25.
            uint32_t rot_dims = p.rot_dims
                                ? p.rot_dims
                                : (uint32_t)(p.head_dim * 0.25f);
            enc->setComputePipelineState(P.rope_partial);
            enc->setBuffer(B.q_norm, 0, 0);
            enc->setBuffer(B.q_norm, 0, 1);
            enc->setBuffer(B.cos,    0, 2);
            enc->setBuffer(B.sin,    0, 3);
            enc->setBytes(&p.seq,      4, 4);
            enc->setBytes(&p.head_dim, 4, 5);
            enc->setBytes(&p.n_heads,  4, 6);
            enc->setBytes(&rot_dims,   4, 7);
            enc->setBytes(&p.write_pos,4, 8);
            enc->dispatchThreadgroups(MTL::Size(p.n_heads, 1, 1),
                                      MTL::Size(1024, 1, 1));
            enc->endEncoding();
            enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(P.rope_partial);
            enc->setBuffer(B.k_tmp,  0, 0);
            enc->setBuffer(B.k_tmp,  0, 1);
            enc->setBuffer(B.cos,    0, 2);
            enc->setBuffer(B.sin,    0, 3);
            enc->setBytes(&p.seq,      4, 4);
            enc->setBytes(&p.head_dim, 4, 5);
            enc->setBytes(&p.n_kv_heads,4, 6);
            enc->setBytes(&rot_dims,   4, 7);
            enc->setBytes(&p.write_pos,4, 8);
            enc->dispatchThreadgroups(MTL::Size(p.n_kv_heads, 1, 1),
                                      MTL::Size(1024, 1, 1));
        } else {
            enc->setComputePipelineState(P.rope);
            enc->setBuffer(B.q_norm, 0, 0);
            enc->setBuffer(B.q_norm, 0, 1);
            enc->setBuffer(B.cos,    0, 2);
            enc->setBuffer(B.sin,    0, 3);
            enc->setBytes(&p.seq,      4, 4);
            enc->setBytes(&p.head_dim, 4, 5);
            enc->setBytes(&p.n_heads,  4, 6);
            enc->setBytes(&p.write_pos,4, 7);
            enc->dispatchThreadgroups(MTL::Size(p.n_heads, 1, 1),
                                      MTL::Size(1024, 1, 1));
            enc->endEncoding();
            enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(P.rope);
            enc->setBuffer(B.k_tmp,  0, 0);
            enc->setBuffer(B.k_tmp,  0, 1);
            enc->setBuffer(B.cos,    0, 2);
            enc->setBuffer(B.sin,    0, 3);
            enc->setBytes(&p.seq,      4, 4);
            enc->setBytes(&p.head_dim, 4, 5);
            enc->setBytes(&p.n_kv_heads,4, 6);
            enc->setBytes(&p.write_pos,4, 7);
            enc->dispatchThreadgroups(MTL::Size(p.n_kv_heads, 1, 1),
                                      MTL::Size(1024, 1, 1));
        }
        enc->endEncoding();
    }

    // 4.5. KV cache write: stash rotated K and V into the layer's cache.
    //      Skipped for kv-shared layers (they read from source layer's cache).
    if (!p.is_kv_shared) {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.kv_cache_write);
        enc->setBuffer(B.k_tmp,   0, 0);
        enc->setBuffer(B.v_tmp,   0, 1);
        enc->setBuffer(B.k_cache, 0, 2);
        enc->setBuffer(B.v_cache, 0, 3);
        enc->setBytes(&p.batch,        4, 4);
        enc->setBytes(&p.n_kv_heads,   4, 5);
        enc->setBytes(&p.head_dim,     4, 6);
        enc->setBytes(&p.seq,          4, 7);
        enc->setBytes(&p.write_pos,    4, 8);
        enc->setBytes(&p.cache_size,   4, 9);
        const uint32_t D4 = p.head_dim / 4;
        enc->dispatchThreads(MTL::Size(D4, p.seq, p.batch * p.n_kv_heads),
                             MTL::Size(32, 4, 1));
        enc->endEncoding();
    }

    // 5. Attention (cache-aware: reads from k_cache/v_cache).
    {
        auto* enc = cmd->computeCommandEncoder();
        if (p.is_global) {
            enc->setComputePipelineState(P.attn_global);
            enc->setBuffer(B.q_norm,   0, 0);
            enc->setBuffer(B.k_cache,  0, 1);
            enc->setBuffer(B.v_cache,  0, 2);
            enc->setBuffer(B.attn_out, 0, 3);
            enc->setBytes(&p.seq,           4, 4);
            enc->setBytes(&p.kv_len,        4, 5);
            enc->setBytes(&p.n_heads,       4, 6);
            enc->setBytes(&p.n_kv_heads,    4, 7);
            enc->setBytes(&p.kv_buf_start,  4, 8);
            enc->setBytes(&p.cache_size,    4, 9);
        } else {
            enc->setComputePipelineState(P.attn_local);
            enc->setBuffer(B.q_norm,   0, 0);
            enc->setBuffer(B.k_cache,  0, 1);
            enc->setBuffer(B.v_cache,  0, 2);
            enc->setBuffer(B.attn_out, 0, 3);
            enc->setBytes(&p.seq,           4, 4);
            enc->setBytes(&p.kv_len,        4, 5);
            enc->setBytes(&p.n_heads,       4, 6);
            enc->setBytes(&p.n_kv_heads,    4, 7);
            enc->setBytes(&p.window,        4, 8);
            enc->setBytes(&p.kv_buf_start,  4, 9);
            enc->setBytes(&p.cache_size,    4, 10);
        }
        enc->dispatchThreadgroups(
            MTL::Size(p.n_heads, (p.seq + 3) / 4, p.batch),
            MTL::Size(128, 1, 1));
        enc->endEncoding();
    }

    // 6. Output projection
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.gemm);
        const uint32_t M = p.batch * p.seq;
        const uint32_t K_v = p.n_heads * p.head_dim;
        const uint32_t N_v = p.d_model;
        uint32_t ldA = K_v, ldB = N_v, ldC = N_v;
        int transA = 0, transB = 0, has_bias = 0;
        enc->setBuffer(B.attn_out, 0,         0);
        enc->setBuffer(B.w_out,    off_w_out, 1);
        enc->setBuffer(B.o_proj,   0,         2);
        enc->setBytes(&M,        4, 3);
        enc->setBytes(&N_v,      4, 4);
        enc->setBytes(&K_v,      4, 5);
        enc->setBytes(&ldA,      4, 6);
        enc->setBytes(&ldB,      4, 7);
        enc->setBytes(&ldC,      4, 8);
        enc->setBytes(&transA,   4, 9);
        enc->setBytes(&transB,   4, 10);
        enc->setBytes(&has_bias, 4, 11);
        enc->setBuffer(B.o_proj, 0, 12);
        enc->dispatchThreadgroups(MTL::Size((N_v + 63) / 64, (M + 63) / 64, 1),
                                  MTL::Size(64, 1, 1));
        enc->endEncoding();
    }

    // 7. Post-attn RMSNorm
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.rmsnorm);
        enc->setBuffer(B.o_proj,           0,        0);
        enc->setBuffer(B.w_post_attn_norm, off_norm, 1);
        enc->setBuffer(B.y_attn,           0,        2);
        uint32_t rows = p.batch * p.seq;
        enc->setBytes(&rows,      4, 3);
        enc->setBytes(&p.d_model, 4, 4);
        enc->setBytes(&p.eps,     4, 5);
        enc->dispatchThreadgroups(MTL::Size(1, (rows + 3) / 4, 1),
                                  MTL::Size(128, 1, 1));
        enc->endEncoding();
    }

    // 8+9. Fused residual + pre-MLP RMSNorm.
    //     y_attn = y_attn + x;  m_in = RMSNorm(y_attn, w_pre_mlp_norm).
    //     Replaces two kernels (add_f16 + rmsnorm) — 1.57–2.34× faster.
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.add_rmsnorm);
        enc->setBuffer(B.x,               0,        0);
        enc->setBuffer(B.y_attn,          0,        1);
        enc->setBuffer(B.w_pre_feedforward_layernorm, off_norm, 2);
        enc->setBuffer(B.y_attn,          0,        3);
        enc->setBuffer(B.m_in,            0,        4);
        uint32_t rows = p.batch * p.seq;
        enc->setBytes(&rows,      4, 5);
        enc->setBytes(&p.d_model, 4, 6);
        enc->setBytes(&p.eps,     4, 7);
        enc->dispatchThreadgroups(MTL::Size(1, rows, 1), MTL::Size(128, 1, 1));
        enc->endEncoding();
    }

    // 10. Dense GeGLU MLP. (For 26B-A4B MoE: caller routes through moe_block.h instead.)
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.gated_mlp_gelu);
        enc->setBuffer(B.m_in,   0,          0);
        enc->setBuffer(B.w_gate, off_w_gate, 1);
        enc->setBuffer(B.w_up,   off_w_gate, 2);   // same per-layer slab size as w_gate
        enc->setBuffer(B.w_down, off_w_down, 3);
        enc->setBuffer(B.m_out,  0,          4);
        uint32_t M = p.batch * p.seq;
        uint32_t N_v = p.d_model;
        uint32_t K_v = p.d_model;
        enc->setBytes(&M,        4, 5);
        enc->setBytes(&N_v,      4, 6);
        enc->setBytes(&K_v,      4, 7);
        enc->setBytes(&p.n_int,  4, 8);
        enc->dispatchThreadgroups(MTL::Size((N_v + 63) / 64, (M + 63) / 64, 1),
                                  MTL::Size(128, 1, 1));
        enc->endEncoding();
    }

    // 11. Post-MLP RMSNorm
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.rmsnorm);
        enc->setBuffer(B.m_out,           0,        0);
        enc->setBuffer(B.w_post_feedforward_layernorm, off_norm, 1);
        enc->setBuffer(B.y_out,           0,        2);
        uint32_t rows = p.batch * p.seq;
        enc->setBytes(&rows,      4, 3);
        enc->setBytes(&p.d_model, 4, 4);
        enc->setBytes(&p.eps,     4, 5);
        enc->dispatchThreadgroups(MTL::Size(1, (rows + 3) / 4, 1),
                                  MTL::Size(128, 1, 1));
        enc->endEncoding();
    }

    // 12. Residual: y_out += y_attn
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.add);
        enc->setBuffer(B.y_attn, 0, 0);
        enc->setBuffer(B.y_out,  0, 1);
        enc->setBuffer(B.y_out,  0, 2);
        uint32_t n = p.batch * p.seq * p.d_model;
        enc->setBytes(&n, 4, 3);
        uint32_t total = (n / 4u) + (n & 3u);
        enc->dispatchThreadgroups(MTL::Size((total + 127) / 128, 1, 1),
                                  MTL::Size(128, 1, 1));
        enc->endEncoding();
    }
}

// PLE inject: applied after the main residual stream update for each layer.
// Mirrors HF Gemma3nTextDecoderLayer.forward steps after `corrected_predictions`:
//   first = act(per_layer_input_gate(x)) * ple_slice
//   first = per_layer_projection(first)
//   first = post_per_layer_input_norm(first)
//   residual += layer_scalar * first
struct PLELayerBuffers {
    MTL::Buffer* residual;           // (T, d_model) read+write
    MTL::Buffer* per_layer_inputs;   // (T, n_layers, PLE_dim)
    MTL::Buffer* w_per_layer_input_gate;      // (n_layers, PLE_dim, d_model)
    MTL::Buffer* w_per_layer_projection;      // (n_layers, d_model, PLE_dim)
    MTL::Buffer* w_post_per_layer_input_norm; // (n_layers, d_model)
    MTL::Buffer* w_layer_scalar;              // (n_layers,) fp32
    MTL::Buffer* ple_gate_out;       // (T, PLE_dim) scratch
    MTL::Buffer* ple_gated;          // (T, PLE_dim) scratch
    MTL::Buffer* ple_proj_back;      // (T, d_model) scratch
};

inline void dispatch_ple_inject(
    MTL::CommandBuffer*      cmd,
    const LayerPSOs&         P,
    const PLELayerBuffers&   B,
    const LayerParams&       p,
    uint32_t                 n_layers)
{
    const uint32_t L = p.layer_idx;
    const uint32_t T = p.batch * p.seq;
    const size_t off_gate    = (size_t)L * p.ple_dim * p.d_model * 2;
    const size_t off_proj    = (size_t)L * p.d_model * p.ple_dim * 2;
    const size_t off_norm_pp = (size_t)L * p.d_model * 2;
    const size_t off_scalar  = (size_t)L * sizeof(float);

    // 1. per_layer_input_gate: (T, d_model) @ (d_model, PLE_dim) → (T, PLE_dim)
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.gemm);
        const uint32_t M = T;
        const uint32_t K_v = p.d_model;
        const uint32_t N_v = p.ple_dim;
        uint32_t ldA = K_v, ldB = N_v, ldC = N_v;
        int transA = 0, transB = 0, has_bias = 0;
        enc->setBuffer(B.residual,                0,        0);
        enc->setBuffer(B.w_per_layer_input_gate,  off_gate, 1);
        enc->setBuffer(B.ple_gate_out,            0,        2);
        enc->setBytes(&M,        4, 3);
        enc->setBytes(&N_v,      4, 4);
        enc->setBytes(&K_v,      4, 5);
        enc->setBytes(&ldA,      4, 6);
        enc->setBytes(&ldB,      4, 7);
        enc->setBytes(&ldC,      4, 8);
        enc->setBytes(&transA,   4, 9);
        enc->setBytes(&transB,   4, 10);
        enc->setBytes(&has_bias, 4, 11);
        enc->setBuffer(B.ple_gate_out, 0, 12);
        enc->dispatchThreadgroups(MTL::Size((N_v + 63) / 64, (M + 63) / 64, 1),
                                  MTL::Size(64, 1, 1));
        enc->endEncoding();
    }

    // 2. ple_gate_act: gated = gelu_approx(gate_out) * ple_slice[L]
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.ple_gate_act);
        enc->setBuffer(B.ple_gate_out,     0, 0);
        enc->setBuffer(B.per_layer_inputs, 0, 1);
        enc->setBuffer(B.ple_gated,        0, 2);
        enc->setBytes(&T,         4, 3);
        enc->setBytes(&n_layers,  4, 4);
        enc->setBytes(&p.ple_dim, 4, 5);
        enc->setBytes(&L,         4, 6);
        const uint32_t P4 = p.ple_dim / 4u;
        enc->dispatchThreads(MTL::Size(P4, T, 1), MTL::Size(64, 1, 1));
        enc->endEncoding();
    }

    // 3. per_layer_projection: (T, PLE_dim) @ (PLE_dim, d_model) → (T, d_model)
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.gemm);
        const uint32_t M = T;
        const uint32_t K_v = p.ple_dim;
        const uint32_t N_v = p.d_model;
        uint32_t ldA = K_v, ldB = N_v, ldC = N_v;
        int transA = 0, transB = 0, has_bias = 0;
        enc->setBuffer(B.ple_gated,                0,        0);
        enc->setBuffer(B.w_per_layer_projection,   off_proj, 1);
        enc->setBuffer(B.ple_proj_back,            0,        2);
        enc->setBytes(&M,        4, 3);
        enc->setBytes(&N_v,      4, 4);
        enc->setBytes(&K_v,      4, 5);
        enc->setBytes(&ldA,      4, 6);
        enc->setBytes(&ldB,      4, 7);
        enc->setBytes(&ldC,      4, 8);
        enc->setBytes(&transA,   4, 9);
        enc->setBytes(&transB,   4, 10);
        enc->setBytes(&has_bias, 4, 11);
        enc->setBuffer(B.ple_proj_back, 0, 12);
        enc->dispatchThreadgroups(MTL::Size((N_v + 63) / 64, (M + 63) / 64, 1),
                                  MTL::Size(64, 1, 1));
        enc->endEncoding();
    }

    // 4. fused rmsnorm + scaled add into residual
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.ple_inject);
        enc->setBuffer(B.ple_proj_back,                0,           0);
        enc->setBuffer(B.w_post_per_layer_input_norm,  off_norm_pp, 1);
        enc->setBuffer(B.residual,                     0,           2);
        enc->setBytes(&T,         4, 3);
        enc->setBytes(&p.d_model, 4, 4);
        enc->setBytes(&p.eps,     4, 5);
        enc->setBuffer(B.w_layer_scalar, off_scalar, 6);
        enc->dispatchThreadgroups(MTL::Size(T, 1, 1), MTL::Size(128, 1, 1));
        enc->endEncoding();
    }
}


// ──────────────────────────────────────────────────────────────────────
//  Model-level
// ──────────────────────────────────────────────────────────────────────

struct ModelParams {
    uint32_t batch              = 1;
    uint32_t seq                = 1;
    uint32_t n_layers           = 32;
    uint32_t local_period       = 6;
    uint32_t d_model            = 4608;
    uint32_t n_int              = 12288;
    uint32_t n_heads            = 16;
    uint32_t n_kv_heads_local   = 16;
    uint32_t n_kv_heads_global  = 4;
    uint32_t head_dim_local     = 256;
    uint32_t head_dim_global    = 512;
    uint32_t window             = 4096;
    uint32_t cache_max          = 8192;     // physical buffer dim for global cache
    uint32_t prope_p_pairs      = 64;
    uint32_t vocab_size         = 256000;
    uint32_t ple_dim            = 256;
    bool     has_ple            = true;
    float    eps                = 1e-5f;
    float    final_logit_softcap = 0.0f;   // 0 = disabled

    // Per-call: position of input[0] in the running sequence (decode tracks this).
    uint32_t current_pos        = 0;

    // Per-layer tables (owned by launcher Handle; length n_layers).
    const uint32_t* n_int_per_layer    = nullptr;  // per-layer intermediate_size
    const size_t*   mlp_gate_off_e     = nullptr;  // cumulative dm*sum n_int_l (elements)
    const size_t*   mlp_down_off_e     = nullptr;  // cumulative sum n_int_l*dm  (elements)
    const int32_t*  kv_source_layer    = nullptr;  // -1 if not shared, else source idx
};

struct ModelPSOs {
    LayerPSOs layer;
    MTL::ComputePipelineState* embedding_lookup;
    MTL::ComputePipelineState* ple_lookup;
    MTL::ComputePipelineState* argmax;
    MTL::ComputePipelineState* logit_softcap;
};

// Caller-allocated cache buffers, one pair per layer. The launcher manages
// allocation; dispatch_model just selects per-layer at iteration.
struct LayerCache {
    MTL::Buffer* k;   // (batch, n_kv_heads, cache_size, head_dim)
    MTL::Buffer* v;
};

struct ModelWeights {
    MTL::Buffer* w_embed;
    MTL::Buffer* w_ple_table;                 // (vocab, n_layers, PLE_dim), null if !has_ple
    MTL::Buffer* w_per_layer_input_gate;      // (n_layers, PLE_dim, d_model)
    MTL::Buffer* w_per_layer_projection;      // (n_layers, d_model, PLE_dim)
    MTL::Buffer* w_layer_scalar;              // (n_layers,) fp32
    MTL::Buffer* w_post_per_layer_input_norm; // (n_layers, d_model)
    MTL::Buffer* w_pre_attn_norm;
    MTL::Buffer* w_post_attn_norm;
    MTL::Buffer* w_pre_feedforward_layernorm;
    MTL::Buffer* w_post_feedforward_layernorm;
    MTL::Buffer* w_final_norm;
    MTL::Buffer* w_qkv;
    MTL::Buffer* w_out;
    MTL::Buffer* gamma_q;
    MTL::Buffer* gamma_k;
    MTL::Buffer* w_gate;
    MTL::Buffer* w_up;
    MTL::Buffer* w_down;
    MTL::Buffer* cos_local;
    MTL::Buffer* sin_local;
    MTL::Buffer* cos_global;
    MTL::Buffer* sin_global;

    // Per-layer caches (vector of pairs). Caller fills this.
    const LayerCache* layer_caches; // length n_layers
};

struct ModelBuffers {
    MTL::Buffer* input_ids;
    MTL::Buffer* output_id;

    MTL::Buffer* x_a;
    MTL::Buffer* x_b;
    MTL::Buffer* logits;

    // Layer scratch (sized for global head_dim worst-case)
    MTL::Buffer* x_norm;
    MTL::Buffer* qkv_packed;
    MTL::Buffer* q_norm;
    MTL::Buffer* k_tmp;
    MTL::Buffer* v_tmp;
    MTL::Buffer* attn_out;
    MTL::Buffer* o_proj;
    MTL::Buffer* y_attn;
    MTL::Buffer* m_in;
    MTL::Buffer* m_out;
    MTL::Buffer* y_out;

    // PLE scratch
    MTL::Buffer* per_layer_inputs;   // (T, n_layers, PLE_dim)
    MTL::Buffer* ple_gate_out;       // (T, PLE_dim)
    MTL::Buffer* ple_gated;          // (T, PLE_dim)
    MTL::Buffer* ple_proj_back;      // (T, d_model)
};

inline void dispatch_model(
    MTL::CommandBuffer* cmd,
    const ModelPSOs&    P,
    const ModelWeights& W,
    ModelBuffers&       B,
    const ModelParams&  M)
{
    const uint32_t T = M.batch * M.seq;

    // A. Input embedding
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

    // A.1 Per-Layer Embedding table lookup (one-time per forward).
    if (M.has_ple) {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.ple_lookup);
        enc->setBuffer(W.w_ple_table,      0, 0);
        enc->setBuffer(B.input_ids,        0, 1);
        enc->setBuffer(B.per_layer_inputs, 0, 2);
        enc->setBytes(&T,             4, 3);
        enc->setBytes(&M.n_layers,    4, 4);
        enc->setBytes(&M.ple_dim,     4, 5);
        enc->setBytes(&M.vocab_size,  4, 6);
        const uint32_t P4 = M.ple_dim / 4u;
        enc->dispatchThreads(MTL::Size(P4, M.n_layers, T),
                             MTL::Size(32, 1, 1));
        enc->endEncoding();
    }

    // B. Layer stack (ping-pong x_a ↔ x_b).
    MTL::Buffer* cur = B.x_a;
    MTL::Buffer* nxt = B.x_b;

    for (uint32_t L = 0; L < M.n_layers; ++L) {
        const bool is_global = ((L % M.local_period) == (M.local_period - 1));

        // Per-layer GQA + cache geometry.
        const uint32_t n_kv  = is_global ? M.n_kv_heads_global : M.n_kv_heads_local;
        const uint32_t hd    = is_global ? M.head_dim_global   : M.head_dim_local;
        const uint32_t csize = is_global ? M.cache_max         : M.window;

        // KV-cache addressing for THIS step.
        // After this layer's cache write completes, cache holds [cur_pos, cur_pos+seq).
        // For attention we read [cur_pos+seq - kv_len, cur_pos+seq).
        const uint32_t total_after = M.current_pos + M.seq;
        const uint32_t kv_len      = (total_after < csize) ? total_after : csize;
        const uint32_t logical_first = total_after - kv_len;
        const uint32_t kv_buf_start  = logical_first % csize;

        LayerParams lp;
        lp.batch          = M.batch;
        lp.seq            = M.seq;
        lp.n_heads        = M.n_heads;
        lp.n_kv_heads     = n_kv;
        lp.n_kv_heads_max = (M.n_kv_heads_local > M.n_kv_heads_global)
                              ? M.n_kv_heads_local : M.n_kv_heads_global;
        lp.head_dim       = hd;
        lp.head_dim_max   = (M.head_dim_local > M.head_dim_global)
                              ? M.head_dim_local : M.head_dim_global;
        lp.window         = M.window;
        lp.is_global      = is_global;
        lp.prope_p_pairs  = M.prope_p_pairs;
        lp.d_model        = M.d_model;
        lp.n_int          = M.n_int_per_layer ? M.n_int_per_layer[L] : M.n_int;
        lp.ple_dim        = M.ple_dim;
        lp.eps            = M.eps;
        lp.layer_idx      = L;
        lp.kv_buf_start   = kv_buf_start;
        lp.kv_len         = kv_len;
        lp.cache_size     = csize;
        lp.write_pos      = M.current_pos;
        lp.off_w_gate_e   = M.mlp_gate_off_e ? M.mlp_gate_off_e[L] : (size_t)L * M.d_model * M.n_int;
        lp.off_w_down_e   = M.mlp_down_off_e ? M.mlp_down_off_e[L] : (size_t)L * M.n_int * M.d_model;
        const int32_t kv_src = M.kv_source_layer ? M.kv_source_layer[L] : -1;
        lp.is_kv_shared   = (kv_src >= 0);

        LayerBuffers lb;
        lb.x = cur;
        lb.w_pre_attn_norm  = W.w_pre_attn_norm;
        lb.w_post_attn_norm = W.w_post_attn_norm;
        lb.w_pre_feedforward_layernorm  = W.w_pre_feedforward_layernorm;
        lb.w_post_feedforward_layernorm = W.w_post_feedforward_layernorm;
        lb.w_per_layer_input_gate       = W.w_per_layer_input_gate;
        lb.w_per_layer_projection       = W.w_per_layer_projection;
        lb.w_post_per_layer_input_norm  = W.w_post_per_layer_input_norm;
        lb.w_layer_scalar               = W.w_layer_scalar;
        lb.per_layer_inputs             = B.per_layer_inputs;
        lb.ple_gate_out                 = B.ple_gate_out;
        lb.ple_gated                    = B.ple_gated;
        lb.ple_proj_back                = B.ple_proj_back;
        lb.w_qkv            = W.w_qkv;
        lb.gamma_q          = W.gamma_q;
        lb.gamma_k          = W.gamma_k;
        lb.cos              = is_global ? W.cos_global : W.cos_local;
        lb.sin              = is_global ? W.sin_global : W.sin_local;
        lb.w_out            = W.w_out;
        lb.w_gate           = W.w_gate;
        lb.w_up             = W.w_up;
        lb.w_down           = W.w_down;
        {
            const uint32_t kv_L = (kv_src >= 0) ? (uint32_t)kv_src : L;
            lb.k_cache      = W.layer_caches[kv_L].k;
            lb.v_cache      = W.layer_caches[kv_L].v;
        }
        lb.x_norm     = B.x_norm;
        lb.qkv_packed = B.qkv_packed;
        lb.q_norm     = B.q_norm;
        lb.k_tmp      = B.k_tmp;
        lb.v_tmp      = B.v_tmp;
        lb.attn_out   = B.attn_out;
        lb.o_proj     = B.o_proj;
        lb.y_attn     = B.y_attn;
        lb.m_in       = B.m_in;
        lb.m_out      = B.m_out;
        lb.y_out      = nxt;

        dispatch_layer(cmd, P.layer, lb, lp);

        if (M.has_ple) {
            PLELayerBuffers pb;
            pb.residual                   = nxt;
            pb.per_layer_inputs           = B.per_layer_inputs;
            pb.w_per_layer_input_gate     = W.w_per_layer_input_gate;
            pb.w_per_layer_projection     = W.w_per_layer_projection;
            pb.w_post_per_layer_input_norm= W.w_post_per_layer_input_norm;
            pb.w_layer_scalar             = W.w_layer_scalar;
            pb.ple_gate_out               = B.ple_gate_out;
            pb.ple_gated                  = B.ple_gated;
            pb.ple_proj_back              = B.ple_proj_back;
            dispatch_ple_inject(cmd, P.layer, pb, lp, M.n_layers);
        }

        MTL::Buffer* tmp = cur; cur = nxt; nxt = tmp;
    }

    // C. Final RMSNorm
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.layer.rmsnorm);
        enc->setBuffer(cur,            0, 0);
        enc->setBuffer(W.w_final_norm, 0, 1);
        enc->setBuffer(nxt,            0, 2);
        enc->setBytes(&T,         4, 3);
        enc->setBytes(&M.d_model, 4, 4);
        enc->setBytes(&M.eps,     4, 5);
        enc->dispatchThreadgroups(MTL::Size(1, (T + 3) / 4, 1),
                                  MTL::Size(128, 1, 1));
        enc->endEncoding();
    }

    // D. LM head GEMM (tied with input embedding)
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
        enc->setBytes(&M_v,      4, 3);
        enc->setBytes(&N_v,      4, 4);
        enc->setBytes(&K_v,      4, 5);
        enc->setBytes(&ldA,      4, 6);
        enc->setBytes(&ldB,      4, 7);
        enc->setBytes(&ldC,      4, 8);
        enc->setBytes(&transA,   4, 9);
        enc->setBytes(&transB,   4, 10);
        enc->setBytes(&has_bias, 4, 11);
        enc->setBuffer(B.logits, 0, 12);
        enc->dispatchThreadgroups(MTL::Size((N_v + 63) / 64, (M_v + 63) / 64, 1),
                                  MTL::Size(64, 1, 1));
        enc->endEncoding();
    }

    // D.5 final logit softcap (Gemma4 specific, HF modeling_gemma4.py:1856-1859)
    if (M.final_logit_softcap > 0.0f) {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.logit_softcap);
        enc->setBuffer(B.logits, 0, 0);
        uint32_t n = T * M.vocab_size;
        float cap = M.final_logit_softcap;
        enc->setBytes(&n,   4, 1);
        enc->setBytes(&cap, 4, 2);
        uint32_t groups = ((n / 4u) + 127u) / 128u;
        enc->dispatchThreadgroups(MTL::Size(groups, 1, 1),
                                  MTL::Size(128, 1, 1));
        enc->endEncoding();
    }

    // E. argmax → output_id
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.argmax);
        enc->setBuffer(B.logits,   0, 0);
        enc->setBuffer(B.output_id,0, 1);
        enc->setBytes(&M.vocab_size, 4, 2);
        enc->dispatchThreadgroups(MTL::Size(T, 1, 1), MTL::Size(1024, 1, 1));
        enc->endEncoding();
    }
}

} // namespace gemma4
} // namespace meow

#endif
