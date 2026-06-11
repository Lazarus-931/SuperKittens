// Granite-4.x hybrid dispatch orchestrator.
//
// Per layer (llama.cpp granite-hybrid.cpp is the reference contract):
//   1. RMSNorm(x, attn_norm)                          -> x_norm
//   2. mixer:
//        mamba layer  — in_proj -> [z|x|B|C|dt] split -> conv1d+SiLU (O(1)
//          decode state carry) -> SSD scan -> SiLU(z)-gated RMSNorm -> out_proj
//          (reuses the mamba2 family kernels verbatim)
//        attn layer   — q/k/v Q8_0 matvecs, NoPE (no RoPE dispatch at all),
//          Q pre-scaled so mha_causal's 1/sqrt(D) becomes attention_scale,
//          kv-cache write, mha_causal (D=128), o-proj
//   3. ffn_inp = x + residual_scale * mixer_out
//   4. RMSNorm(ffn_inp, ffn_norm) -> SwiGLU FFN (gate/up/silu_mul/down)
//   5. x_out = ffn_inp + residual_scale * ffn_out
// Model: embed * embedding_scale -> layers -> final RMSNorm -> tied Q8_0 head
// (last row only) -> logits / logit_scale -> argmax.

#ifndef SK_GRANITE_MODEL_H
#define SK_GRANITE_MODEL_H

#include <Metal/Metal.hpp>
#include <cstdint>
#include <cmath>
#include <vector>

namespace meow {
namespace granite {

struct PSOs {
    MTL::ComputePipelineState* rmsnorm        = nullptr;
    MTL::ComputePipelineState* rmsnorm_t1     = nullptr;  // optional T=1 fast path
    MTL::ComputePipelineState* q8_0_matvec    = nullptr;
    MTL::ComputePipelineState* split_packed   = nullptr;
    MTL::ComputePipelineState* conv1d_silu    = nullptr;
    MTL::ComputePipelineState* conv1d_silu_step    = nullptr;
    MTL::ComputePipelineState* conv_state_capture  = nullptr;
    MTL::ComputePipelineState* mamba2_ssd     = nullptr;  // handles L=1 decode (state carry)
    MTL::ComputePipelineState* gate_norm      = nullptr;
    MTL::ComputePipelineState* silu_mul       = nullptr;
    MTL::ComputePipelineState* attn           = nullptr;  // mha_causal (D=128)
    MTL::ComputePipelineState* kv_cache_write = nullptr;
    MTL::ComputePipelineState* t_seq_to_head  = nullptr;
    MTL::ComputePipelineState* t_head_to_seq  = nullptr;
    MTL::ComputePipelineState* scale          = nullptr;  // granite_scale_f16
    MTL::ComputePipelineState* add_scaled     = nullptr;  // granite_add_scaled_f16
    MTL::ComputePipelineState* embedding_lookup = nullptr;
    MTL::ComputePipelineState* argmax         = nullptr;
};

// Per-layer weights. Mamba and attention members are mutually exclusive
// (nullptr for the other type); FFN members exist on every layer.
struct LayerWeights {
    bool is_attn = false;
    // mamba2 (Q8_0 projections, fp16 small tensors)
    MTL::Buffer* ssm_in   = nullptr;  // Q8_0 (2E+2GN+H, D)
    MTL::Buffer* ssm_out  = nullptr;  // Q8_0 (D, E)
    MTL::Buffer* conv_w   = nullptr;  // fp16 (C_in, K) — conv1d_silu reads w[c*K+k]
    MTL::Buffer* conv_b   = nullptr;  // fp16 (C_in,)
    MTL::Buffer* dt_bias  = nullptr;  // fp16 (H,)
    MTL::Buffer* A_log    = nullptr;  // fp16 (H,) = log(-ssm_a_gguf)
    MTL::Buffer* ssm_D    = nullptr;  // fp16 (H,)
    MTL::Buffer* ssm_norm = nullptr;  // fp16 (E,)
    // attention (Q8_0)
    MTL::Buffer* wq = nullptr;        // (qN, D)
    MTL::Buffer* wk = nullptr;        // (kvN, D)
    MTL::Buffer* wv = nullptr;        // (kvN, D)
    MTL::Buffer* wo = nullptr;        // (D, qN)
    // FFN (Q8_0, every layer)
    MTL::Buffer* gate = nullptr;      // (n_int, D)
    MTL::Buffer* up   = nullptr;      // (n_int, D)
    MTL::Buffer* down = nullptr;      // (D, n_int)
};

struct LayerState {
    // attention layers
    MTL::Buffer* k_cache = nullptr;   // (cache_max, n_kv, hd) fp16
    MTL::Buffer* v_cache = nullptr;
    // mamba layers
    MTL::Buffer* conv_state = nullptr;  // (K-1, C_in) fp16
    MTL::Buffer* ssm_state  = nullptr;  // (H, P, N) fp32
};

struct Weights {
    MTL::Buffer* embed      = nullptr;  // fp16 (V, D) dequant — lookup table
    MTL::Buffer* head_q8    = nullptr;  // Q8_0 (V, D) raw — tied LM head matvec
    MTL::Buffer* attn_norm  = nullptr;  // fp16 (n_layers, D) pre-mixer norm
    MTL::Buffer* ffn_norm   = nullptr;  // fp16 (n_layers, D)
    MTL::Buffer* final_norm = nullptr;  // fp16 (D,)
    std::vector<LayerWeights> layers;
};

struct Buffers {
    MTL::Buffer* input_ids = nullptr;
    MTL::Buffer* output_id = nullptr;
    MTL::Buffer* x_a = nullptr;       // residual ping
    MTL::Buffer* x_b = nullptr;       // residual pong
    MTL::Buffer* x_norm = nullptr;
    MTL::Buffer* ffn_inp = nullptr;   // post-mixer residual
    MTL::Buffer* mixer_out = nullptr; // (T, D) mixer block output
    MTL::Buffer* logits = nullptr;    // (seq_max, V) fp16
    // mamba scratch
    MTL::Buffer* in_proj_out = nullptr;  // (T, 2E+2GN+H)
    MTL::Buffer* z = nullptr;            // (T, E)
    MTL::Buffer* xBC = nullptr;          // (T, C_in)
    MTL::Buffer* dt_raw = nullptr;       // (T, H)
    MTL::Buffer* xBC_post = nullptr;     // (T, C_in)  (also split scratch)
    MTL::Buffer* ssd_out = nullptr;      // (T, E)
    MTL::Buffer* gated = nullptr;        // (T, E)
    // attention scratch
    MTL::Buffer* q = nullptr;            // (T, qN)
    MTL::Buffer* k_tmp = nullptr;        // (T, kvN)
    MTL::Buffer* v_tmp = nullptr;        // (T, kvN)
    MTL::Buffer* attn_out = nullptr;     // (T, qN)
    MTL::Buffer* q_th = nullptr;         // head-major prefill scratch
    MTL::Buffer* k_th = nullptr;
    MTL::Buffer* v_th = nullptr;
    MTL::Buffer* attn_out_seq = nullptr;
    // FFN scratch
    MTL::Buffer* gate_buf = nullptr;     // (T, n_int)
    MTL::Buffer* up_buf = nullptr;       // (T, n_int)
    MTL::Buffer* mlp_out = nullptr;      // (T, D)
};

struct Params {
    uint32_t seq         = 1;
    uint32_t n_layers    = 40;
    uint32_t d_model     = 1536;
    uint32_t n_heads     = 12;
    uint32_t n_kv_heads  = 4;
    uint32_t head_dim    = 128;
    uint32_t n_int       = 4096;
    uint32_t d_inner     = 3072;   // E
    uint32_t ssm_heads   = 48;     // H
    uint32_t ssm_pdim    = 64;     // P
    uint32_t ssm_state   = 128;    // N
    uint32_t ssm_groups  = 1;      // G
    uint32_t ssm_conv    = 4;      // K
    uint32_t vocab_size  = 100352;
    uint32_t cache_max   = 4096;
    uint32_t current_pos = 0;
    float    eps         = 1e-5f;
    float    embedding_scale = 12.0f;
    float    residual_scale  = 0.22f;
    float    attention_scale = 0.0078125f;
    float    logit_scale     = 6.0f;
};

// ── encode helpers (one encoder per op; encoder boundaries order producers) ──

inline void enc_rmsnorm(MTL::CommandBuffer* cmd, const PSOs& P,
                        MTL::Buffer* x, MTL::Buffer* gamma, size_t off_g,
                        MTL::Buffer* out, uint32_t rows, uint32_t n, float eps)
{
    const bool t1 = (P.rmsnorm_t1 != nullptr) && (rows == 1u);
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(t1 ? P.rmsnorm_t1 : P.rmsnorm);
    enc->setBuffer(x,     0,     0);
    enc->setBuffer(gamma, off_g, 1);
    enc->setBuffer(out,   0,     2);
    enc->setBytes(&rows, 4, 3);
    enc->setBytes(&n,    4, 4);
    enc->setBytes(&eps,  4, 5);
    if (t1) enc->dispatchThreadgroups(MTL::Size(1, rows, 1), MTL::Size(256, 1, 1));
    else    enc->dispatchThreadgroups(MTL::Size(1, (rows + 3) / 4, 1), MTL::Size(128, 1, 1));
    enc->endEncoding();
}

// y[T,N] (row stride ldC) = x[T,K] @ W(Q8_0 [N,K] row-major)^T, per-row matvec.
inline void enc_q8_matvec(MTL::CommandBuffer* cmd, const PSOs& P,
                          MTL::Buffer* W, MTL::Buffer* x, MTL::Buffer* y,
                          uint32_t T, uint32_t N, uint32_t K,
                          size_t off_y = 0, uint32_t ldC = 0)
{
    if (ldC == 0) ldC = N;
    auto* enc = cmd->computeCommandEncoder();
    for (uint32_t m = 0; m < T; ++m) {
        if (m) enc->memoryBarrier(MTL::BarrierScopeBuffers);
        enc->setComputePipelineState(P.q8_0_matvec);
        enc->setBuffer(x, (size_t)m * K * 2, 0);
        enc->setBuffer(W, 0, 1);
        enc->setBuffer(y, off_y + (size_t)m * ldC * 2, 2);
        enc->setBytes(&K, 4, 3);
        enc->setBytes(&N, 4, 4);
        enc->dispatchThreadgroups(MTL::Size((N + 1) / 2, 1, 1), MTL::Size(128, 1, 1));
    }
    enc->endEncoding();
}

inline void enc_split(MTL::CommandBuffer* cmd, const PSOs& P,
                      MTL::Buffer* src, MTL::Buffer* outA, MTL::Buffer* outB,
                      uint32_t T, uint32_t A, uint32_t B)
{
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(P.split_packed);
    enc->setBuffer(src,  0, 0);
    enc->setBuffer(outA, 0, 1);
    enc->setBuffer(outB, 0, 2);
    enc->setBytes(&T, 4, 3);
    enc->setBytes(&A, 4, 4);
    enc->setBytes(&B, 4, 5);
    enc->dispatchThreads(MTL::Size(A + B, T, 1), MTL::Size(128, 1, 1));
    enc->endEncoding();
}

inline void enc_scale(MTL::CommandBuffer* cmd, const PSOs& P,
                      MTL::Buffer* x, float s, uint32_t n, size_t off_x = 0)
{
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(P.scale);
    enc->setBuffer(x, off_x, 0);
    enc->setBytes(&s, 4, 1);
    enc->setBytes(&n, 4, 2);
    enc->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
    enc->endEncoding();
}

inline void enc_add_scaled(MTL::CommandBuffer* cmd, const PSOs& P,
                           MTL::Buffer* a, MTL::Buffer* b, MTL::Buffer* y,
                           float s, uint32_t n)
{
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(P.add_scaled);
    enc->setBuffer(a, 0, 0);
    enc->setBuffer(b, 0, 1);
    enc->setBuffer(y, 0, 2);
    enc->setBytes(&s, 4, 3);
    enc->setBytes(&n, 4, 4);
    enc->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
    enc->endEncoding();
}

inline void enc_transpose(MTL::CommandBuffer* cmd, MTL::ComputePipelineState* pso,
                          MTL::Buffer* src, MTL::Buffer* dst,
                          uint32_t T, uint32_t H, uint32_t D)
{
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(src, 0, 0);
    enc->setBuffer(dst, 0, 1);
    enc->setBytes(&T, 4, 2);
    enc->setBytes(&H, 4, 3);
    enc->setBytes(&D, 4, 4);
    enc->dispatchThreads(MTL::Size(D, T, H), MTL::Size(32, 1, 1));
    enc->endEncoding();
}

// ── mamba2 mixer (reuses the models/ssm/mamba2 kernels) ─────────────────────

inline void dispatch_mamba_mixer(MTL::CommandBuffer* cmd, const PSOs& P,
                                 const Params& p, const LayerWeights& W,
                                 const LayerState& S, Buffers& B)
{
    const uint32_t T    = p.seq;
    const uint32_t D    = p.d_model;
    const uint32_t E    = p.d_inner;
    const uint32_t H    = p.ssm_heads;
    const uint32_t Pd   = p.ssm_pdim;
    const uint32_t G    = p.ssm_groups;
    const uint32_t N    = p.ssm_state;
    const uint32_t K    = p.ssm_conv;
    const uint32_t C_in = E + 2 * G * N;
    const uint32_t IN_OUT = 2 * E + 2 * G * N + H;
    const bool decode = (T == 1) && (p.current_pos > 0);

    enc_q8_matvec(cmd, P, W.ssm_in, B.x_norm, B.in_proj_out, T, IN_OUT, D);

    // [z(E) | xBC(C_in) | dt(H)] — two splits, xBC_post doubles as scratch.
    enc_split(cmd, P, B.in_proj_out, B.z, B.xBC_post, T, E, C_in + H);
    enc_split(cmd, P, B.xBC_post, B.xBC, B.dt_raw, T, C_in, H);

    if (decode) {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.conv1d_silu_step);
        enc->setBuffer(B.xBC,      0, 0);
        enc->setBuffer(W.conv_w,   0, 1);
        enc->setBuffer(W.conv_b,   0, 2);
        enc->setBuffer(B.xBC_post, 0, 3);
        enc->setBuffer(S.conv_state, 0, 4);
        const uint32_t one = 1;
        enc->setBytes(&one,  4, 5);
        enc->setBytes(&C_in, 4, 6);
        enc->setBytes(&K,    4, 7);
        enc->dispatchThreads(MTL::Size(C_in, 1, 1), MTL::Size(128, 1, 1));
        enc->endEncoding();
    } else {
        {
            auto* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(P.conv1d_silu);
            enc->setBuffer(B.xBC,      0, 0);
            enc->setBuffer(W.conv_w,   0, 1);
            enc->setBuffer(W.conv_b,   0, 2);
            enc->setBuffer(B.xBC_post, 0, 3);
            const uint32_t one = 1;
            enc->setBytes(&one,  4, 4);
            enc->setBytes(&T,    4, 5);
            enc->setBytes(&C_in, 4, 6);
            enc->dispatchThreadgroups(MTL::Size(1, (T + 3) / 4, 1), MTL::Size(128, 1, 1));
            enc->endEncoding();
        }
        {
            auto* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(P.conv_state_capture);
            enc->setBuffer(B.xBC,        0, 0);
            enc->setBuffer(S.conv_state, 0, 1);
            const uint32_t one = 1;
            enc->setBytes(&one,  4, 2);
            enc->setBytes(&T,    4, 3);
            enc->setBytes(&C_in, 4, 4);
            enc->setBytes(&K,    4, 5);
            enc->dispatchThreads(MTL::Size(C_in, K - 1, 1), MTL::Size(128, 1, 1));
            enc->endEncoding();
        }
    }

    // SSD scan: x/B/C alias xBC_post (token stride C_in). Granite has no
    // time_step_limit -> (0, +inf) matches ggml's softplus-only dt.
    {
        const size_t off_B_in = (size_t)E * 2;
        const size_t off_C_in = (size_t)(E + G * N) * 2;
        const float dt_min = 0.0f, dt_max = INFINITY;
        const uint32_t one = 1;
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.mamba2_ssd);
        enc->setBuffer(B.xBC_post, 0,        0);
        enc->setBuffer(B.dt_raw,   0,        1);
        enc->setBuffer(W.A_log,    0,        2);
        enc->setBuffer(B.xBC_post, off_B_in, 3);
        enc->setBuffer(B.xBC_post, off_C_in, 4);
        enc->setBuffer(W.ssm_D,    0,        5);
        enc->setBuffer(W.dt_bias,  0,        6);
        enc->setBuffer(B.ssd_out,  0,        7);
        enc->setBuffer(S.ssm_state, 0,       8);
        enc->setBytes(&one,    4, 9);
        enc->setBytes(&T,      4, 10);
        enc->setBytes(&H,      4, 11);
        enc->setBytes(&Pd,     4, 12);
        enc->setBytes(&G,      4, 13);
        enc->setBytes(&N,      4, 14);
        enc->setBytes(&dt_min, 4, 15);
        enc->setBytes(&dt_max, 4, 16);
        enc->setBytes(&C_in,   4, 17);
        enc->dispatchThreadgroups(MTL::Size(H, Pd, 1), MTL::Size(N, 1, 1));
        enc->endEncoding();
    }

    // y = RMSNorm(ssd_out * SiLU(z)) * ssm_norm  (G=1: norm over full E)
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.gate_norm);
        enc->setBuffer(B.ssd_out, 0, 0);
        enc->setBuffer(B.z,       0, 1);
        enc->setBuffer(W.ssm_norm, 0, 2);
        enc->setBuffer(B.gated,   0, 3);
        enc->setBytes(&T,     4, 4);
        enc->setBytes(&E,     4, 5);
        enc->setBytes(&p.eps, 4, 6);
        enc->dispatchThreadgroups(MTL::Size(1, (T + 3) / 4, 1), MTL::Size(128, 1, 1));
        enc->endEncoding();
    }

    enc_q8_matvec(cmd, P, W.ssm_out, B.gated, B.mixer_out, T, D, E);
}

// ── attention mixer (NoPE; Q pre-scaled for granite's attention_scale) ──────

inline void dispatch_attn_mixer(MTL::CommandBuffer* cmd, const PSOs& P,
                                const Params& p, const LayerWeights& W,
                                const LayerState& S, Buffers& B)
{
    const uint32_t T   = p.seq;
    const uint32_t D   = p.d_model;
    const uint32_t hd  = p.head_dim;
    const uint32_t qN  = p.n_heads * hd;
    const uint32_t kvN = p.n_kv_heads * hd;

    enc_q8_matvec(cmd, P, W.wq, B.x_norm, B.q,     T, qN,  D);
    enc_q8_matvec(cmd, P, W.wk, B.x_norm, B.k_tmp, T, kvN, D);
    enc_q8_matvec(cmd, P, W.wv, B.x_norm, B.v_tmp, T, kvN, D);

    // mha_causal hardcodes softmax scale 1/sqrt(D); pre-scaling Q by
    // attention_scale*sqrt(D) makes the effective scale attention_scale.
    enc_scale(cmd, P, B.q, p.attention_scale * std::sqrt((float)hd), T * qN);

    MTL::Buffer* q_in = B.q;
    MTL::Buffer* k_in = B.k_tmp;
    MTL::Buffer* v_in = B.v_tmp;
    if (T > 1) {
        enc_transpose(cmd, P.t_seq_to_head, B.q,     B.q_th, T, p.n_heads,    hd);
        enc_transpose(cmd, P.t_seq_to_head, B.k_tmp, B.k_th, T, p.n_kv_heads, hd);
        enc_transpose(cmd, P.t_seq_to_head, B.v_tmp, B.v_th, T, p.n_kv_heads, hd);
        q_in = B.q_th; k_in = B.k_th; v_in = B.v_th;
    }

    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.kv_cache_write);
        enc->setBuffer(k_in,      0, 0);
        enc->setBuffer(v_in,      0, 1);
        enc->setBuffer(S.k_cache, 0, 2);
        enc->setBuffer(S.v_cache, 0, 3);
        const uint32_t one = 1;
        enc->setBytes(&one,           4, 4);
        enc->setBytes(&p.n_kv_heads,  4, 5);
        enc->setBytes(&hd,            4, 6);
        enc->setBytes(&T,             4, 7);
        enc->setBytes(&p.current_pos, 4, 8);
        enc->setBytes(&p.cache_max,   4, 9);
        enc->dispatchThreads(MTL::Size(hd / 4, T, p.n_kv_heads), MTL::Size(32, 4, 1));
        enc->endEncoding();
    }

    {
        const uint32_t kv_len = p.current_pos + T;
        const uint32_t Hg = p.n_heads / p.n_kv_heads;
        const uint32_t br = 2;
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.attn);
        enc->setBuffer(q_in,       0, 0);
        enc->setBuffer(S.k_cache,  0, 1);
        enc->setBuffer(S.v_cache,  0, 2);
        enc->setBuffer(B.attn_out, 0, 3);
        enc->setBytes(&T,            4, 4);
        enc->setBytes(&p.n_heads,    4, 5);
        enc->setBytes(&p.n_kv_heads, 4, 6);
        enc->setBytes(&kv_len,       4, 7);
        enc->setBytes(&p.cache_max,  4, 8);
        enc->dispatchThreadgroups(MTL::Size(p.n_kv_heads, (T + br - 1) / br, 1),
                                  MTL::Size(Hg * br * 32, 1, 1));
        enc->endEncoding();
    }

    MTL::Buffer* o_in = B.attn_out;
    if (T > 1) {
        enc_transpose(cmd, P.t_head_to_seq, B.attn_out, B.attn_out_seq,
                      T, p.n_heads, hd);
        o_in = B.attn_out_seq;
    }

    enc_q8_matvec(cmd, P, W.wo, o_in, B.mixer_out, T, D, qN);
}

// ── full model ───────────────────────────────────────────────────────────────

inline void dispatch_model(MTL::CommandBuffer* cmd, const PSOs& P,
                           const Weights& W, const std::vector<LayerState>& S,
                           Buffers& B, const Params& p)
{
    const uint32_t T = p.seq;
    const uint32_t D = p.d_model;

    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.embedding_lookup);
        enc->setBuffer(W.embed,     0, 0);
        enc->setBuffer(B.input_ids, 0, 1);
        enc->setBuffer(B.x_a,       0, 2);
        enc->setBytes(&T,            4, 3);
        enc->setBytes(&D,            4, 4);
        enc->setBytes(&p.vocab_size, 4, 5);
        enc->dispatchThreadgroups(MTL::Size((D / 4 + 127) / 128, T, 1),
                                  MTL::Size(128, 1, 1));
        enc->endEncoding();
    }
    enc_scale(cmd, P, B.x_a, p.embedding_scale, T * D);

    MTL::Buffer* cur = B.x_a;
    MTL::Buffer* nxt = B.x_b;
    for (uint32_t L = 0; L < p.n_layers; ++L) {
        const LayerWeights& lw = W.layers[L];
        const size_t off_norm = (size_t)L * D * 2;

        // ffn_inp/mixer scratch use the layer-shared buffers; x_norm is
        // recomputed per stage so cur/nxt are the only cross-layer carriers.
        Buffers& b = B;
        b.x_norm = B.x_norm;

        enc_rmsnorm(cmd, P, cur, W.attn_norm, off_norm, B.x_norm, T, D, p.eps);
        if (lw.is_attn) dispatch_attn_mixer(cmd, P, p, lw, S[L], b);
        else            dispatch_mamba_mixer(cmd, P, p, lw, S[L], b);

        enc_add_scaled(cmd, P, cur, B.mixer_out, B.ffn_inp, p.residual_scale, T * D);

        enc_rmsnorm(cmd, P, B.ffn_inp, W.ffn_norm, off_norm, B.x_norm, T, D, p.eps);
        enc_q8_matvec(cmd, P, lw.gate, B.x_norm, B.gate_buf, T, p.n_int, D);
        enc_q8_matvec(cmd, P, lw.up,   B.x_norm, B.up_buf,   T, p.n_int, D);
        {
            auto* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(P.silu_mul);
            enc->setBuffer(B.gate_buf, 0, 0);
            enc->setBuffer(B.up_buf,   0, 1);
            enc->setBuffer(B.up_buf,   0, 2);
            uint32_t n_total = T * p.n_int;
            enc->setBytes(&n_total, 4, 3);
            enc->dispatchThreadgroups(MTL::Size((n_total + 255) / 256, 1, 1),
                                      MTL::Size(256, 1, 1));
            enc->endEncoding();
        }
        enc_q8_matvec(cmd, P, lw.down, B.up_buf, B.mlp_out, T, D, p.n_int);

        enc_add_scaled(cmd, P, B.ffn_inp, B.mlp_out, nxt, p.residual_scale, T * D);

        MTL::Buffer* tmp = cur; cur = nxt; nxt = tmp;
    }

    enc_rmsnorm(cmd, P, cur, W.final_norm, 0, nxt, T, D, p.eps);

    // Tied Q8_0 head on the LAST row only (prefill projects just row T-1);
    // argmax is scale-invariant but the logit_scale division keeps
    // get_last_logits honest.
    {
        const uint32_t last = T - 1;
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.q8_0_matvec);
        enc->setBuffer(nxt, (size_t)last * D * 2, 0);
        enc->setBuffer(W.head_q8, 0, 1);
        enc->setBuffer(B.logits, 0, 2);
        enc->setBytes(&D, 4, 3);
        enc->setBytes(&p.vocab_size, 4, 4);
        enc->dispatchThreadgroups(MTL::Size((p.vocab_size + 1) / 2, 1, 1),
                                  MTL::Size(128, 1, 1));
        enc->endEncoding();
    }
    enc_scale(cmd, P, B.logits, 1.0f / p.logit_scale, p.vocab_size);

    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.argmax);
        enc->setBuffer(B.logits,    0, 0);
        enc->setBuffer(B.output_id, 0, 1);
        enc->setBytes(&p.vocab_size, 4, 2);
        enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1024, 1, 1));
        enc->endEncoding();
    }
}

}  // namespace granite
}  // namespace meow

#endif
