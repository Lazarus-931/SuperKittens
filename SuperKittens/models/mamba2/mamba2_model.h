// Mamba 2 (mamba2-130m-hf) dispatch orchestrator scaffold.
//
// Pipeline per layer (HF Mamba2Mixer, transformers/models/mamba2/modeling_mamba2.py):
//   1. RMSNorm(x)                                                  -> x_n
//   2. in_proj: GEMM (T, D) @ (D, IN_OUT) -> packed (T, IN_OUT)
//        IN_OUT = 2*E + 2*G*N + H
//        split: z (T,E) | xBC (T, E+2*G*N) | dt_raw (T, H)
//   3. conv1d_silu on xBC (causal depthwise, K=conv_kernel=4, then SiLU)
//        -> xBC_post (T, E+2*G*N)
//        split: x (T,E) reshape (T,H,P) | B (T,G,N) | C (T,G,N)
//   4. dt = softplus(dt_raw + dt_bias).clamp(time_step_min, time_step_max)
//        A  = -exp(A_log)                                          (H,)
//   5. SSD: mamba2_ssd / mamba2_step
//        prefill: chunked associative scan over L (HF L398-586)
//        decode : per-token recurrence using ssm_state
//        out: y (T, H, P) reshape (T, E)
//        + D[h] * x[t,h,p] skip
//   6. gate_norm: y = norm_gated(y, z, γ=mixer.norm.weight)        // SiLU(z) * RMSNorm(y)
//   7. out_proj: GEMM (T, E) @ (E, D) -> y (T, D)
//   8. residual + y -> hidden
// Final: RMSNorm(hidden) -> logits via tied lm_head (transpose embed).

#ifndef SUPERKITTENS_MAMBA2_MODEL_H
#define SUPERKITTENS_MAMBA2_MODEL_H

#include <Metal/Metal.hpp>
#include <cstdint>

namespace meow {
namespace mamba2 {

struct LayerParams {
    uint32_t batch        = 1;
    uint32_t seq          = 1;
    uint32_t d_model      = 768;
    uint32_t intermediate = 1536;   // E
    uint32_t n_heads      = 24;     // H
    uint32_t head_dim     = 64;     // P
    uint32_t state_size   = 128;    // N
    uint32_t n_groups     = 1;      // G
    uint32_t conv_kernel  = 4;
    uint32_t chunk_size   = 256;
    float    eps          = 1e-5f;
    float    dt_min       = 0.001f;
    float    dt_max       = 0.1f;

    uint32_t layer_idx    = 0;
    uint32_t is_decode    = 0;      // 0 = prefill (chunked SSD), 1 = decode (single-step)
    uint32_t pos          = 0;
};

struct LayerPSOs {
    MTL::ComputePipelineState* rmsnorm;
    MTL::ComputePipelineState* gemm;          // gemm_fp16 (M=1 fast-path for decode)
    MTL::ComputePipelineState* split_packed;
    MTL::ComputePipelineState* conv1d_silu;
    MTL::ComputePipelineState* mamba2_ssd;    // prefill chunked associative scan
    MTL::ComputePipelineState* mamba2_step;   // decode per-token recurrence
    MTL::ComputePipelineState* gate_norm;     // SiLU(z) * RMSNorm(y) γ
    MTL::ComputePipelineState* add;
};

struct ModelPSOs {
    LayerPSOs layer;
    MTL::ComputePipelineState* embedding_lookup;
    MTL::ComputePipelineState* argmax;
    // Optional 2-pass fp16 argmax PSOs (≈1.8× at V=50288).
    MTL::ComputePipelineState* argmax_partial = nullptr;
    MTL::ComputePipelineState* argmax_reduce  = nullptr;
};

struct ModelWeights {
    MTL::Buffer* w_embed;
    MTL::Buffer* w_final_norm;
    // per-layer concatenated
    MTL::Buffer* w_pre_norm;     // (n_layers, d_model)
    MTL::Buffer* w_in_proj;      // (n_layers, d_model, IN_OUT)
    MTL::Buffer* w_conv;         // (n_layers, conv_kernel, C_in)
    MTL::Buffer* w_conv_b;       // (n_layers, C_in)
    MTL::Buffer* w_dt_bias;      // (n_layers, n_heads)
    MTL::Buffer* w_A_log;        // (n_layers, n_heads)
    MTL::Buffer* w_D;            // (n_layers, n_heads)
    MTL::Buffer* w_norm;         // (n_layers, intermediate)  mixer.norm.weight
    MTL::Buffer* w_out_proj;     // (n_layers, intermediate, d_model)
};

struct LayerState {
    // conv1d sliding window of last (K-1) inputs to xBC pre-conv.
    MTL::Buffer* conv_state;     // (batch, K-1, C_in) fp16
    // SSM hidden state per head / channel / state.
    MTL::Buffer* ssm_state;      // (batch, H, P, N)   fp16
};

struct ModelBuffers {
    MTL::Buffer* tok_ids;
    MTL::Buffer* x;              // (T, D)
    MTL::Buffer* x_norm;
    MTL::Buffer* in_proj_out;    // (T, IN_OUT)
    MTL::Buffer* z;              // (T, E)
    MTL::Buffer* xBC;            // (T, E + 2*G*N)
    MTL::Buffer* dt_raw;         // (T, H)
    MTL::Buffer* xBC_post;       // (T, E + 2*G*N) after conv+silu
    MTL::Buffer* ssd_out;        // (T, E)
    MTL::Buffer* gated;          // (T, E)
    MTL::Buffer* out_proj_out;   // (T, D)
    MTL::Buffer* logits;         // (V,)
    LayerState* layer_states;    // [n_layers]

    // 2-pass argmax scratch (ceil(vocab_size/16384) partials each).
    MTL::Buffer* argmax_val_buf = nullptr;
    MTL::Buffer* argmax_idx_buf = nullptr;
};

// ── Encode helpers ──────────────────────────────────────────────────

inline void encode_gemm_mb(
    MTL::CommandBuffer* cmd, MTL::ComputePipelineState* pso,
    MTL::Buffer* A, size_t off_A,
    MTL::Buffer* B, size_t off_B,
    MTL::Buffer* C, size_t off_C,
    uint32_t M, uint32_t N, uint32_t K, int transB = 0)
{
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    uint32_t ldA = K;
    uint32_t ldB = (transB ? K : N);
    uint32_t ldC = N;
    int transA = 0, has_bias = 0;
    enc->setBuffer(A, off_A, 0);
    enc->setBuffer(B, off_B, 1);
    enc->setBuffer(C, off_C, 2);
    enc->setBytes(&M, 4, 3); enc->setBytes(&N, 4, 4); enc->setBytes(&K, 4, 5);
    enc->setBytes(&ldA, 4, 6); enc->setBytes(&ldB, 4, 7); enc->setBytes(&ldC, 4, 8);
    enc->setBytes(&transA, 4, 9); enc->setBytes(&transB, 4, 10); enc->setBytes(&has_bias, 4, 11);
    enc->setBuffer(C, off_C, 12);
    enc->dispatchThreadgroups(MTL::Size((N + 63) / 64, (M + 31) / 32, 1),
                              MTL::Size(64, 1, 1));
    enc->endEncoding();
}

inline void encode_rmsnorm_mb(
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

inline void encode_add_f16(MTL::CommandBuffer* cmd, MTL::ComputePipelineState* pso,
                           MTL::Buffer* a, MTL::Buffer* b, MTL::Buffer* y, uint32_t n)
{
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(a, 0, 0);
    enc->setBuffer(b, 0, 1);
    enc->setBuffer(y, 0, 2);
    enc->setBytes(&n, 4, 3);
    uint32_t total = (n / 4u) + (n & 3u);
    enc->dispatchThreadgroups(MTL::Size((total + 127) / 128, 1, 1),
                              MTL::Size(128, 1, 1));
    enc->endEncoding();
}

inline void encode_conv1d_silu(
    MTL::CommandBuffer* cmd, MTL::ComputePipelineState* pso,
    MTL::Buffer* x, MTL::Buffer* w, size_t off_w,
    MTL::Buffer* bias, size_t off_b,
    MTL::Buffer* y, uint32_t Bn, uint32_t L, uint32_t C)
{
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(x,    0,     0);
    enc->setBuffer(w,    off_w, 1);
    enc->setBuffer(bias, off_b, 2);
    enc->setBuffer(y,    0,     3);
    enc->setBytes(&Bn, 4, 4);
    enc->setBytes(&L,  4, 5);
    enc->setBytes(&C,  4, 6);
    uint32_t row_blocks = (L + 3) / 4;
    enc->dispatchThreadgroups(MTL::Size(Bn, row_blocks, 1),
                              MTL::Size(128, 1, 1));
    enc->endEncoding();
}

inline void encode_gate_norm(
    MTL::CommandBuffer* cmd, MTL::ComputePipelineState* pso,
    MTL::Buffer* ssm_out, MTL::Buffer* z,
    MTL::Buffer* w_norm, size_t off_w,
    MTL::Buffer* y, uint32_t rows, uint32_t E, float eps)
{
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(ssm_out, 0,     0);
    enc->setBuffer(z,       0,     1);
    enc->setBuffer(w_norm,  off_w, 2);
    enc->setBuffer(y,       0,     3);
    enc->setBytes(&rows, 4, 4);
    enc->setBytes(&E,    4, 5);
    enc->setBytes(&eps,  4, 6);
    uint32_t row_blocks = (rows + 3) / 4;
    enc->dispatchThreadgroups(MTL::Size(1, row_blocks, 1),
                              MTL::Size(128, 1, 1));
    enc->endEncoding();
}

inline void encode_split(MTL::CommandBuffer* cmd, MTL::ComputePipelineState* pso,
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

// ── Per-layer dispatch ─────────────────────────────────────────────

struct DispatchBufs {
    MTL::Buffer* x_in;
    MTL::Buffer* x_out;
    MTL::Buffer* x_norm;
    MTL::Buffer* in_proj_out;
    MTL::Buffer* z;
    MTL::Buffer* xBC;
    MTL::Buffer* dt_raw;
    MTL::Buffer* xBC_post;
    MTL::Buffer* ssd_out;
    MTL::Buffer* gated;
    MTL::Buffer* out_proj_out;
    MTL::Buffer* conv_state;
    MTL::Buffer* ssm_state;
    MTL::Buffer* w_pre_norm;
    MTL::Buffer* w_in_proj;
    MTL::Buffer* w_conv;
    MTL::Buffer* w_conv_b;
    MTL::Buffer* w_dt_bias;
    MTL::Buffer* w_A_log;
    MTL::Buffer* w_D;
    MTL::Buffer* w_norm;
    MTL::Buffer* w_out_proj;
};

inline void dispatch_layer(
    MTL::CommandBuffer* cmd,
    const LayerPSOs&    P,
    const LayerParams&  p,
    DispatchBufs&       b)
{
    const uint32_t T  = p.batch * p.seq;
    const uint32_t D  = p.d_model;
    const uint32_t E  = p.intermediate;
    const uint32_t H  = p.n_heads;
    const uint32_t Pd = p.head_dim;
    const uint32_t Gv = p.n_groups;
    const uint32_t Nv = p.state_size;
    const uint32_t K  = p.conv_kernel;
    const uint32_t IN_OUT = 2*E + 2*Gv*Nv + H;
    const uint32_t C_in   = E + 2*Gv*Nv;
    const uint32_t L  = p.layer_idx;
    const size_t fp16 = 2;

    const size_t off_pre   = (size_t)L * D * fp16;
    const size_t off_in    = (size_t)L * D * IN_OUT * fp16;
    const size_t off_conv  = (size_t)L * K * C_in * fp16;
    const size_t off_convb = (size_t)L * C_in * fp16;
    const size_t off_dtb   = (size_t)L * H * fp16;
    const size_t off_A     = (size_t)L * H * fp16;
    const size_t off_D     = (size_t)L * H * fp16;
    const size_t off_norm  = (size_t)L * E * fp16;
    const size_t off_outp  = (size_t)L * E * D * fp16;

    // 1. pre-norm
    encode_rmsnorm_mb(cmd, P.rmsnorm, b.x_in, b.w_pre_norm, off_pre,
                      b.x_norm, T, D, p.eps);

    // 2. in_proj GEMM: (T,D) x (D,IN_OUT) -> (T,IN_OUT)
    encode_gemm_mb(cmd, P.gemm, b.x_norm, 0, b.w_in_proj, off_in,
                   b.in_proj_out, 0, T, IN_OUT, D);

    // 3. Split packed [z(E) | xBC(C_in) | dt(H)]
    //    First split: z vs (xBC+dt). Write xBC+dt into xBC_post temporarily.
    encode_split(cmd, P.split_packed, b.in_proj_out, b.z, b.xBC_post,
                 T, E, C_in + H);
    //    Second split: xBC vs dt_raw.
    encode_split(cmd, P.split_packed, b.xBC_post, b.xBC, b.dt_raw,
                 T, C_in, H);

    // 4. conv1d + silu on xBC (B=1, L=seq, C=C_in).
    encode_conv1d_silu(cmd, P.conv1d_silu, b.xBC, b.w_conv, off_conv,
                       b.w_conv_b, off_convb, b.xBC_post,
                       p.batch, p.seq, C_in);

    // 5. SSD reference. xBC_post layout: [x(T,E) | B(T,G*N) | C(T,G*N)] flat.
    const size_t off_x_in = 0;
    const size_t off_B_in = (size_t)E * fp16;
    const size_t off_C_in = (size_t)(E + Gv * Nv) * fp16;
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.mamba2_ssd);
        enc->setBuffer(b.xBC_post,  off_x_in, 0);
        enc->setBuffer(b.dt_raw,    0,        1);
        enc->setBuffer(b.w_A_log,   off_A,    2);
        enc->setBuffer(b.xBC_post,  off_B_in, 3);
        enc->setBuffer(b.xBC_post,  off_C_in, 4);
        enc->setBuffer(b.w_D,       off_D,    5);
        enc->setBuffer(b.w_dt_bias, off_dtb,  6);
        enc->setBuffer(b.ssd_out,   0,        7);
        enc->setBuffer(b.ssm_state, 0,        8);
        enc->setBytes(&p.batch,  4, 9);
        enc->setBytes(&p.seq,    4, 10);
        enc->setBytes(&H,        4, 11);
        enc->setBytes(&Pd,       4, 12);
        enc->setBytes(&Gv,       4, 13);
        enc->setBytes(&Nv,       4, 14);
        enc->setBytes(&p.dt_min, 4, 15);
        enc->setBytes(&p.dt_max, 4, 16);
        enc->dispatchThreadgroups(MTL::Size(p.batch * H, Pd, 1),
                                  MTL::Size(Nv, 1, 1));
        enc->endEncoding();
    }

    // 6. gate_norm
    encode_gate_norm(cmd, P.gate_norm, b.ssd_out, b.z,
                     b.w_norm, off_norm, b.gated, T, E, p.eps);

    // 7. out_proj GEMM (T,E)x(E,D)->(T,D)
    encode_gemm_mb(cmd, P.gemm, b.gated, 0, b.w_out_proj, off_outp,
                   b.out_proj_out, 0, T, D, E);

    // 8. residual in-place: x_in += out_proj_out  (add allows aliasing)
    encode_add_f16(cmd, P.add, b.x_in, b.out_proj_out, b.x_out, T * D);
}

// ── Whole-model dispatch ──────────────────────────────────────────

struct ModelParams {
    uint32_t batch        = 1;
    uint32_t seq          = 1;
    uint32_t n_layers     = 24;
    uint32_t d_model      = 768;
    uint32_t intermediate = 1536;
    uint32_t n_heads      = 24;
    uint32_t head_dim     = 64;
    uint32_t state_size   = 128;
    uint32_t n_groups     = 1;
    uint32_t conv_kernel  = 4;
    uint32_t chunk_size   = 256;
    uint32_t vocab_size   = 50288;
    float    eps          = 1e-5f;
    float    dt_min       = 0.001f;
    float    dt_max       = 0.1f;
};

inline void dispatch_model(
    MTL::CommandBuffer* cmd,
    const ModelPSOs&    P,
    const ModelWeights& W,
    ModelBuffers&       B,
    LayerState*         states,
    const ModelParams&  M,
    MTL::Buffer*        output_id)
{
    const uint32_t T = M.batch * M.seq;

    // A. Embedding lookup
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.embedding_lookup);
        enc->setBuffer(W.w_embed,   0, 0);
        enc->setBuffer(B.tok_ids,   0, 1);
        enc->setBuffer(B.x,         0, 2);
        enc->setBytes(&T,            4, 3);
        enc->setBytes(&M.d_model,    4, 4);
        enc->setBytes(&M.vocab_size, 4, 5);
        const uint32_t D4 = M.d_model / 4;
        enc->dispatchThreadgroups(MTL::Size((D4 + 127) / 128, T, 1),
                                  MTL::Size(128, 1, 1));
        enc->endEncoding();
    }

    // B. Layer stack — in-place residual on B.x
    for (uint32_t L = 0; L < M.n_layers; ++L) {
        LayerParams lp;
        lp.batch        = M.batch;
        lp.seq          = M.seq;
        lp.d_model      = M.d_model;
        lp.intermediate = M.intermediate;
        lp.n_heads      = M.n_heads;
        lp.head_dim     = M.head_dim;
        lp.state_size   = M.state_size;
        lp.n_groups     = M.n_groups;
        lp.conv_kernel  = M.conv_kernel;
        lp.chunk_size   = M.chunk_size;
        lp.eps          = M.eps;
        lp.dt_min       = M.dt_min;
        lp.dt_max       = M.dt_max;
        lp.layer_idx    = L;
        lp.is_decode    = (M.seq == 1) ? 1u : 0u;

        DispatchBufs db{};
        db.x_in         = B.x;
        db.x_out        = B.x;
        db.x_norm       = B.x_norm;
        db.in_proj_out  = B.in_proj_out;
        db.z            = B.z;
        db.xBC          = B.xBC;
        db.dt_raw       = B.dt_raw;
        db.xBC_post     = B.xBC_post;
        db.ssd_out      = B.ssd_out;
        db.gated        = B.gated;
        db.out_proj_out = B.out_proj_out;
        db.conv_state   = states[L].conv_state;
        db.ssm_state    = states[L].ssm_state;
        db.w_pre_norm   = W.w_pre_norm;
        db.w_in_proj    = W.w_in_proj;
        db.w_conv       = W.w_conv;
        db.w_conv_b     = W.w_conv_b;
        db.w_dt_bias    = W.w_dt_bias;
        db.w_A_log      = W.w_A_log;
        db.w_D          = W.w_D;
        db.w_norm       = W.w_norm;
        db.w_out_proj   = W.w_out_proj;

        dispatch_layer(cmd, P.layer, lp, db);
    }

    // C. Final RMSNorm  B.x -> B.x_norm
    encode_rmsnorm_mb(cmd, P.layer.rmsnorm, B.x, W.w_final_norm, 0,
                      B.x_norm, T, M.d_model, M.eps);

    // D. LM head (tied): (T,D) x (V,D)^T → (T,V) logits
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.layer.gemm);
        const uint32_t M_v = T, K_v = M.d_model, N_v = M.vocab_size;
        uint32_t ldA = K_v, ldB = K_v, ldC = N_v;
        int transA = 0, transB = 1, has_bias = 0;
        enc->setBuffer(B.x_norm,   0, 0);
        enc->setBuffer(W.w_embed,  0, 1);
        enc->setBuffer(B.logits,   0, 2);
        enc->setBytes(&M_v,      4, 3); enc->setBytes(&N_v,      4, 4);
        enc->setBytes(&K_v,      4, 5); enc->setBytes(&ldA,      4, 6);
        enc->setBytes(&ldB,      4, 7); enc->setBytes(&ldC,      4, 8);
        enc->setBytes(&transA,   4, 9); enc->setBytes(&transB,   4, 10);
        enc->setBytes(&has_bias, 4, 11);
        enc->setBuffer(B.logits, 0, 12);
        enc->dispatchThreadgroups(MTL::Size((N_v + 63) / 64, (M_v + 31) / 32, 1),
                                  MTL::Size(64, 1, 1));
        enc->endEncoding();
    }

    // E. Argmax over LAST token row only
    {
        const size_t last_off = (size_t)(T - 1) * M.vocab_size * 2;
        const bool can_2pass = P.argmax_partial && P.argmax_reduce
                            && B.argmax_val_buf && B.argmax_idx_buf;
        if (can_2pass) {
            constexpr uint32_t ELTS_PER_TG = 16384u;
            const uint32_t n_blocks = (M.vocab_size + ELTS_PER_TG - 1u) / ELTS_PER_TG;
            {
                auto* enc = cmd->computeCommandEncoder();
                enc->setComputePipelineState(P.argmax_partial);
                enc->setBuffer(B.logits,         last_off, 0);
                enc->setBuffer(B.argmax_val_buf, 0,        1);
                enc->setBuffer(B.argmax_idx_buf, 0,        2);
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
                enc->setBuffer(output_id,        0, 2);
                enc->setBytes(&n_blocks,         4, 3);
                enc->dispatchThreadgroups(MTL::Size(1, 1, 1),
                                          MTL::Size(1024, 1, 1));
                enc->endEncoding();
            }
        } else {
            auto* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(P.argmax);
            enc->setBuffer(B.logits,  last_off, 0);
            enc->setBuffer(output_id, 0,        1);
            enc->setBytes(&M.vocab_size, 4, 2);
            enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1024, 1, 1));
            enc->endEncoding();
        }
    }
}

}  // namespace mamba2
}  // namespace meow

#endif
