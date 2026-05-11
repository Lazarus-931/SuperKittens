//
//  moe_ffn.h — fused MoE FFN block for DeepSeek-style layers.
//
//  Input:  x [T, D]        (post-attn residual)
//  Output: out [T, D]      (x + sum_slot route_w · expert_FFN(x))
//
//  Pipeline (3 dispatches):
//     1. moe_router        x → top_idx [T, K], top_score [T, K]
//     2. moe_swiglu_pair   x, W_gate, W_up, top_idx → hidden [T, K, N_int]
//     3. moe_down_scatter  hidden, W_down, top_idx, top_score, x → out [T, D]
//
//  The router-softmax weights flow through as the routing scale baked into
//  step 3. Caller pre-allocates all buffers and resolves PSOs.
//
//  Header-only. Mirrors the gemma4_model.h dispatch pattern.
//

#ifndef SUPERKITTENS_DEEPSEEK_MOE_FFN_H
#define SUPERKITTENS_DEEPSEEK_MOE_FFN_H

#include <Metal/Metal.hpp>
#include <cstdint>

namespace meow {
namespace deepseek {

struct MoeFfnPSOs {
    MTL::ComputePipelineState* router;
    MTL::ComputePipelineState* swiglu_pair;          // fp16 weights
    MTL::ComputePipelineState* down_scatter;         // fp16 weights
    // Int2 variants for DS4-shaped models (IQ2_XXS up/gate, Q2_K down).
    // Caller picks via MoeFfnParams::quant. Both may be null if only the
    // fp16 path is wanted.
    MTL::ComputePipelineState* swiglu_pair_iq2xxs;
    MTL::ComputePipelineState* down_scatter_q2k;
};

struct MoeFfnBuffers {
    // Inputs
    MTL::Buffer* x;             // (T, D)         fp16   — post-attn residual
    MTL::Buffer* w_router;      // (D, n_expert)  fp16
    MTL::Buffer* w_gate;        // (E, n_int, D)  fp16
    MTL::Buffer* w_up;          // (E, n_int, D)  fp16
    MTL::Buffer* w_down;        // (E, D, n_int)  fp16

    // Per-call scratch (caller-allocated)
    MTL::Buffer* top_idx;       // (T, top_k)     i32
    MTL::Buffer* top_score;     // (T, top_k)     fp16
    MTL::Buffer* hidden;        // (T, top_k, n_int) fp16

    // Residual buffer for down_scatter's fused add. If null, x is used as
    // residual (legacy behavior — appropriate when MoE is the only FFN path).
    // DS4: set this to (y_attn + shared_out) so the layer's residual stream
    // accumulates both attn output, shared expert, and routed experts.
    MTL::Buffer* residual = nullptr;

    // Output
    MTL::Buffer* out;           // (T, D)         fp16
};

enum class MoeQuant : uint32_t {
    FP16     = 0,
    INT2_DS4 = 1,   // IQ2_XXS up/gate + Q2_K down — DS4 V4 Flash production
};

struct MoeFfnParams {
    uint32_t T;
    uint32_t D;
    uint32_t n_int;
    uint32_t n_expert;
    uint32_t top_k;
    MoeQuant quant = MoeQuant::FP16;
};

inline void dispatch_moe_ffn(
    MTL::CommandBuffer*  cmd,
    const MoeFfnPSOs&    P,
    const MoeFfnBuffers& B,
    const MoeFfnParams&  p)
{
    // 1. Router: x @ W_router → softmax → top-k.
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(P.router);
        enc->setBuffer(B.x,         0, 0);
        enc->setBuffer(B.w_router,  0, 1);
        enc->setBuffer(B.top_idx,   0, 2);
        enc->setBuffer(B.top_score, 0, 3);
        enc->setBytes(&p.T,        4, 4);
        enc->setBytes(&p.D,        4, 5);
        enc->setBytes(&p.n_expert, 4, 6);
        enc->setBytes(&p.top_k,    4, 7);
        enc->dispatchThreadgroups(MTL::Size(p.T, 1, 1), MTL::Size(256, 1, 1));
        enc->endEncoding();
    }

    // 2. Fused gate + up + SiLU + mul (per-expert matvec).
    //    INT2_DS4 routes to moe_swiglu_pair_iq2xxs (W_gate/W_up are IQ2_XXS bytes).
    //    Same C-side signature; the int2 kernel reinterprets the W buffers
    //    as block_iq2_xxs[] internally.
    {
        auto* enc = cmd->computeCommandEncoder();
        auto* pso = (p.quant == MoeQuant::INT2_DS4) ? P.swiglu_pair_iq2xxs
                                                    : P.swiglu_pair;
        enc->setComputePipelineState(pso);
        enc->setBuffer(B.x,        0, 0);
        enc->setBuffer(B.w_gate,   0, 1);
        enc->setBuffer(B.w_up,     0, 2);
        enc->setBuffer(B.top_idx,  0, 3);
        enc->setBuffer(B.hidden,   0, 4);
        enc->setBytes(&p.T,        4, 5);
        enc->setBytes(&p.top_k,    4, 6);
        enc->setBytes(&p.D,        4, 7);
        enc->setBytes(&p.n_int,    4, 8);
        const uint32_t COLS_PER_TG = 16;
        enc->dispatchThreadgroups(
            MTL::Size((p.n_int + COLS_PER_TG - 1) / COLS_PER_TG, p.T * p.top_k, 1),
            MTL::Size(256, 1, 1));
        enc->endEncoding();
    }

    // 3. Fused down + routing-weight scale + residual add.
    //    INT2_DS4 routes to moe_down_scatter_q2k (W_down is Q2_K bytes).
    {
        auto* enc = cmd->computeCommandEncoder();
        auto* pso = (p.quant == MoeQuant::INT2_DS4) ? P.down_scatter_q2k
                                                    : P.down_scatter;
        enc->setComputePipelineState(pso);
        enc->setBuffer(B.hidden,    0, 0);
        enc->setBuffer(B.w_down,    0, 1);
        enc->setBuffer(B.top_idx,   0, 2);
        enc->setBuffer(B.top_score, 0, 3);
        enc->setBuffer(B.residual ? B.residual : B.x, 0, 4);   // residual buffer
        enc->setBuffer(B.out,       0, 5);
        enc->setBytes(&p.T,        4, 6);
        enc->setBytes(&p.top_k,    4, 7);
        enc->setBytes(&p.D,        4, 8);
        enc->setBytes(&p.n_int,    4, 9);
        const uint32_t COLS_PER_TG = 16;
        enc->dispatchThreadgroups(
            MTL::Size((p.D + COLS_PER_TG - 1) / COLS_PER_TG, p.T, 1),
            MTL::Size(256, 1, 1));
        enc->endEncoding();
    }
}

} // namespace deepseek
} // namespace meow

#endif
