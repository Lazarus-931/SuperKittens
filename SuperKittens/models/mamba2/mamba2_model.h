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
};

}  // namespace mamba2
}  // namespace meow

#endif
