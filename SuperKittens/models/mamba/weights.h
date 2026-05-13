#pragma once
// HF safetensor name map for state-spaces/mamba-*-hf checkpoints.
//
// Top-level (no `model.` prefix; state-spaces uses `backbone.`):
//   backbone.embeddings.weight                          [V, H]
//   backbone.norm_f.weight                              [H]
//   backbone.layers.{L}.norm.weight                     [H]
//   backbone.layers.{L}.mixer.in_proj.weight            [2*E, H]
//   backbone.layers.{L}.mixer.conv1d.weight             [E, 1, K]   (K=conv_kernel=4)
//   backbone.layers.{L}.mixer.conv1d.bias               [E]
//   backbone.layers.{L}.mixer.x_proj.weight             [dt_rank + 2*N, E]
//   backbone.layers.{L}.mixer.dt_proj.weight            [E, dt_rank]
//   backbone.layers.{L}.mixer.dt_proj.bias              [E]
//   backbone.layers.{L}.mixer.A_log                     [E, N]   (A = -exp(A_log))
//   backbone.layers.{L}.mixer.D                         [E]
//   backbone.layers.{L}.mixer.out_proj.weight           [H, E]
//   lm_head.weight                                      tied with backbone.embeddings.weight
//
// Dims for mamba-2.8b-hf:
//   hidden_size  H = 2560
//   intermediate E = 5120          (= 2 * H, mamba expansion factor 2)
//   state_size   N = 16
//   conv_kernel  K = 4
//   dt_rank          = 160         (= ceil(H/16))
//   n_layers         = 64
//   vocab            = 50280 (gpt-neox tokenizer)
//
// Storage convention used by SK loader (matches HF):
//   Linear weights stored row-major [out, in]; multiplied as y = x @ W.T.
//   conv1d.weight is depthwise [E, 1, K]; loader flattens to [E, K].

#include <cstdint>

namespace sk::mamba {

struct Dims {
    uint32_t H;          // hidden_size
    uint32_t E;          // intermediate_size
    uint32_t N;          // ssm_state_size
    uint32_t K;          // conv_kernel
    uint32_t dt_rank;
    uint32_t n_layers;
    uint32_t vocab;
    float layer_norm_eps;
};

struct LayerWeights {
    void* norm_w;        // [H]   bf16
    void* in_proj_w;     // [2E, H] bf16
    void* conv1d_w;      // [E, K] bf16
    void* conv1d_b;      // [E]    bf16
    void* x_proj_w;      // [dt_rank + 2N, E] bf16
    void* dt_proj_w;     // [E, dt_rank] bf16
    void* dt_proj_b;     // [E]    fp32 (kept fp32 like HF)
    void* A_log;         // [E, N] fp32 (kept fp32 like HF)
    void* D;             // [E]    fp32
    void* out_proj_w;    // [H, E] bf16
};

struct ModelWeights {
    Dims dims;
    void* embed_w;       // [V, H] bf16   (also serves as lm_head when tied)
    void* norm_f_w;      // [H]    bf16
    LayerWeights* layers;
};

} // namespace sk::mamba
