// Family-agnostic decode shell: embedding lookup + per-layer ping-pong loop +
// final rmsnorm. Per-layer dispatch (the fused, family-specific bulk) stays
// behind a callback so iter-11 fusions are preserved intact. LM-head + argmax
// remain in the per-family launcher: those have 3+ fused fast-paths (Q8_0
// LM-head, 2-pass argmax, ICB-tail argmax) whose branching can't be hidden
// behind a single PSO without losing perf.

#ifndef SUPERKITTENS_INFERENCE_DECODE_ORCHESTRATOR_H
#define SUPERKITTENS_INFERENCE_DECODE_ORCHESTRATOR_H

#include <Metal/Metal.hpp>
#include <cstdint>
#include <functional>

namespace sk {

struct DecodeOrchestratorCtx {
    MTL::Buffer* token_in    = nullptr;  // int32 input token id(s), (T,)
    MTL::Buffer* x_a         = nullptr;  // residual stream ping (fp16)
    MTL::Buffer* x_b         = nullptr;  // residual stream pong (fp16); also receives final-rmsnorm when n_layers is even
    uint32_t     T           = 1;        // batch * seq
    uint32_t     n_layers    = 0;
    uint32_t     d_model     = 0;
    uint32_t     vocab_size  = 0;
    float        eps         = 1e-6f;
};

struct DecodeOrchestratorPSOs {
    MTL::ComputePipelineState* embedding_lookup = nullptr;
    MTL::ComputePipelineState* final_rmsnorm    = nullptr;  // generic (rows>1 ok)
    MTL::ComputePipelineState* final_rmsnorm_t1 = nullptr;  // optional T==1 fast path
};

// Per-layer callback. cur/nxt are the ping-pong residual buffers; callback
// must read from `cur` and write its layer output to `nxt`. Orchestrator
// swaps them between iterations.
using DispatchLayerFn = std::function<void(
    MTL::CommandBuffer* /*cmd*/,
    uint32_t            /*layer_idx*/,
    MTL::Buffer*        /*cur*/,
    MTL::Buffer*        /*nxt*/)>;

// Encodes embedding → loop(dispatch_layer) → final rmsnorm. Caller owns
// LM-head + argmax (see top-of-file rationale). Returns the post-rmsnorm
// buffer (one of ctx.x_a / ctx.x_b) for the caller's LM-head input.
MTL::Buffer* encode_decode_step(
    MTL::CommandBuffer*           cmd,
    const DecodeOrchestratorCtx&  ctx,
    const DecodeOrchestratorPSOs& psos,
    DispatchLayerFn               layer_fn,
    MTL::Buffer*                  w_embed,
    MTL::Buffer*                  w_final_norm);

}  // namespace sk

#endif
