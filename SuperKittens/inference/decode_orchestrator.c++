#include "decode_orchestrator.h"

namespace sk {

MTL::Buffer* encode_decode_step(
    MTL::CommandBuffer*           cmd,
    const DecodeOrchestratorCtx&  ctx,
    const DecodeOrchestratorPSOs& psos,
    DispatchLayerFn               layer_fn,
    MTL::Buffer*                  w_embed,
    MTL::Buffer*                  w_final_norm)
{
    // A. Embedding lookup → x_a
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(psos.embedding_lookup);
        enc->setBuffer(w_embed,       0, 0);
        enc->setBuffer(ctx.token_in,  0, 1);
        enc->setBuffer(ctx.x_a,       0, 2);
        enc->setBytes(&ctx.T,          4, 3);
        enc->setBytes(&ctx.d_model,    4, 4);
        enc->setBytes(&ctx.vocab_size, 4, 5);
        const uint32_t D4 = ctx.d_model / 4;
        enc->dispatchThreadgroups(MTL::Size((D4 + 127) / 128, ctx.T, 1),
                                  MTL::Size(128, 1, 1));
        enc->endEncoding();
    }

    // B. Per-layer ping-pong.
    MTL::Buffer* cur = ctx.x_a;
    MTL::Buffer* nxt = ctx.x_b;
    for (uint32_t L = 0; L < ctx.n_layers; ++L) {
        layer_fn(cmd, L, cur, nxt);
        MTL::Buffer* tmp = cur; cur = nxt; nxt = tmp;
    }
    // After loop, `cur` holds the last layer's output.

    // C. Final RMSNorm — reads `cur` (last layer out), writes `nxt`.
    {
        auto* enc = cmd->computeCommandEncoder();
        const bool use_t1 = (psos.final_rmsnorm_t1 != nullptr) && (ctx.T == 1u);
        enc->setComputePipelineState(use_t1 ? psos.final_rmsnorm_t1
                                            : psos.final_rmsnorm);
        enc->setBuffer(cur,           0, 0);
        enc->setBuffer(w_final_norm,  0, 1);
        enc->setBuffer(nxt,           0, 2);
        uint32_t rows = ctx.T;
        enc->setBytes(&rows,       4, 3);
        enc->setBytes(&ctx.d_model, 4, 4);
        enc->setBytes(&ctx.eps,    4, 5);
        if (use_t1) {
            enc->dispatchThreadgroups(MTL::Size(1, rows, 1),
                                      MTL::Size(256, 1, 1));
        } else {
            enc->dispatchThreadgroups(MTL::Size(1, (rows + 3) / 4, 1),
                                      MTL::Size(128, 1, 1));
        }
        enc->endEncoding();
    }
    return nxt;
}

}  // namespace sk
