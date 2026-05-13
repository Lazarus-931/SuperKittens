// Host wrapper for moe_router_v3. Stages a replacement for
// SuperKittens/kernels/moe/router.h.

#ifndef SK_MOE_ROUTER_V3_H
#define SK_MOE_ROUTER_V3_H

#include <Metal/Metal.hpp>
#include <cstdint>

namespace meow {

struct RouterV3Args {
    uint32_t T;
    uint32_t D;
    uint32_t N;
    uint32_t K;
    uint32_t n_group;
    uint32_t topk_group;
    float    routed_scaling;
    uint32_t norm_topk_prob;
    uint32_t has_bias;
};

inline void encode_moe_router_v3(
    MTL::CommandBuffer*        cmd,
    MTL::ComputePipelineState* pso,
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* bias,
    MTL::Buffer* top_idx, MTL::Buffer* top_score,
    const RouterV3Args& args)
{
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(x,         0, 0);
    enc->setBuffer(W,         0, 1);
    enc->setBuffer(bias,      0, 2);
    enc->setBuffer(top_idx,   0, 3);
    enc->setBuffer(top_score, 0, 4);
    enc->setBytes(&args, sizeof(args), 5);
    enc->dispatchThreadgroups(MTL::Size(args.T, 1, 1),
                              MTL::Size(256, 1, 1));
    enc->endEncoding();
}

} // namespace meow
#endif
