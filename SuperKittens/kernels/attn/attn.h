//
//  attn.h
//  SuperKittens — unified attention dispatch
//
//  Auto-selects: d=64 → FA (attn_d64.metal), d=128 → MHA (attn_d128.metal)
//

#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

namespace meow::attn {

struct AttnConfig {
    int batch;
    int n_heads;
    int seq;
    int head_dim;
    bool causal;
};

inline const char* kernel_name(const AttnConfig& cfg) {
    if (cfg.head_dim == 64) {
        return cfg.causal ? "fa_causal_64" : "fa_noncausal_64";
    }
    // d=128 fast path + generic fallback
    return cfg.causal ? "mha_causal" : "mha_noncausal";
}

inline size_t buffer_bytes(const AttnConfig& cfg) {
    return (size_t)cfg.batch * cfg.n_heads * cfg.seq * cfg.head_dim * sizeof(__fp16);
}

inline double dispatch(
    MTL::Device* device,
    MTL::CommandQueue* queue,
    MTL::Library* lib,
    const AttnConfig& cfg,
    MTL::Buffer* bufQ,
    MTL::Buffer* bufK,
    MTL::Buffer* bufV,
    MTL::Buffer* bufO,
    MTL::Buffer* bufSeq,
    MTL::Buffer* bufDim,
    MTL::Buffer* bufHeads)
{
    const char* name = kernel_name(cfg);
    if (!name) return -1.0;

    auto* fn = lib->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
    if (!fn) return -1.0;
    NS::Error* err = nullptr;
    auto* pso = device->newComputePipelineState(fn, &err);
    fn->release();
    if (!pso) return -1.0;

    bool is_fa = (cfg.head_dim == 64);
    uint32_t grid_y = is_fa ? (cfg.seq + 31) / 32 : (cfg.seq + 3) / 4;
    uint32_t threads = is_fa ? 1024 : 128;

    auto* cmd = queue->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bufQ,     0, 0);
    enc->setBuffer(bufK,     0, 1);
    enc->setBuffer(bufV,     0, 2);
    enc->setBuffer(bufO,     0, 3);
    enc->setBuffer(bufSeq,   0, 4);
    if (is_fa) {
        enc->setBuffer(bufHeads, 0, 5);  // FA: buffer 5 = n_heads
    } else {
        enc->setBuffer(bufDim,   0, 5);  // MHA: buffer 5 = head_dim
        enc->setBuffer(bufHeads, 0, 6);  // MHA: buffer 6 = num_heads
    }
    enc->dispatchThreadgroups(MTL::Size(cfg.n_heads, grid_y, cfg.batch),
                              MTL::Size(threads, 1, 1));
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();

    double ms = (cmd->GPUEndTime() - cmd->GPUStartTime()) * 1000.0;
    pso->release();
    return ms;
}

} // namespace meow::attn
