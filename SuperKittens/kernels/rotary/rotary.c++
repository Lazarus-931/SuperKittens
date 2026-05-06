//
//  rotary.c++ — C binding implementation for RoPE
//

#include "rotary.h"
#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>
#include <cstring>

static MTL::Device*      g_dev = nullptr;
static MTL::CommandQueue* g_q  = nullptr;
static MTL::Library*     g_lib = nullptr;
static MTL::ComputePipelineState* g_pso = nullptr;

static void ensure() {
    if (g_dev) return;
    g_dev = MTL::CreateSystemDefaultDevice();
    g_q   = g_dev->newCommandQueue();
    const char* env = getenv("SK_METALLIB");
    const char* path = env ? env : "build/libsk.metallib";
    NS::Error* err = nullptr;
    auto* url = NS::URL::fileURLWithPath(NS::String::string(path, NS::UTF8StringEncoding));
    g_lib = g_dev->newLibrary(url, &err);
}

int sk_rope(void* Q, void* K, void* cos, void* sin,
            uint32_t seq, uint32_t head_dim, uint32_t n_heads) {
    ensure();
    if (!g_lib) return -1;

    if (!g_pso) {
        auto* fn = g_lib->newFunction(NS::String::string("rope_qk", NS::UTF8StringEncoding));
        if (!fn) return -2;
        NS::Error* err = nullptr;
        g_pso = g_dev->newComputePipelineState(fn, &err); fn->release();
        if (!g_pso) { if (err) err->release(); return -3; }
    }

    size_t bytes = (size_t)n_heads * seq * head_dim * sizeof(__fp16);
    size_t cs_bytes = (size_t)seq * (head_dim / 2) * sizeof(__fp16);

    auto* bQ   = g_dev->newBuffer(bytes,    MTL::ResourceStorageModeShared);
    auto* bK   = g_dev->newBuffer(bytes,    MTL::ResourceStorageModeShared);
    auto* bCos = g_dev->newBuffer(cs_bytes, MTL::ResourceStorageModeShared);
    auto* bSin = g_dev->newBuffer(cs_bytes, MTL::ResourceStorageModeShared);
    memcpy(bQ->contents(), Q, bytes);
    memcpy(bK->contents(), K, bytes);
    memcpy(bCos->contents(), cos, cs_bytes);
    memcpy(bSin->contents(), sin, cs_bytes);

    auto* cmd = g_q->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(g_pso);
    enc->setBuffer(bQ,   0, 0);
    enc->setBuffer(bK,   0, 1);
    enc->setBuffer(bCos, 0, 2);
    enc->setBuffer(bSin, 0, 3);
    enc->setBytes(&seq,      sizeof(uint32_t), 4);
    enc->setBytes(&head_dim, sizeof(uint32_t), 5);
    enc->setBytes(&n_heads,  sizeof(uint32_t), 6);
    enc->dispatchThreadgroups(MTL::Size(n_heads, 1, 1), MTL::Size(1024, 1, 1));
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();

    memcpy(Q, bQ->contents(), bytes);
    memcpy(K, bK->contents(), bytes);

    cmd->release();
    bQ->release(); bK->release(); bCos->release(); bSin->release();
    return 0;
}
