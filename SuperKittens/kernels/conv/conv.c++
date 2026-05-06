//
//  conv.c++ — C binding implementation for conv1d
//

#include "conv.h"
#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>

static MTL::Device*      g_dev  = nullptr;
static MTL::CommandQueue* g_q   = nullptr;
static MTL::Library*     g_lib  = nullptr;
static MTL::ComputePipelineState* g_pso = nullptr;

static void ensure_device() {
    if (g_dev) return;
    g_dev = MTL::CreateSystemDefaultDevice();
    g_q   = g_dev->newCommandQueue();
    const char* env = getenv("SK_METALLIB");
    const char* path = env ? env : "build/libsk.metallib";
    NS::Error* err = nullptr;
    auto* url = NS::URL::fileURLWithPath(NS::String::string(path, NS::UTF8StringEncoding));
    g_lib = g_dev->newLibrary(url, &err);
    if (!g_lib) fprintf(stderr, "conv: failed to load %s\n", path);
}

int sk_conv1d(void* x, void* weight, void* bias, void* y,
              uint32_t B, uint32_t L, uint32_t C, uint32_t K) {
    ensure_device();
    if (!g_lib) return -1;

    if (!g_pso) {
        auto* fn = g_lib->newFunction(NS::String::string("conv1d", NS::UTF8StringEncoding));
        if (!fn) return -2;
        NS::Error* err = nullptr;
        g_pso = g_dev->newComputePipelineState(fn, &err); fn->release();
        if (!g_pso) { if (err) err->release(); return -3; }
    }

    size_t xb = (size_t)B * L * C * sizeof(__fp16);
    size_t wb = (size_t)C * K * sizeof(__fp16);
    size_t bb = (size_t)C * sizeof(__fp16);

    auto* bX    = g_dev->newBuffer(xb, MTL::ResourceStorageModeShared);
    auto* bW    = g_dev->newBuffer(wb, MTL::ResourceStorageModeShared);
    auto* bBias = g_dev->newBuffer(bb, MTL::ResourceStorageModeShared);
    auto* bY    = g_dev->newBuffer(xb, MTL::ResourceStorageModeShared);
    memcpy(bX->contents(), x, xb);
    memcpy(bW->contents(), weight, wb);
    memcpy(bBias->contents(), bias, bb);

    uint32_t gy = (L + 3) / 4;

    auto* cmd = g_q->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(g_pso);
    enc->setBuffer(bX,    0, 0);
    enc->setBuffer(bW,    0, 1);
    enc->setBuffer(bBias, 0, 2);
    enc->setBuffer(bY,    0, 3);
    enc->setBytes(&B, sizeof(uint32_t), 4);
    enc->setBytes(&L, sizeof(uint32_t), 5);
    enc->setBytes(&C, sizeof(uint32_t), 6);
    enc->setBytes(&K, sizeof(uint32_t), 7);
    enc->dispatchThreadgroups(MTL::Size(B, gy, 1), MTL::Size(128, 1, 1));
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();

    memcpy(y, bY->contents(), xb);

    cmd->release();
    bX->release(); bW->release(); bBias->release(); bY->release();
    return 0;
}
