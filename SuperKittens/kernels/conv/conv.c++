//
//  conv.c++ — C binding implementation for conv1d
//

#include "conv.h"
#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>

static MTL::Device*      g_dev  = nullptr;
static MTL::CommandQueue* g_q   = nullptr;
static MTL::Library*     g_lib  = nullptr;

static void ensure_device() {
    if (g_dev) return;
    g_dev = MTL::CreateSystemDefaultDevice();
    g_q   = g_dev->newCommandQueue();
    const char* path = getenv("SK_METALLIB") ? getenv("SK_METALLIB") : "build/libsk.metallib";
    NS::Error* err = nullptr;
    auto* url = NS::URL::fileURLWithPath(NS::String::string(path, NS::UTF8StringEncoding));
    g_lib = g_dev->newLibrary(url, &err);
    if (!g_lib) fprintf(stderr, "conv: failed to load %s\n", path);
}

int sk_conv1d(void* x, void* weight, void* bias, void* y,
              uint32_t B, uint32_t L, uint32_t C, uint32_t K) {
    ensure_device();
    if (!g_lib) return -1;

    auto* fn = g_lib->newFunction(NS::String::string("conv1d", NS::UTF8StringEncoding));
    if (!fn) return -2;
    NS::Error* err = nullptr;
    auto* pso = g_dev->newComputePipelineState(fn, &err); fn->release();
    if (!pso) { if (err) err->release(); return -3; }

    size_t xb = (size_t)B * L * C * sizeof(__fp16);
    size_t wb = (size_t)C * K * sizeof(__fp16);
    size_t bb = (size_t)C * sizeof(__fp16);

    auto* bX = g_dev->newBuffer(xb, MTL::ResourceStorageModeShared);
    auto* bW = g_dev->newBuffer(wb, MTL::ResourceStorageModeShared);
    auto* bB = g_dev->newBuffer(bb, MTL::ResourceStorageModeShared);
    auto* bY = g_dev->newBuffer(xb, MTL::ResourceStorageModeShared);
    memcpy(bX->contents(), x, xb);
    memcpy(bW->contents(), weight, wb);
    memcpy(bB->contents(), bias, bb);

    auto* bB1 = g_dev->newBuffer(&B, sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* bL  = g_dev->newBuffer(&L, sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* bC  = g_dev->newBuffer(&C, sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* bK  = g_dev->newBuffer(&K, sizeof(uint32_t), MTL::ResourceStorageModeShared);

    uint32_t gy = (L + 3) / 4;

    auto* cmd = g_q->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bX,  0, 0);
    enc->setBuffer(bW,  0, 1);
    enc->setBuffer(bB,  0, 2);
    enc->setBuffer(bY,  0, 3);
    enc->setBuffer(bB1, 0, 4);
    enc->setBuffer(bL,  0, 5);
    enc->setBuffer(bC,  0, 6);
    enc->setBuffer(bK,  0, 7);
    enc->dispatchThreadgroups(MTL::Size(B, gy, 1), MTL::Size(128, 1, 1));
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();

    memcpy(y, bY->contents(), xb);

    cmd->release(); pso->release();
    bX->release(); bW->release(); bB->release(); bY->release();
    bB1->release(); bL->release(); bC->release(); bK->release();
    return 0;
}
