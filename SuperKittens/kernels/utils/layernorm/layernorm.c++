//
//  layernorm.c++ — C binding implementation
//

#include "layernorm.h"
#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>
#include <cstring>

static MTL::Device*      g_dev = nullptr;
static MTL::CommandQueue* g_q  = nullptr;
static MTL::Library*     g_lib = nullptr;

static void ensure() {
    if (g_dev) return;
    g_dev = MTL::CreateSystemDefaultDevice();
    g_q   = g_dev->newCommandQueue();
    const char* path = getenv("SK_METALLIB") ? getenv("SK_METALLIB") : "build/libsk.metallib";
    NS::Error* err = nullptr;
    auto* url = NS::URL::fileURLWithPath(NS::String::string(path, NS::UTF8StringEncoding));
    g_lib = g_dev->newLibrary(url, &err);
}

static int dispatch(const char* name,
                    void* x, void* w, void* b, void* y,
                    uint32_t rows, uint32_t d, float eps) {
    ensure();
    if (!g_lib) return -1;

    auto* fn = g_lib->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
    if (!fn) return -2;
    NS::Error* err = nullptr;
    auto* pso = g_dev->newComputePipelineState(fn, &err); fn->release();
    if (!pso) { if (err) err->release(); return -3; }

    size_t xb = (size_t)rows * d * sizeof(__fp16);
    size_t db = (size_t)d * sizeof(__fp16);

    auto* bX = g_dev->newBuffer(xb, MTL::ResourceStorageModeShared);
    auto* bW = g_dev->newBuffer(db, MTL::ResourceStorageModeShared);
    auto* bY = g_dev->newBuffer(xb, MTL::ResourceStorageModeShared);
    memcpy(bX->contents(), x, xb);
    memcpy(bW->contents(), w, db);

    auto* bB = b ? g_dev->newBuffer(db, MTL::ResourceStorageModeShared) : nullptr;
    if (bB) memcpy(bB->contents(), b, db);

    auto* bRows = g_dev->newBuffer(&rows, sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* bD    = g_dev->newBuffer(&d,    sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* bEps  = g_dev->newBuffer(&eps,  sizeof(float),    MTL::ResourceStorageModeShared);

    uint32_t gy = (rows + 3) / 4;

    auto* cmd = g_q->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bX,    0, 0);
    enc->setBuffer(bW,    0, 1);
    enc->setBuffer(bB ? bB : bW, 0, 2);  // beta or dummy (unused by rmsnorm but buffer slot exists)
    enc->setBuffer(bY,    0, 3);
    enc->setBuffer(bRows, 0, 4);
    enc->setBuffer(bD,    0, 5);
    enc->setBuffer(bEps,  0, 6);
    enc->dispatchThreadgroups(MTL::Size(1, gy, 1), MTL::Size(128, 1, 1));
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();

    memcpy(y, bY->contents(), xb);

    cmd->release(); pso->release();
    bX->release(); bW->release(); if (bB) bB->release(); bY->release();
    bRows->release(); bD->release(); bEps->release();
    return 0;
}

int sk_layernorm(void* x, void* gamma, void* beta, void* y,
                 uint32_t rows, uint32_t d, float eps) {
    return dispatch("layernorm", x, gamma, beta, y, rows, d, eps);
}

int sk_rmsnorm(void* x, void* weight, void* y,
               uint32_t rows, uint32_t d, float eps) {
    return dispatch("rmsnorm", x, weight, nullptr, y, rows, d, eps);
}
