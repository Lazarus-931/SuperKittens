//  add_rmsnorm.c++ — host launcher for fused residual-add + RMSNorm.

#include "add_rmsnorm.h"
#include "../runtime_bindings.h"
#include <cstring>

extern "C" int sk_add_rmsnorm(void* x, void* delta, void* gamma,
                              void* y, void* y_norm,
                              uint32_t T, uint32_t D, float eps) {
    auto* pso = sk::bindings_pso("add_rmsnorm");
    if (!pso) return -1;
    auto* dev = sk::bindings_device();

    const size_t xb  = (size_t)T * D * 2;
    const size_t gb  = (size_t)D * 2;

    auto* bX = dev->newBuffer(xb, MTL::ResourceStorageModeShared);
    auto* bD = dev->newBuffer(xb, MTL::ResourceStorageModeShared);
    auto* bG = dev->newBuffer(gb, MTL::ResourceStorageModeShared);
    auto* bY = dev->newBuffer(xb, MTL::ResourceStorageModeShared);
    auto* bN = dev->newBuffer(xb, MTL::ResourceStorageModeShared);
    std::memcpy(bX->contents(), x,     xb);
    std::memcpy(bD->contents(), delta, xb);
    std::memcpy(bG->contents(), gamma, gb);

    auto* cmd = sk::bindings_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bX, 0, 0); enc->setBuffer(bD, 0, 1);
    enc->setBuffer(bG, 0, 2);
    enc->setBuffer(bY, 0, 3); enc->setBuffer(bN, 0, 4);
    enc->setBytes(&T,   4, 5);
    enc->setBytes(&D,   4, 6);
    enc->setBytes(&eps, 4, 7);

    enc->dispatchThreadgroups(MTL::Size(1, T, 1), MTL::Size(128, 1, 1));
    enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted();
    std::memcpy(y,      bY->contents(), xb);
    std::memcpy(y_norm, bN->contents(), xb);

    cmd->release();
    bX->release(); bD->release(); bG->release(); bY->release(); bN->release();
    return 0;
}
