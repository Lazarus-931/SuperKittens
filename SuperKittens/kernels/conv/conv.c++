#include "conv.h"
#include "../runtime_bindings.h"

int sk_conv1d(void* x, void* weight, void* bias, void* y,
              uint32_t B, uint32_t L, uint32_t C, uint32_t K) {
    auto* pso = sk::bindings_pso("conv1d");
    if (!pso) return -1;

    size_t xb = (size_t)B * L * C * sizeof(__fp16);
    size_t wb = (size_t)C * K * sizeof(__fp16);
    size_t bb = (size_t)C * sizeof(__fp16);

    auto* bX = sk::bindings_device()->newBuffer(xb, MTL::ResourceStorageModeShared);
    auto* bW = sk::bindings_device()->newBuffer(wb, MTL::ResourceStorageModeShared);
    auto* bBias = sk::bindings_device()->newBuffer(bb, MTL::ResourceStorageModeShared);
    auto* bY = sk::bindings_device()->newBuffer(xb, MTL::ResourceStorageModeShared);
    memcpy(bX->contents(), x, xb);
    memcpy(bW->contents(), weight, wb);
    memcpy(bBias->contents(), bias, bb);

    uint32_t gy = (L + 3) / 4;

    auto* cmd = sk::bindings_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bX,    0, 0); enc->setBuffer(bW,    0, 1);
    enc->setBuffer(bBias, 0, 2); enc->setBuffer(bY,    0, 3);
    enc->setBytes(&B, sizeof(uint32_t), 4); enc->setBytes(&L, sizeof(uint32_t), 5);
    enc->setBytes(&C, sizeof(uint32_t), 6); enc->setBytes(&K, sizeof(uint32_t), 7);
    enc->dispatchThreadgroups(MTL::Size(B, gy, 1), MTL::Size(128, 1, 1));
    enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted();
    memcpy(y, bY->contents(), xb);

    cmd->release(); bX->release(); bW->release(); bBias->release(); bY->release();
    return 0;
}
