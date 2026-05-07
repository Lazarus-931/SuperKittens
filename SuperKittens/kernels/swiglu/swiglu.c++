#include "swiglu.h"
#include "../runtime_bindings.h"

int sk_swiglu(void* x, void* y, uint32_t rows, uint32_t d) {
    auto* pso = sk::bindings_pso("fused_swiglu");
    if (!pso) return -1;

    size_t xb = (size_t)rows * d * sizeof(__fp16);
    auto* bX = sk::bindings_device()->newBuffer(xb, MTL::ResourceStorageModeShared);
    auto* bY = sk::bindings_device()->newBuffer(xb / 2, MTL::ResourceStorageModeShared);
    memcpy(bX->contents(), x, xb);

    auto* cmd = sk::bindings_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bX, 0, 0); enc->setBuffer(bY, 0, 1);
    enc->setBytes(&rows, sizeof(uint32_t), 2); enc->setBytes(&d, sizeof(uint32_t), 3);
    enc->dispatchThreadgroups(MTL::Size(1, (rows + 3) / 4, 1), MTL::Size(128, 1, 1));
    enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted();
    memcpy(y, bY->contents(), xb / 2);

    cmd->release(); bX->release(); bY->release();
    return 0;
}
