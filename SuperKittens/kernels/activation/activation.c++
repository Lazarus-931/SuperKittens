//
//  activation.c++ — gelu/silu/relu binding (uses runtime_bindings.h)
//
#include "activation.h"
#include "../runtime_bindings.h"

int sk_activation(const char* kernel_name, void* x, void* y, uint32_t rows, uint32_t cols) {
    auto* pso = sk::bindings_pso(kernel_name);
    if (!pso) return -1;

    size_t bytes = (size_t)rows * cols * sizeof(__fp16);
    auto* bx = sk::bindings_device()->newBuffer(bytes, MTL::ResourceStorageModeShared);
    auto* by = sk::bindings_device()->newBuffer(bytes, MTL::ResourceStorageModeShared);
    memcpy(bx->contents(), x, bytes);

    auto* cmd = sk::bindings_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bx, 0, 0);
    enc->setBuffer(by, 0, 1);
    enc->setBytes(&rows, sizeof(uint32_t), 2);
    enc->setBytes(&cols, sizeof(uint32_t), 3);
    enc->dispatchThreadgroups(MTL::Size(1, (rows + 3) / 4, 1), MTL::Size(128, 1, 1));
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    memcpy(y, by->contents(), bytes);

    cmd->release(); bx->release(); by->release();
    return 0;
}
