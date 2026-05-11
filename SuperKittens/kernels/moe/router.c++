//
//  router.c++ — host dispatcher for moe_router
//
#include "router.h"
#include "../runtime_bindings.h"

extern "C" int sk_moe_router(void* x, void* W,
                             void* top_idx, void* top_score,
                             uint32_t T, uint32_t D, uint32_t N, uint32_t K) {
    auto* pso = sk::bindings_pso("moe_router");
    if (!pso) return -1;
    auto* dev = sk::bindings_device();

    size_t xb = (size_t)T * D * sizeof(__fp16);
    size_t wb = (size_t)D * N * sizeof(__fp16);
    size_t ib = (size_t)T * K * sizeof(int32_t);
    size_t sb = (size_t)T * K * sizeof(__fp16);

    auto* bX = dev->newBuffer(xb, MTL::ResourceStorageModeShared);
    auto* bW = dev->newBuffer(wb, MTL::ResourceStorageModeShared);
    auto* bI = dev->newBuffer(ib, MTL::ResourceStorageModeShared);
    auto* bS = dev->newBuffer(sb, MTL::ResourceStorageModeShared);
    memcpy(bX->contents(), x, xb);
    memcpy(bW->contents(), W, wb);

    auto* cmd = sk::bindings_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bX, 0, 0);
    enc->setBuffer(bW, 0, 1);
    enc->setBuffer(bI, 0, 2);
    enc->setBuffer(bS, 0, 3);
    enc->setBytes(&T, sizeof(uint32_t), 4);
    enc->setBytes(&D, sizeof(uint32_t), 5);
    enc->setBytes(&N, sizeof(uint32_t), 6);
    enc->setBytes(&K, sizeof(uint32_t), 7);
    enc->dispatchThreadgroups(MTL::Size(T, 1, 1), MTL::Size(256, 1, 1));
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();

    memcpy(top_idx,   bI->contents(), ib);
    memcpy(top_score, bS->contents(), sb);

    bX->release(); bW->release(); bI->release(); bS->release();
    cmd->release();
    return 0;
}
