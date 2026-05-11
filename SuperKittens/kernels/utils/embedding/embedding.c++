//
//  embedding.c++ — host dispatch for sk_embedding_lookup.
//

#include "embedding.h"
#include "../../runtime_bindings.h"

#include <cstring>

int sk_embedding_lookup(void* table, void* ids, void* out,
                        uint32_t N, uint32_t D, uint32_t V) {
    if (D % 4 != 0) return -2;

    auto* pso = sk::bindings_pso("embedding_lookup");
    if (!pso) return -1;

    size_t tb = (size_t)V * D * sizeof(__fp16);
    size_t ib = (size_t)N * sizeof(int32_t);
    size_t ob = (size_t)N * D * sizeof(__fp16);

    auto* dev = sk::bindings_device();
    auto* bT  = dev->newBuffer(tb, MTL::ResourceStorageModeShared);
    auto* bI  = dev->newBuffer(ib, MTL::ResourceStorageModeShared);
    auto* bO  = dev->newBuffer(ob, MTL::ResourceStorageModeShared);
    if (!bT || !bI || !bO) return -3;

    std::memcpy(bT->contents(), table, tb);
    std::memcpy(bI->contents(), ids,   ib);

    auto* cmd = sk::bindings_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bT, 0, 0);
    enc->setBuffer(bI, 0, 1);
    enc->setBuffer(bO, 0, 2);
    enc->setBytes(&N, sizeof(uint32_t), 3);
    enc->setBytes(&D, sizeof(uint32_t), 4);
    enc->setBytes(&V, sizeof(uint32_t), 5);

    const uint32_t D4 = D / 4;
    const uint32_t gx = (D4 + 127) / 128;
    enc->dispatchThreadgroups(MTL::Size(gx, N, 1), MTL::Size(128, 1, 1));
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();

    std::memcpy(out, bO->contents(), ob);

    cmd->release();
    bT->release();
    bI->release();
    bO->release();
    return 0;
}
