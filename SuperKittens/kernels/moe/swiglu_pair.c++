//  swiglu_pair.c++ — host launcher for fused MoE swiglu pair.

#include "swiglu_pair.h"
#include "../runtime_bindings.h"
#include <cstring>

extern "C" int sk_moe_swiglu_pair(void* x, void* W_gate, void* W_up,
                                  void* exp_ids, void* out,
                                  uint32_t T, uint32_t top_k, uint32_t E,
                                  uint32_t D, uint32_t N_int) {
    auto* pso = sk::bindings_pso("moe_swiglu_pair");
    if (!pso) return -1;
    auto* dev = sk::bindings_device();

    const size_t xb     = (size_t)T  * D * 2;
    const size_t wgb    = (size_t)E  * N_int * D * 2;
    const size_t idxb   = (size_t)T  * top_k * sizeof(int32_t);
    const size_t outb   = (size_t)T  * top_k * N_int * 2;

    auto* bX  = dev->newBuffer(xb,   MTL::ResourceStorageModeShared);
    auto* bWG = dev->newBuffer(wgb,  MTL::ResourceStorageModeShared);
    auto* bWU = dev->newBuffer(wgb,  MTL::ResourceStorageModeShared);
    auto* bId = dev->newBuffer(idxb, MTL::ResourceStorageModeShared);
    auto* bO  = dev->newBuffer(outb, MTL::ResourceStorageModeShared);
    std::memcpy(bX->contents(),  x,       xb);
    std::memcpy(bWG->contents(), W_gate,  wgb);
    std::memcpy(bWU->contents(), W_up,    wgb);
    std::memcpy(bId->contents(), exp_ids, idxb);

    auto* cmd = sk::bindings_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bX,  0, 0); enc->setBuffer(bWG, 0, 1); enc->setBuffer(bWU, 0, 2);
    enc->setBuffer(bId, 0, 3); enc->setBuffer(bO,  0, 4);
    enc->setBytes(&T,     4, 5); enc->setBytes(&top_k, 4, 6);
    enc->setBytes(&D,     4, 7); enc->setBytes(&N_int, 4, 8);

    const uint32_t COLS_PER_TG = 16;
    enc->dispatchThreadgroups(
        MTL::Size((N_int + COLS_PER_TG - 1) / COLS_PER_TG, T * top_k, 1),
        MTL::Size(256, 1, 1));
    enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted();
    std::memcpy(out, bO->contents(), outb);

    cmd->release();
    bX->release(); bWG->release(); bWU->release(); bId->release(); bO->release();
    return 0;
}
