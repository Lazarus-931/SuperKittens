//
//  down_scatter.c++ — host launcher for fused MoE down + scatter + residual.
//

#include "down_scatter.h"
#include "../runtime_bindings.h"
#include <cstring>

extern "C" int sk_moe_down_scatter(void* hidden, void* W_down, void* exp_ids,
                                   void* route_w, void* residual, void* out,
                                   uint32_t T, uint32_t top_k, uint32_t E,
                                   uint32_t D, uint32_t N_int) {
    auto* pso = sk::bindings_pso("moe_down_scatter");
    if (!pso) return -1;
    auto* dev = sk::bindings_device();

    const size_t hb   = (size_t)T  * top_k * N_int * 2;
    const size_t wb   = (size_t)E  * D * N_int * 2;
    const size_t idxb = (size_t)T  * top_k * sizeof(int32_t);
    const size_t rwb  = (size_t)T  * top_k * 2;
    const size_t rsb  = (size_t)T  * D * 2;

    auto* bH  = dev->newBuffer(hb,   MTL::ResourceStorageModeShared);
    auto* bW  = dev->newBuffer(wb,   MTL::ResourceStorageModeShared);
    auto* bId = dev->newBuffer(idxb, MTL::ResourceStorageModeShared);
    auto* bRw = dev->newBuffer(rwb,  MTL::ResourceStorageModeShared);
    auto* bRs = dev->newBuffer(rsb,  MTL::ResourceStorageModeShared);
    auto* bO  = dev->newBuffer(rsb,  MTL::ResourceStorageModeShared);
    std::memcpy(bH->contents(),  hidden,   hb);
    std::memcpy(bW->contents(),  W_down,   wb);
    std::memcpy(bId->contents(), exp_ids,  idxb);
    std::memcpy(bRw->contents(), route_w,  rwb);
    std::memcpy(bRs->contents(), residual, rsb);

    auto* cmd = sk::bindings_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bH,  0, 0); enc->setBuffer(bW,  0, 1);
    enc->setBuffer(bId, 0, 2); enc->setBuffer(bRw, 0, 3);
    enc->setBuffer(bRs, 0, 4); enc->setBuffer(bO,  0, 5);
    enc->setBytes(&T,     4, 6); enc->setBytes(&top_k, 4, 7);
    enc->setBytes(&D,     4, 8); enc->setBytes(&N_int, 4, 9);

    const uint32_t COLS_PER_TG = 16;
    enc->dispatchThreadgroups(
        MTL::Size((D + COLS_PER_TG - 1) / COLS_PER_TG, T, 1),
        MTL::Size(256, 1, 1));
    enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted();
    std::memcpy(out, bO->contents(), rsb);

    cmd->release();
    bH->release(); bW->release(); bId->release(); bRw->release();
    bRs->release(); bO->release();
    return 0;
}
