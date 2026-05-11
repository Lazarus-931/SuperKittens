//
//  swiglu_pair_iq2xxs.c++ — host launcher for the IQ2_XXS-quantized
//  fused MoE swiglu pair.
//

#include "swiglu_pair_iq2xxs.h"
#include "../runtime_bindings.h"
#include <cstring>

extern "C" int sk_moe_swiglu_pair_iq2xxs(void* x, void* W_gate, void* W_up,
                                         void* exp_ids, void* out,
                                         uint32_t T, uint32_t top_k, uint32_t E,
                                         uint32_t D, uint32_t N_int) {
    auto* pso = sk::bindings_pso("moe_swiglu_pair_iq2xxs");
    if (!pso) return -1;
    if (D % 256) return -4;
    auto* dev = sk::bindings_device();

    // 66 bytes per 256-weight IQ2_XXS block (half d + ushort qs[32]).
    const size_t IQ2XXS_BLOCK_BYTES = 66;
    const size_t n_blocks_per_row   = D / 256;

    const size_t xb   = (size_t)T * D * 2;
    const size_t wb   = (size_t)E * N_int * n_blocks_per_row * IQ2XXS_BLOCK_BYTES;
    const size_t idxb = (size_t)T * top_k * sizeof(int32_t);
    const size_t outb = (size_t)T * top_k * N_int * 2;

    auto* bX  = dev->newBuffer(xb,   MTL::ResourceStorageModeShared);
    auto* bWG = dev->newBuffer(wb,   MTL::ResourceStorageModeShared);
    auto* bWU = dev->newBuffer(wb,   MTL::ResourceStorageModeShared);
    auto* bId = dev->newBuffer(idxb, MTL::ResourceStorageModeShared);
    auto* bO  = dev->newBuffer(outb, MTL::ResourceStorageModeShared);
    std::memcpy(bX->contents(),  x,       xb);
    std::memcpy(bWG->contents(), W_gate,  wb);
    std::memcpy(bWU->contents(), W_up,    wb);
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
