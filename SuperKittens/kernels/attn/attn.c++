#include "attn.h"
#include "../runtime_bindings.h"

int sk_attn(void* Q, void* K, void* V, void* O, uint32_t seq, uint32_t head_dim, uint32_t nheads, int causal) {
    const char* kname = (head_dim == 64)
        ? (causal ? "fa_causal_64" : "fa_noncausal_64")
        : (causal ? "mha_causal"   : "mha_noncausal");
    auto* pso = sk::bindings_pso(kname);
    if (!pso) return -1;

    size_t nb = (size_t)nheads * seq * head_dim * sizeof(__fp16);
    auto* bQ = sk::bindings_device()->newBuffer(nb, MTL::ResourceStorageModeShared);
    auto* bK = sk::bindings_device()->newBuffer(nb, MTL::ResourceStorageModeShared);
    auto* bV = sk::bindings_device()->newBuffer(nb, MTL::ResourceStorageModeShared);
    auto* bO = sk::bindings_device()->newBuffer(nb, MTL::ResourceStorageModeShared);
    memcpy(bQ->contents(), Q, nb); memcpy(bK->contents(), K, nb); memcpy(bV->contents(), V, nb);

    bool is64 = (head_dim == 64);
    uint gy = is64 ? ((seq + 31) / 32) : ((seq + 3) / 4);
    uint threads = is64 ? 1024 : 128;

    auto* cmd = sk::bindings_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bQ, 0, 0); enc->setBuffer(bK, 0, 1); enc->setBuffer(bV, 0, 2); enc->setBuffer(bO, 0, 3);
    enc->setBytes(&seq, sizeof(uint32_t), 4);
    if (is64) { enc->setBytes(&nheads, sizeof(uint32_t), 5); }
    else      { enc->setBytes(&head_dim, sizeof(uint32_t), 5); enc->setBytes(&nheads, sizeof(uint32_t), 6); }
    enc->dispatchThreadgroups(MTL::Size(nheads, gy, 1), MTL::Size(threads, 1, 1));
    enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted();
    memcpy(O, bO->contents(), nb);

    cmd->release(); bQ->release(); bK->release(); bV->release(); bO->release();
    return 0;
}
