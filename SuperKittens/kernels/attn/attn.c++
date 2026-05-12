#include "attn.h"
#include "../runtime_bindings.h"

int sk_attn(void* Q, void* K, void* V, void* O,
            uint32_t seq, uint32_t head_dim,
            uint32_t nheads, uint32_t n_kv_heads, int causal) {
    const bool is64 = (head_dim == 64);
    const char* kname = is64
        ? (causal ? "fa_causal_64" : "fa_noncausal_64")
        : (causal ? "mha_causal"   : "mha_noncausal");
    auto* pso = sk::bindings_pso(kname);
    if (!pso) return -1;

    const size_t q_nb = (size_t)nheads    * seq * head_dim * sizeof(__fp16);
    const size_t k_nb = (size_t)n_kv_heads * seq * head_dim * sizeof(__fp16);
    auto* dev = sk::bindings_device();
    auto* bQ = dev->newBuffer(q_nb, MTL::ResourceStorageModeShared);
    auto* bK = dev->newBuffer(k_nb, MTL::ResourceStorageModeShared);
    auto* bV = dev->newBuffer(k_nb, MTL::ResourceStorageModeShared);
    auto* bO = dev->newBuffer(q_nb, MTL::ResourceStorageModeShared);
    memcpy(bQ->contents(), Q, q_nb);
    memcpy(bK->contents(), K, k_nb);
    memcpy(bV->contents(), V, k_nb);

    const uint32_t kv_len = seq;
    const uint32_t cache_stride = seq;
    const uint32_t Hg = nheads / n_kv_heads;
    const uint gx = is64 ? nheads     : n_kv_heads;
    const uint gy = is64 ? ((seq + 31) / 32) : ((seq + 1) / 2);
    const uint threads = is64 ? 1024 : (Hg * 2 * 32);

    auto* cmd = sk::bindings_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bQ, 0, 0); enc->setBuffer(bK, 0, 1);
    enc->setBuffer(bV, 0, 2); enc->setBuffer(bO, 0, 3);
    enc->setBytes(&seq, 4, 4);
    if (is64) {
        enc->setBytes(&nheads, 4, 5);
    } else {
        enc->setBytes(&nheads,       4, 5);
        enc->setBytes(&n_kv_heads,   4, 6);
        enc->setBytes(&kv_len,       4, 7);
        enc->setBytes(&cache_stride, 4, 8);
    }
    enc->dispatchThreadgroups(MTL::Size(gx, gy, 1), MTL::Size(threads, 1, 1));
    enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted();
    memcpy(O, bO->contents(), q_nb);

    cmd->release(); bQ->release(); bK->release(); bV->release(); bO->release();
    return 0;
}
