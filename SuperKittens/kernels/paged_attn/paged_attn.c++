#include "paged_attn.h"
#include "../runtime_bindings.h"

int sk_paged_attn(void* Q, void* K_cache, void* V_cache, void* block_table,
                  void* seq_lens, void* O, uint32_t num_seqs, uint32_t num_heads,
                  uint32_t head_dim, uint32_t num_kv_heads, uint32_t block_size,
                  uint32_t max_blocks) {
    auto* pso = sk::bindings_pso("paged_attn");
    if (!pso) return -1;

    size_t qb = (size_t)num_seqs * num_heads * head_dim * 2, ob = qb;
    size_t kcb = (size_t)max_blocks * block_size * num_kv_heads * head_dim * 2;
    size_t btb = (size_t)num_seqs * max_blocks * 4, slb = (size_t)num_seqs * 4;

    auto* bQ  = sk::bindings_device()->newBuffer(qb, MTL::ResourceStorageModeShared);
    auto* bK  = sk::bindings_device()->newBuffer(kcb, MTL::ResourceStorageModeShared);
    auto* bV  = sk::bindings_device()->newBuffer(kcb, MTL::ResourceStorageModeShared);
    auto* bBT = sk::bindings_device()->newBuffer(btb, MTL::ResourceStorageModeShared);
    auto* bSL = sk::bindings_device()->newBuffer(slb, MTL::ResourceStorageModeShared);
    auto* bO  = sk::bindings_device()->newBuffer(ob, MTL::ResourceStorageModeShared);
    memcpy(bQ->contents(), Q, qb); memcpy(bK->contents(), K_cache, kcb);
    memcpy(bV->contents(), V_cache, kcb); memcpy(bBT->contents(), block_table, btb);
    memcpy(bSL->contents(), seq_lens, slb);

    uint32_t ss = num_heads * head_dim, bls = block_size * num_kv_heads * head_dim;

    auto* cmd = sk::bindings_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bQ, 0, 0); enc->setBuffer(bK, 0, 1); enc->setBuffer(bV, 0, 2);
    enc->setBuffer(bBT, 0, 3); enc->setBuffer(bSL, 0, 4); enc->setBuffer(bO, 0, 5);
    enc->setBytes(&num_seqs, 4, 6); enc->setBytes(&num_heads, 4, 7);
    enc->setBytes(&head_dim, 4, 8); enc->setBytes(&num_kv_heads, 4, 9);
    enc->setBytes(&block_size, 4, 10); enc->setBytes(&max_blocks, 4, 11);
    enc->setBytes(&ss, 4, 12); enc->setBytes(&bls, 4, 13);
    enc->dispatchThreadgroups(MTL::Size(num_seqs, (num_heads + 3) / 4, 1), MTL::Size(128, 1, 1));
    enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted();
    memcpy(O, bO->contents(), ob);

    cmd->release(); bQ->release(); bK->release(); bV->release(); bBT->release(); bSL->release(); bO->release();
    return 0;
}
