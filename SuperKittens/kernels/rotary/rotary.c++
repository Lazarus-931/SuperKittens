#include "rotary.h"
#include "../runtime_bindings.h"

int sk_rope(void* Q, void* K, void* cos, void* sin,
            uint32_t seq, uint32_t head_dim, uint32_t n_heads) {
    auto* pso = sk::bindings_pso("rope_qk");
    if (!pso) return -1;

    size_t bytes = (size_t)n_heads * seq * head_dim * sizeof(__fp16);
    size_t cs = (size_t)seq * (head_dim / 2) * sizeof(__fp16);
    auto* bQ  = sk::bindings_device()->newBuffer(bytes, MTL::ResourceStorageModeShared);
    auto* bK  = sk::bindings_device()->newBuffer(bytes, MTL::ResourceStorageModeShared);
    auto* bCos = sk::bindings_device()->newBuffer(cs, MTL::ResourceStorageModeShared);
    auto* bSin = sk::bindings_device()->newBuffer(cs, MTL::ResourceStorageModeShared);
    memcpy(bQ->contents(), Q, bytes); memcpy(bK->contents(), K, bytes);
    memcpy(bCos->contents(), cos, cs); memcpy(bSin->contents(), sin, cs);

    auto* cmd = sk::bindings_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bQ, 0, 0); enc->setBuffer(bK, 0, 1);
    enc->setBuffer(bCos, 0, 2); enc->setBuffer(bSin, 0, 3);
    enc->setBytes(&seq, sizeof(uint32_t), 4);
    enc->setBytes(&head_dim, sizeof(uint32_t), 5);
    enc->setBytes(&n_heads, sizeof(uint32_t), 6);
    uint32_t hd4 = head_dim / 8;
    if (hd4 == 0) hd4 = 1;
    uint32_t rows_per_tg = 256 / hd4;
    if (rows_per_tg < 1) rows_per_tg = 1;
    if (rows_per_tg > seq) rows_per_tg = seq;
    uint32_t row_blks = (seq + rows_per_tg - 1) / rows_per_tg;
    enc->dispatchThreadgroups(MTL::Size(n_heads, row_blks, 1),
                              MTL::Size(hd4, rows_per_tg, 1));
    enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted();
    memcpy(Q, bQ->contents(), bytes); memcpy(K, bK->contents(), bytes);

    cmd->release(); bQ->release(); bK->release(); bCos->release(); bSin->release();
    return 0;
}
