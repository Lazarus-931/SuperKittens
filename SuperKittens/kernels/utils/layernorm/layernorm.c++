#include "layernorm.h"
#include "../../runtime_bindings.h"

static int dispatch(const char* name, void* x, void* w, void* b, void* y,
                    uint32_t rows, uint32_t d, float eps) {
    auto* pso = sk::bindings_pso(name);
    if (!pso) return -1;

    size_t xb = (size_t)rows * d * sizeof(__fp16);
    size_t db = (size_t)d * sizeof(__fp16);
    auto* bX = sk::bindings_device()->newBuffer(xb, MTL::ResourceStorageModeShared);
    auto* bW = sk::bindings_device()->newBuffer(db, MTL::ResourceStorageModeShared);
    auto* bY = sk::bindings_device()->newBuffer(xb, MTL::ResourceStorageModeShared);
    memcpy(bX->contents(), x, xb); memcpy(bW->contents(), w, db);

    auto* bB = b ? sk::bindings_device()->newBuffer(db, MTL::ResourceStorageModeShared) : nullptr;
    if (bB) memcpy(bB->contents(), b, db);

    bool is_ln = (b != nullptr);
    auto* cmd = sk::bindings_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bX, 0, 0); enc->setBuffer(bW, 0, 1);
    if (is_ln) {
        enc->setBuffer(bB, 0, 2); enc->setBuffer(bY, 0, 3);
        enc->setBytes(&rows, sizeof(uint32_t), 4); enc->setBytes(&d, sizeof(uint32_t), 5); enc->setBytes(&eps, sizeof(float), 6);
    } else {
        enc->setBuffer(bY, 0, 2);
        enc->setBytes(&rows, sizeof(uint32_t), 3); enc->setBytes(&d, sizeof(uint32_t), 4); enc->setBytes(&eps, sizeof(float), 5);
    }
    enc->dispatchThreadgroups(MTL::Size(1, (rows + 3) / 4, 1), MTL::Size(128, 1, 1));
    enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted();
    memcpy(y, bY->contents(), xb);

    cmd->release(); bX->release(); bW->release(); if (bB) bB->release(); bY->release();
    return 0;
}

int sk_layernorm(void* x, void* gamma, void* beta, void* y, uint32_t rows, uint32_t d, float eps)
    { return dispatch("layernorm", x, gamma, beta, y, rows, d, eps); }

int sk_rmsnorm(void* x, void* weight, void* y, uint32_t rows, uint32_t d, float eps)
    { return dispatch("rmsnorm", x, weight, nullptr, y, rows, d, eps); }
