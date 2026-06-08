#include "mamba2.h"
#include "../../../kernels/runtime_bindings.h"

// The legacy Q/K/V `sk_mamba2_ssd*` / `sk_mamba2_step` stubs were removed:
// mamba2_ssd.metal now carries the HF-correct (x, dt, A_log, B, C, D, dt_bias)
// signature dispatched by the launcher (dispatch_layer); the old buffer layout
// would misbind it. Prefill/decode both route through the launcher now.

int sk_conv1d_silu(void* x, void* weight, void* bias, void* y, uint32_t B, uint32_t L, uint32_t C) {
    auto* pso = sk::bindings_pso("conv1d_silu");
    if (!pso) return -1;

    size_t xb = (size_t)B*L*C*2, wb = (size_t)C*4*2, bb = (size_t)C*2;
    auto* bX = sk::bindings_device()->newBuffer(xb, MTL::ResourceStorageModeShared);
    auto* bW = sk::bindings_device()->newBuffer(wb, MTL::ResourceStorageModeShared);
    auto* bBias = sk::bindings_device()->newBuffer(bb, MTL::ResourceStorageModeShared);
    auto* bY = sk::bindings_device()->newBuffer(xb, MTL::ResourceStorageModeShared);
    memcpy(bX->contents(), x, xb); memcpy(bW->contents(), weight, wb);
    memcpy(bBias->contents(), bias, bb);

    auto* cmd = sk::bindings_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bX, 0, 0); enc->setBuffer(bW, 0, 1); enc->setBuffer(bBias, 0, 2);
    enc->setBuffer(bY, 0, 3);
    enc->setBytes(&B, 4, 4); enc->setBytes(&L, 4, 5); enc->setBytes(&C, 4, 6);
    enc->dispatchThreadgroups(MTL::Size(B, (L+3)/4, 1), MTL::Size(128, 1, 1));
    enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted();
    memcpy(y, bY->contents(), xb);

    cmd->release(); bX->release(); bW->release(); bBias->release(); bY->release();
    return 0;
}

int sk_gate_norm(void* ssm_out, void* z, void* weight, void* y,
                 uint32_t B, uint32_t L, uint32_t E, float eps) {
    auto* pso = sk::bindings_pso("gate_norm");
    if (!pso) return -1;

    size_t xb = (size_t)B*L*E*2, wb = (size_t)E*2;
    auto* bS = sk::bindings_device()->newBuffer(xb, MTL::ResourceStorageModeShared);
    auto* bZ = sk::bindings_device()->newBuffer(xb, MTL::ResourceStorageModeShared);
    auto* bW = sk::bindings_device()->newBuffer(wb, MTL::ResourceStorageModeShared);
    auto* bY = sk::bindings_device()->newBuffer(xb, MTL::ResourceStorageModeShared);
    memcpy(bS->contents(), ssm_out, xb); memcpy(bZ->contents(), z, xb);
    memcpy(bW->contents(), weight, wb);

    auto* cmd = sk::bindings_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bS, 0, 0); enc->setBuffer(bZ, 0, 1);
    enc->setBuffer(bW, 0, 2); enc->setBuffer(bY, 0, 3);
    enc->setBytes(&L, 4, 4); enc->setBytes(&E, 4, 5); enc->setBytes(&eps, 4, 6);
    enc->dispatchThreadgroups(MTL::Size(B, (L+3)/4, 1), MTL::Size(128, 1, 1));
    enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted();
    memcpy(y, bY->contents(), xb);

    cmd->release(); bS->release(); bZ->release(); bW->release(); bY->release();
    return 0;
}
