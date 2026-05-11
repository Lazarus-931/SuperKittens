#include "mamba2.h"
#include "../../kernels/runtime_bindings.h"

int sk_mamba2_ssd_state(void* Q, void* K, void* V, void* A_log, void* y,
                        void* h_state_in, void* h_state_out,
                        uint32_t B, uint32_t L, uint32_t Ds, uint32_t Dv, uint32_t H) {
    auto* pso = sk::bindings_pso("mamba2_ssd");
    if (!pso) return -1;

    size_t qb = (size_t)B*H*L*Ds*2, vb = (size_t)B*H*L*Dv*2, ab = (size_t)B*H*L*2;
    size_t sb = (size_t)B*H*Ds*Dv*4;
    auto* dev = sk::bindings_device();
    auto* bQ = dev->newBuffer(qb, MTL::ResourceStorageModeShared);
    auto* bK = dev->newBuffer(qb, MTL::ResourceStorageModeShared);
    auto* bV = dev->newBuffer(vb, MTL::ResourceStorageModeShared);
    auto* bA = dev->newBuffer(ab, MTL::ResourceStorageModeShared);
    auto* bY = dev->newBuffer(vb, MTL::ResourceStorageModeShared);
    auto* bSi = dev->newBuffer(sb, MTL::ResourceStorageModeShared);
    auto* bSo = dev->newBuffer(sb, MTL::ResourceStorageModeShared);
    memcpy(bQ->contents(), Q, qb); memcpy(bK->contents(), K, qb);
    memcpy(bV->contents(), V, vb); memcpy(bA->contents(), A_log, ab);
    uint32_t flags = 0;
    if (h_state_in)  { flags |= 1u; memcpy(bSi->contents(), h_state_in, sb); }
    if (h_state_out) { flags |= 2u; }

    auto* cmd = sk::bindings_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bQ, 0, 0); enc->setBuffer(bK, 0, 1); enc->setBuffer(bV, 0, 2);
    enc->setBuffer(bA, 0, 3); enc->setBuffer(bY, 0, 4);
    enc->setBytes(&L, 4, 5); enc->setBytes(&Ds, 4, 6); enc->setBytes(&Dv, 4, 7); enc->setBytes(&H, 4, 8);
    enc->setBuffer(bSi, 0, 9); enc->setBuffer(bSo, 0, 10);
    enc->setBytes(&flags, 4, 11);
    // v2: Dv-sliced grid (BV=32) when Dv % 32 == 0; falls back to v1's
    // grid layout otherwise (smaller threadgroups; still correct).
    const uint32_t BV = 32;
    if (Dv % BV == 0) {
        enc->dispatchThreadgroups(MTL::Size(B * H, Dv / BV, 1),
                                  MTL::Size(64, 1, 1));
    } else {
        enc->dispatchThreadgroups(MTL::Size(B, H, 1),
                                  MTL::Size(128, 1, 1));
    }
    enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted();
    memcpy(y, bY->contents(), vb);
    if (h_state_out) memcpy(h_state_out, bSo->contents(), sb);

    cmd->release(); bQ->release(); bK->release(); bV->release(); bA->release(); bY->release();
    bSi->release(); bSo->release();
    return 0;
}

int sk_mamba2_ssd(void* Q, void* K, void* V, void* A_log, void* y,
                  uint32_t B, uint32_t L, uint32_t Ds, uint32_t Dv, uint32_t H) {
    return sk_mamba2_ssd_state(Q, K, V, A_log, y, nullptr, nullptr, B, L, Ds, Dv, H);
}

// ===== sk_mamba2_step ===== single-token decode kernel
int sk_mamba2_step(void* x_t, void* B_t, void* C_t, void* A_log_t,
                   void* h_state, void* y_t,
                   uint32_t BH, uint32_t Ds, uint32_t Dv) {
    auto* pso = sk::bindings_pso("mamba2_step");
    if (!pso) return -1;

    size_t xb = (size_t)BH*Dv*2, sb = (size_t)BH*Ds*2, cb = (size_t)BH*Ds*2;
    size_t ab = (size_t)BH*2,    yb = (size_t)BH*Dv*2, hb = (size_t)BH*Ds*Dv*4;
    auto* dev = sk::bindings_device();
    auto* bX = dev->newBuffer(xb, MTL::ResourceStorageModeShared);
    auto* bB = dev->newBuffer(sb, MTL::ResourceStorageModeShared);
    auto* bC = dev->newBuffer(cb, MTL::ResourceStorageModeShared);
    auto* bA = dev->newBuffer(ab, MTL::ResourceStorageModeShared);
    auto* bH = dev->newBuffer(hb, MTL::ResourceStorageModeShared);
    auto* bY = dev->newBuffer(yb, MTL::ResourceStorageModeShared);
    memcpy(bX->contents(), x_t, xb); memcpy(bB->contents(), B_t, sb);
    memcpy(bC->contents(), C_t, cb); memcpy(bA->contents(), A_log_t, ab);
    memcpy(bH->contents(), h_state, hb);

    auto* cmd = sk::bindings_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bX, 0, 0); enc->setBuffer(bB, 0, 1); enc->setBuffer(bC, 0, 2);
    enc->setBuffer(bA, 0, 3); enc->setBuffer(bH, 0, 4); enc->setBuffer(bY, 0, 5);
    enc->setBytes(&Ds, 4, 6); enc->setBytes(&Dv, 4, 7);
    enc->dispatchThreadgroups(MTL::Size(BH, 1, 1), MTL::Size(64, 1, 1));
    enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted();
    memcpy(y_t, bY->contents(), yb);
    memcpy(h_state, bH->contents(), hb);

    cmd->release();
    bX->release(); bB->release(); bC->release(); bA->release(); bH->release(); bY->release();
    return 0;
}

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
