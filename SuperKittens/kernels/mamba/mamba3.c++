#include "mamba3.h"
#include "../runtime_bindings.h"

int sk_mamba3_pre_ssm(void* xBC, void* dt, void* angle, void* norm_w,
                      void* Q_out, void* K_out, void* V_out,
                      void* A_out, void* B_out, uint32_t BH, uint32_t L, uint32_t DQ, float eps) {
    auto* pso = sk::bindings_pso("mamba3_pre_ssm");
    if (!pso) return -1;

    size_t xb = (size_t)BH*L*2*DQ*2, db = (size_t)BH*L*2, ab = (size_t)BH*L*(DQ/2)*2;
    size_t nb = (size_t)DQ*2, ob = (size_t)BH*L*DQ*2;
    auto* bX   = sk::bindings_device()->newBuffer(xb, MTL::ResourceStorageModeShared);
    auto* bDt  = sk::bindings_device()->newBuffer(db, MTL::ResourceStorageModeShared);
    auto* bAng = sk::bindings_device()->newBuffer(ab, MTL::ResourceStorageModeShared);
    auto* bNw  = sk::bindings_device()->newBuffer(nb, MTL::ResourceStorageModeShared);
    auto* bQ   = sk::bindings_device()->newBuffer(ob, MTL::ResourceStorageModeShared);
    auto* bK   = sk::bindings_device()->newBuffer(ob, MTL::ResourceStorageModeShared);
    auto* bV   = sk::bindings_device()->newBuffer(ob, MTL::ResourceStorageModeShared);
    auto* bA   = sk::bindings_device()->newBuffer(db, MTL::ResourceStorageModeShared);
    auto* bB   = sk::bindings_device()->newBuffer(ob, MTL::ResourceStorageModeShared);
    memcpy(bX->contents(), xBC, xb); memcpy(bDt->contents(), dt, db);
    memcpy(bAng->contents(), angle, ab); memcpy(bNw->contents(), norm_w, nb);

    auto* cmd = sk::bindings_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bX, 0, 0); enc->setBuffer(bDt, 0, 1); enc->setBuffer(bAng, 0, 2);
    enc->setBuffer(bNw, 0, 3); enc->setBuffer(bQ, 0, 4); enc->setBuffer(bK, 0, 5);
    enc->setBuffer(bV, 0, 6); enc->setBuffer(bA, 0, 7); enc->setBuffer(bB, 0, 8);
    enc->setBytes(&BH, 4, 9); enc->setBytes(&L, 4, 10); enc->setBytes(&DQ, 4, 11); enc->setBytes(&eps, 4, 12);
    enc->dispatchThreadgroups(MTL::Size(BH, (L+3)/4, 1), MTL::Size(128, 1, 1));
    enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted();
    memcpy(Q_out, bQ->contents(), ob); memcpy(K_out, bK->contents(), ob);
    memcpy(V_out, bV->contents(), ob); memcpy(A_out, bA->contents(), db);
    memcpy(B_out, bB->contents(), ob);

    cmd->release(); bX->release(); bDt->release(); bAng->release(); bNw->release();
    bQ->release(); bK->release(); bV->release(); bA->release(); bB->release();
    return 0;
}

int sk_mamba3_ssm(void* Q, void* K, void* V, void* A, void* B, void* angle,
                  void* O, uint32_t BH, uint32_t L, uint32_t DQ, uint32_t DV, uint32_t CS) {
    auto* pso = sk::bindings_pso("mamba3_ssm");
    if (!pso) return -1;

    size_t qb = (size_t)BH*L*DQ*2, vb = (size_t)BH*L*DV*2, ab = (size_t)BH*L*2, angb = (size_t)BH*L*(DQ/2)*2;
    auto* bQ   = sk::bindings_device()->newBuffer(qb, MTL::ResourceStorageModeShared);
    auto* bK   = sk::bindings_device()->newBuffer(qb, MTL::ResourceStorageModeShared);
    auto* bV   = sk::bindings_device()->newBuffer(vb, MTL::ResourceStorageModeShared);
    auto* bA   = sk::bindings_device()->newBuffer(ab, MTL::ResourceStorageModeShared);
    auto* bB   = sk::bindings_device()->newBuffer(ab, MTL::ResourceStorageModeShared);
    auto* bAng = sk::bindings_device()->newBuffer(angb, MTL::ResourceStorageModeShared);
    auto* bO   = sk::bindings_device()->newBuffer(vb, MTL::ResourceStorageModeShared);
    memcpy(bQ->contents(), Q, qb); memcpy(bK->contents(), K, qb);
    memcpy(bV->contents(), V, vb); memcpy(bA->contents(), A, ab);
    memcpy(bB->contents(), B, ab); memcpy(bAng->contents(), angle, angb);

    auto* cmd = sk::bindings_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bQ, 0, 0); enc->setBuffer(bK, 0, 1); enc->setBuffer(bV, 0, 2);
    enc->setBuffer(bA, 0, 3); enc->setBuffer(bB, 0, 4); enc->setBuffer(bAng, 0, 5);
    enc->setBuffer(bO, 0, 6); enc->setBytes(&L, 4, 7); enc->setBytes(&DQ, 4, 8);
    enc->setBytes(&DV, 4, 9); enc->setBytes(&CS, 4, 10);
    enc->dispatchThreadgroups(MTL::Size(BH, 1, 1), MTL::Size(128, 1, 1));
    enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted();
    memcpy(O, bO->contents(), vb);

    cmd->release(); bQ->release(); bK->release(); bV->release(); bA->release(); bB->release(); bAng->release(); bO->release();
    return 0;
}

int sk_mamba3_post_ssm(void* z, void* ssm_out, void* norm_w, void* gated,
                       uint32_t BH, uint32_t L, uint32_t DV, float eps) {
    auto* pso = sk::bindings_pso("mamba3_post_ssm");
    if (!pso) return -1;

    size_t xb = (size_t)BH*L*DV*2, wb = (size_t)DV*2;
    auto* bZ = sk::bindings_device()->newBuffer(xb, MTL::ResourceStorageModeShared);
    auto* bS = sk::bindings_device()->newBuffer(xb, MTL::ResourceStorageModeShared);
    auto* bW = sk::bindings_device()->newBuffer(wb, MTL::ResourceStorageModeShared);
    auto* bY = sk::bindings_device()->newBuffer(xb, MTL::ResourceStorageModeShared);
    memcpy(bZ->contents(), z, xb); memcpy(bS->contents(), ssm_out, xb);
    memcpy(bW->contents(), norm_w, wb);

    auto* cmd = sk::bindings_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bZ, 0, 0); enc->setBuffer(bS, 0, 1);
    enc->setBuffer(bW, 0, 2); enc->setBuffer(bY, 0, 3);
    enc->setBytes(&L, 4, 4); enc->setBytes(&DV, 4, 5); enc->setBytes(&eps, 4, 6);
    enc->dispatchThreadgroups(MTL::Size(BH, (L+3)/4, 1), MTL::Size(128, 1, 1));
    enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted();
    memcpy(gated, bY->contents(), xb);

    cmd->release(); bZ->release(); bS->release(); bW->release(); bY->release();
    return 0;
}
