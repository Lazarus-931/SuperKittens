#include "mamba3.h"
#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>
#include <cstring>

static MTL::Device* g_d = nullptr; static MTL::CommandQueue* g_q = nullptr; static MTL::Library* g_l = nullptr;
static void ensure() {
    if (g_d) return;
    g_d = MTL::CreateSystemDefaultDevice(); g_q = g_d->newCommandQueue();
    const char* p = getenv("SK_METALLIB") ? getenv("SK_METALLIB") : "build/libsk.metallib";
    NS::Error* e = nullptr;
    auto* u = NS::URL::fileURLWithPath(NS::String::string(p, NS::UTF8StringEncoding));
    g_l = g_d->newLibrary(u, &e);
}

int sk_mamba3_pre_ssm(void* xBC, void* dt, void* angle, void* norm_w,
                      void* Q_out, void* K_out, void* V_out,
                      void* A_out, void* B_out,
                      uint32_t BH, uint32_t L, uint32_t DQ, float eps) {
    ensure(); if (!g_l) return -1;
    auto* fn = g_l->newFunction(NS::String::string("mamba3_pre_ssm", NS::UTF8StringEncoding)); if (!fn) return -2;
    NS::Error* e = nullptr; auto* pso = g_d->newComputePipelineState(fn, &e); fn->release(); if (!pso) return -3;

    size_t xb=(size_t)BH*L*2*DQ*2, db=(size_t)BH*L*2, ab=(size_t)BH*L*(DQ/2)*2, nb=(size_t)DQ*2;
    size_t ob=(size_t)BH*L*DQ*2;

    auto* bX=g_d->newBuffer(xb,MTL::ResourceStorageModeShared); auto* bDt=g_d->newBuffer(db,MTL::ResourceStorageModeShared);
    auto* bAng=g_d->newBuffer(ab,MTL::ResourceStorageModeShared); auto* bNw=g_d->newBuffer(nb,MTL::ResourceStorageModeShared);
    auto* bQ=g_d->newBuffer(ob,MTL::ResourceStorageModeShared); auto* bK=g_d->newBuffer(ob,MTL::ResourceStorageModeShared);
    auto* bV=g_d->newBuffer(ob,MTL::ResourceStorageModeShared);
    auto* bA=g_d->newBuffer(db,MTL::ResourceStorageModeShared); auto* bB=g_d->newBuffer(ob,MTL::ResourceStorageModeShared);
    memcpy(bX->contents(),xBC,xb); memcpy(bDt->contents(),dt,db); memcpy(bAng->contents(),angle,ab); memcpy(bNw->contents(),norm_w,nb);

    auto* bBH=g_d->newBuffer(&BH,sizeof(uint32_t),MTL::ResourceStorageModeShared);
    auto* bL=g_d->newBuffer(&L,sizeof(uint32_t),MTL::ResourceStorageModeShared);
    auto* bDQ=g_d->newBuffer(&DQ,sizeof(uint32_t),MTL::ResourceStorageModeShared);
    auto* bEps=g_d->newBuffer(&eps,sizeof(float),MTL::ResourceStorageModeShared);

    auto* c=g_q->commandBuffer(); auto* enc=c->computeCommandEncoder(); enc->setComputePipelineState(pso);
    enc->setBuffer(bX,0,0);enc->setBuffer(bDt,0,1);enc->setBuffer(bAng,0,2);enc->setBuffer(bNw,0,3);
    enc->setBuffer(bQ,0,4);enc->setBuffer(bK,0,5);enc->setBuffer(bV,0,6);enc->setBuffer(bA,0,7);enc->setBuffer(bB,0,8);
    enc->setBuffer(bBH,0,9);enc->setBuffer(bL,0,10);enc->setBuffer(bDQ,0,11);enc->setBuffer(bEps,0,12);
    enc->dispatchThreadgroups(MTL::Size(BH,(L+3)/4,1),MTL::Size(128,1,1)); enc->endEncoding(); c->commit(); c->waitUntilCompleted();
    memcpy(Q_out,bQ->contents(),ob); memcpy(K_out,bK->contents(),ob); memcpy(V_out,bV->contents(),ob);
    memcpy(A_out,bA->contents(),db); memcpy(B_out,bB->contents(),ob);
    c->release();pso->release();
    bX->release();bDt->release();bAng->release();bNw->release();bQ->release();bK->release();bV->release();bA->release();bB->release();
    bBH->release();bL->release();bDQ->release();bEps->release();
    return 0;
}

int sk_mamba3_ssm(void* Q, void* K, void* V, void* A, void* B, void* angle,
                  void* O, uint32_t BH, uint32_t L, uint32_t DQ, uint32_t DV, uint32_t CS) {
    ensure(); if (!g_l) return -1;
    auto* fn = g_l->newFunction(NS::String::string("mamba3_ssm", NS::UTF8StringEncoding)); if (!fn) return -2;
    NS::Error* e = nullptr; auto* pso = g_d->newComputePipelineState(fn, &e); fn->release(); if (!pso) return -3;

    size_t qb=(size_t)BH*L*DQ*2, vb=(size_t)BH*L*DV*2, ab=(size_t)BH*L*2, angb=(size_t)BH*L*(DQ/2)*2;
    auto* bQ=g_d->newBuffer(qb,MTL::ResourceStorageModeShared); auto* bK=g_d->newBuffer(qb,MTL::ResourceStorageModeShared);
    auto* bV=g_d->newBuffer(vb,MTL::ResourceStorageModeShared);
    auto* bA=g_d->newBuffer(ab,MTL::ResourceStorageModeShared); auto* bB=g_d->newBuffer(ab,MTL::ResourceStorageModeShared);
    auto* bAng=g_d->newBuffer(angb,MTL::ResourceStorageModeShared); auto* bO=g_d->newBuffer(vb,MTL::ResourceStorageModeShared);
    memcpy(bQ->contents(),Q,qb);memcpy(bK->contents(),K,qb);memcpy(bV->contents(),V,vb);
    memcpy(bA->contents(),A,ab);memcpy(bB->contents(),B,ab);memcpy(bAng->contents(),angle,angb);

    auto* bL=g_d->newBuffer(&L,sizeof(uint32_t),MTL::ResourceStorageModeShared);
    auto* bDQ=g_d->newBuffer(&DQ,sizeof(uint32_t),MTL::ResourceStorageModeShared);
    auto* bDV=g_d->newBuffer(&DV,sizeof(uint32_t),MTL::ResourceStorageModeShared);
    auto* bCS=g_d->newBuffer(&CS,sizeof(uint32_t),MTL::ResourceStorageModeShared);

    auto* c=g_q->commandBuffer(); auto* enc=c->computeCommandEncoder(); enc->setComputePipelineState(pso);
    enc->setBuffer(bQ,0,0);enc->setBuffer(bK,0,1);enc->setBuffer(bV,0,2);enc->setBuffer(bA,0,3);enc->setBuffer(bB,0,4);
    enc->setBuffer(bAng,0,5);enc->setBuffer(bO,0,6);
    enc->setBuffer(bL,0,7);enc->setBuffer(bDQ,0,8);enc->setBuffer(bDV,0,9);enc->setBuffer(bCS,0,10);
    enc->dispatchThreadgroups(MTL::Size(BH,1,1),MTL::Size(128,1,1)); enc->endEncoding(); c->commit(); c->waitUntilCompleted();
    memcpy(O,bO->contents(),vb); c->release();pso->release();
    bQ->release();bK->release();bV->release();bA->release();bB->release();bAng->release();bO->release();
    bL->release();bDQ->release();bDV->release();bCS->release();
    return 0;
}

int sk_mamba3_post_ssm(void* z, void* ssm_out, void* norm_w, void* gated,
                       uint32_t BH, uint32_t L, uint32_t DV, float eps) {
    ensure(); if (!g_l) return -1;
    auto* fn = g_l->newFunction(NS::String::string("mamba3_post_ssm", NS::UTF8StringEncoding)); if (!fn) return -2;
    NS::Error* e = nullptr; auto* pso = g_d->newComputePipelineState(fn, &e); fn->release(); if (!pso) return -3;

    size_t xb=(size_t)BH*L*DV*2, wb=(size_t)DV*2;
    auto* bZ=g_d->newBuffer(xb,MTL::ResourceStorageModeShared); auto* bS=g_d->newBuffer(xb,MTL::ResourceStorageModeShared);
    auto* bW=g_d->newBuffer(wb,MTL::ResourceStorageModeShared); auto* bY=g_d->newBuffer(xb,MTL::ResourceStorageModeShared);
    memcpy(bZ->contents(),z,xb);memcpy(bS->contents(),ssm_out,xb);memcpy(bW->contents(),norm_w,wb);

    auto* bL=g_d->newBuffer(&L,sizeof(uint32_t),MTL::ResourceStorageModeShared);
    auto* bDV=g_d->newBuffer(&DV,sizeof(uint32_t),MTL::ResourceStorageModeShared);
    auto* bEps=g_d->newBuffer(&eps,sizeof(float),MTL::ResourceStorageModeShared);

    auto* c=g_q->commandBuffer(); auto* enc=c->computeCommandEncoder(); enc->setComputePipelineState(pso);
    enc->setBuffer(bZ,0,0);enc->setBuffer(bS,0,1);enc->setBuffer(bW,0,2);enc->setBuffer(bY,0,3);
    enc->setBuffer(bL,0,4);enc->setBuffer(bDV,0,5);enc->setBuffer(bEps,0,6);
    enc->dispatchThreadgroups(MTL::Size(BH,(L+3)/4,1),MTL::Size(128,1,1)); enc->endEncoding(); c->commit(); c->waitUntilCompleted();
    memcpy(gated,bY->contents(),xb); c->release();pso->release();bZ->release();bS->release();bW->release();bY->release();bL->release();bDV->release();bEps->release();
    return 0;
}
