#include "mamba3.h"
#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>
#include <cstring>

static MTL::Device* g_d = nullptr; static MTL::CommandQueue* g_q = nullptr; static MTL::Library* g_l = nullptr;
static MTL::ComputePipelineState* g_pso_pre  = nullptr;
static MTL::ComputePipelineState* g_pso_ssm  = nullptr;
static MTL::ComputePipelineState* g_pso_post = nullptr;
static void ensure() {
    if (g_d) return;
    g_d = MTL::CreateSystemDefaultDevice(); g_q = g_d->newCommandQueue();
    const char* env = getenv("SK_METALLIB");
    const char* p = env ? env : "build/libsk.metallib";
    NS::Error* e = nullptr;
    auto* u = NS::URL::fileURLWithPath(NS::String::string(p, NS::UTF8StringEncoding));
    g_l = g_d->newLibrary(u, &e);
}

int sk_mamba3_pre_ssm(void* xBC, void* dt, void* angle, void* norm_w,
                      void* Q_out, void* K_out, void* V_out,
                      void* A_out, void* B_out,
                      uint32_t BH, uint32_t L, uint32_t DQ, float eps) {
    ensure(); if (!g_l) return -1;
    if (!g_pso_pre) {
        auto* fn = g_l->newFunction(NS::String::string("mamba3_pre_ssm", NS::UTF8StringEncoding)); if (!fn) return -2;
        NS::Error* e = nullptr; g_pso_pre = g_d->newComputePipelineState(fn, &e); fn->release();
        if (!g_pso_pre) { if (e) e->release(); return -3; }
    }

    size_t xb=(size_t)BH*L*2*DQ*2, db=(size_t)BH*L*2, ab=(size_t)BH*L*(DQ/2)*2, nb=(size_t)DQ*2;
    size_t ob=(size_t)BH*L*DQ*2;

    auto* bX  =g_d->newBuffer(xb,MTL::ResourceStorageModeShared); auto* bDt=g_d->newBuffer(db,MTL::ResourceStorageModeShared);
    auto* bAng=g_d->newBuffer(ab,MTL::ResourceStorageModeShared); auto* bNw=g_d->newBuffer(nb,MTL::ResourceStorageModeShared);
    auto* bQ  =g_d->newBuffer(ob,MTL::ResourceStorageModeShared); auto* bK=g_d->newBuffer(ob,MTL::ResourceStorageModeShared);
    auto* bV  =g_d->newBuffer(ob,MTL::ResourceStorageModeShared);
    auto* bA  =g_d->newBuffer(db,MTL::ResourceStorageModeShared); auto* bB=g_d->newBuffer(ob,MTL::ResourceStorageModeShared);
    memcpy(bX->contents(),xBC,xb); memcpy(bDt->contents(),dt,db); memcpy(bAng->contents(),angle,ab); memcpy(bNw->contents(),norm_w,nb);

    auto* c=g_q->commandBuffer(); auto* enc=c->computeCommandEncoder(); enc->setComputePipelineState(g_pso_pre);
    enc->setBuffer(bX,0,0);enc->setBuffer(bDt,0,1);enc->setBuffer(bAng,0,2);enc->setBuffer(bNw,0,3);
    enc->setBuffer(bQ,0,4);enc->setBuffer(bK,0,5);enc->setBuffer(bV,0,6);enc->setBuffer(bA,0,7);enc->setBuffer(bB,0,8);
    enc->setBytes(&BH,sizeof(uint32_t),9);enc->setBytes(&L,sizeof(uint32_t),10);
    enc->setBytes(&DQ,sizeof(uint32_t),11);enc->setBytes(&eps,sizeof(float),12);
    enc->dispatchThreadgroups(MTL::Size(BH,(L+3)/4,1),MTL::Size(128,1,1)); enc->endEncoding(); c->commit(); c->waitUntilCompleted();
    memcpy(Q_out,bQ->contents(),ob); memcpy(K_out,bK->contents(),ob); memcpy(V_out,bV->contents(),ob);
    memcpy(A_out,bA->contents(),db); memcpy(B_out,bB->contents(),ob);
    c->release();
    bX->release();bDt->release();bAng->release();bNw->release();bQ->release();bK->release();bV->release();bA->release();bB->release();
    return 0;
}

int sk_mamba3_ssm(void* Q, void* K, void* V, void* A, void* B, void* angle,
                  void* O, uint32_t BH, uint32_t L, uint32_t DQ, uint32_t DV, uint32_t CS) {
    ensure(); if (!g_l) return -1;
    if (!g_pso_ssm) {
        auto* fn = g_l->newFunction(NS::String::string("mamba3_ssm", NS::UTF8StringEncoding)); if (!fn) return -2;
        NS::Error* e = nullptr; g_pso_ssm = g_d->newComputePipelineState(fn, &e); fn->release();
        if (!g_pso_ssm) { if (e) e->release(); return -3; }
    }

    size_t qb=(size_t)BH*L*DQ*2, vb=(size_t)BH*L*DV*2, ab=(size_t)BH*L*2, angb=(size_t)BH*L*(DQ/2)*2;
    auto* bQ  =g_d->newBuffer(qb,MTL::ResourceStorageModeShared); auto* bK=g_d->newBuffer(qb,MTL::ResourceStorageModeShared);
    auto* bV  =g_d->newBuffer(vb,MTL::ResourceStorageModeShared);
    auto* bA  =g_d->newBuffer(ab,MTL::ResourceStorageModeShared); auto* bB=g_d->newBuffer(ab,MTL::ResourceStorageModeShared);
    auto* bAng=g_d->newBuffer(angb,MTL::ResourceStorageModeShared); auto* bO=g_d->newBuffer(vb,MTL::ResourceStorageModeShared);
    memcpy(bQ->contents(),Q,qb);memcpy(bK->contents(),K,qb);memcpy(bV->contents(),V,vb);
    memcpy(bA->contents(),A,ab);memcpy(bB->contents(),B,ab);memcpy(bAng->contents(),angle,angb);

    auto* c=g_q->commandBuffer(); auto* enc=c->computeCommandEncoder(); enc->setComputePipelineState(g_pso_ssm);
    enc->setBuffer(bQ,0,0);enc->setBuffer(bK,0,1);enc->setBuffer(bV,0,2);enc->setBuffer(bA,0,3);enc->setBuffer(bB,0,4);
    enc->setBuffer(bAng,0,5);enc->setBuffer(bO,0,6);
    enc->setBytes(&L, sizeof(uint32_t),7);enc->setBytes(&DQ,sizeof(uint32_t),8);
    enc->setBytes(&DV,sizeof(uint32_t),9);enc->setBytes(&CS,sizeof(uint32_t),10);
    enc->dispatchThreadgroups(MTL::Size(BH,1,1),MTL::Size(128,1,1)); enc->endEncoding(); c->commit(); c->waitUntilCompleted();
    memcpy(O,bO->contents(),vb); c->release();
    bQ->release();bK->release();bV->release();bA->release();bB->release();bAng->release();bO->release();
    return 0;
}

int sk_mamba3_post_ssm(void* z, void* ssm_out, void* norm_w, void* gated,
                       uint32_t BH, uint32_t L, uint32_t DV, float eps) {
    ensure(); if (!g_l) return -1;
    if (!g_pso_post) {
        auto* fn = g_l->newFunction(NS::String::string("mamba3_post_ssm", NS::UTF8StringEncoding)); if (!fn) return -2;
        NS::Error* e = nullptr; g_pso_post = g_d->newComputePipelineState(fn, &e); fn->release();
        if (!g_pso_post) { if (e) e->release(); return -3; }
    }

    size_t xb=(size_t)BH*L*DV*2, wb=(size_t)DV*2;
    auto* bZ=g_d->newBuffer(xb,MTL::ResourceStorageModeShared); auto* bS=g_d->newBuffer(xb,MTL::ResourceStorageModeShared);
    auto* bW=g_d->newBuffer(wb,MTL::ResourceStorageModeShared); auto* bY=g_d->newBuffer(xb,MTL::ResourceStorageModeShared);
    memcpy(bZ->contents(),z,xb);memcpy(bS->contents(),ssm_out,xb);memcpy(bW->contents(),norm_w,wb);

    auto* c=g_q->commandBuffer(); auto* enc=c->computeCommandEncoder(); enc->setComputePipelineState(g_pso_post);
    enc->setBuffer(bZ,0,0);enc->setBuffer(bS,0,1);enc->setBuffer(bW,0,2);enc->setBuffer(bY,0,3);
    enc->setBytes(&L,  sizeof(uint32_t),4);enc->setBytes(&DV,sizeof(uint32_t),5);enc->setBytes(&eps,sizeof(float),6);
    enc->dispatchThreadgroups(MTL::Size(BH,(L+3)/4,1),MTL::Size(128,1,1)); enc->endEncoding(); c->commit(); c->waitUntilCompleted();
    memcpy(gated,bY->contents(),xb); c->release();bZ->release();bS->release();bW->release();bY->release();
    return 0;
}
