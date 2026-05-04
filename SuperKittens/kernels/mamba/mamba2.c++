#include "mamba2.h"
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

int sk_mamba2_ssd(void* Q, void* K, void* V, void* A_log, void* y,
                  uint32_t B, uint32_t L, uint32_t Ds, uint32_t Dv, uint32_t H) {
    ensure(); if (!g_l) return -1;
    auto* fn = g_l->newFunction(NS::String::string("mamba2_ssd", NS::UTF8StringEncoding)); if (!fn) return -2;
    NS::Error* e = nullptr; auto* pso = g_d->newComputePipelineState(fn, &e); fn->release(); if (!pso) return -3;

    size_t qb = (size_t)B*H*L*Ds*2, vb = (size_t)B*H*L*Dv*2, ab = (size_t)B*H*L*2;
    auto* bQ=g_d->newBuffer(qb,MTL::ResourceStorageModeShared); auto* bK=g_d->newBuffer(qb,MTL::ResourceStorageModeShared);
    auto* bV=g_d->newBuffer(vb,MTL::ResourceStorageModeShared); auto* bA=g_d->newBuffer(ab,MTL::ResourceStorageModeShared);
    auto* bY=g_d->newBuffer(vb,MTL::ResourceStorageModeShared);
    memcpy(bQ->contents(),Q,qb); memcpy(bK->contents(),K,qb); memcpy(bV->contents(),V,vb); memcpy(bA->contents(),A_log,ab);

    auto* bL=g_d->newBuffer(&L,sizeof(uint32_t),MTL::ResourceStorageModeShared);
    auto* bDs=g_d->newBuffer(&Ds,sizeof(uint32_t),MTL::ResourceStorageModeShared);
    auto* bDv=g_d->newBuffer(&Dv,sizeof(uint32_t),MTL::ResourceStorageModeShared);
    auto* bH=g_d->newBuffer(&H,sizeof(uint32_t),MTL::ResourceStorageModeShared);

    auto* c=g_q->commandBuffer(); auto* enc=c->computeCommandEncoder(); enc->setComputePipelineState(pso);
    enc->setBuffer(bQ,0,0); enc->setBuffer(bK,0,1); enc->setBuffer(bV,0,2); enc->setBuffer(bA,0,3); enc->setBuffer(bY,0,4);
    enc->setBuffer(bL,0,5); enc->setBuffer(bDs,0,6); enc->setBuffer(bDv,0,7); enc->setBuffer(bH,0,8);
    enc->dispatchThreadgroups(MTL::Size(B,H,1),MTL::Size(128,1,1)); enc->endEncoding(); c->commit(); c->waitUntilCompleted();
    memcpy(y,bY->contents(),vb); c->release(); pso->release();
    bQ->release();bK->release();bV->release();bA->release();bY->release();bL->release();bDs->release();bDv->release();bH->release();
    return 0;
}

int sk_conv1d_silu(void* x, void* weight, void* bias, void* y, uint32_t B, uint32_t L, uint32_t C) {
    ensure(); if (!g_l) return -1;
    auto* fn = g_l->newFunction(NS::String::string("conv1d_silu", NS::UTF8StringEncoding)); if (!fn) return -2;
    NS::Error* e = nullptr; auto* pso = g_d->newComputePipelineState(fn, &e); fn->release(); if (!pso) return -3;

    size_t xb=(size_t)B*L*C*2, wb=(size_t)C*4*2, bb=(size_t)C*2;
    auto* bX=g_d->newBuffer(xb,MTL::ResourceStorageModeShared); auto* bW=g_d->newBuffer(wb,MTL::ResourceStorageModeShared);
    auto* bB=g_d->newBuffer(bb,MTL::ResourceStorageModeShared); auto* bY=g_d->newBuffer(xb,MTL::ResourceStorageModeShared);
    memcpy(bX->contents(),x,xb); memcpy(bW->contents(),weight,wb); memcpy(bB->contents(),bias,bb);

    auto* bB1=g_d->newBuffer(&B,sizeof(uint32_t),MTL::ResourceStorageModeShared);
    auto* bL=g_d->newBuffer(&L,sizeof(uint32_t),MTL::ResourceStorageModeShared);
    auto* bC=g_d->newBuffer(&C,sizeof(uint32_t),MTL::ResourceStorageModeShared);

    auto* c=g_q->commandBuffer(); auto* enc=c->computeCommandEncoder(); enc->setComputePipelineState(pso);
    enc->setBuffer(bX,0,0);enc->setBuffer(bW,0,1);enc->setBuffer(bB,0,2);enc->setBuffer(bY,0,3);
    enc->setBuffer(bB1,0,4);enc->setBuffer(bL,0,5);enc->setBuffer(bC,0,6);
    enc->dispatchThreadgroups(MTL::Size(B,(L+3)/4,1),MTL::Size(128,1,1)); enc->endEncoding(); c->commit(); c->waitUntilCompleted();
    memcpy(y,bY->contents(),xb); c->release();pso->release();bX->release();bW->release();bB->release();bY->release();bB1->release();bL->release();bC->release();
    return 0;
}

int sk_gate_norm(void* ssm_out, void* z, void* weight, void* y, uint32_t B, uint32_t L, uint32_t E, float eps) {
    ensure(); if (!g_l) return -1;
    auto* fn = g_l->newFunction(NS::String::string("gate_norm", NS::UTF8StringEncoding)); if (!fn) return -2;
    NS::Error* err = nullptr; auto* pso = g_d->newComputePipelineState(fn, &err); fn->release(); if (!pso) return -3;

    size_t xb=(size_t)B*L*E*2, wb=(size_t)E*2;
    auto* bS=g_d->newBuffer(xb,MTL::ResourceStorageModeShared); auto* bZ=g_d->newBuffer(xb,MTL::ResourceStorageModeShared);
    auto* bW=g_d->newBuffer(wb,MTL::ResourceStorageModeShared); auto* bY=g_d->newBuffer(xb,MTL::ResourceStorageModeShared);
    memcpy(bS->contents(),ssm_out,xb); memcpy(bZ->contents(),z,xb); memcpy(bW->contents(),weight,wb);

    auto* bL=g_d->newBuffer(&L,sizeof(uint32_t),MTL::ResourceStorageModeShared);
    auto* bE=g_d->newBuffer(&E,sizeof(uint32_t),MTL::ResourceStorageModeShared);
    auto* bEps=g_d->newBuffer(&eps,sizeof(float),MTL::ResourceStorageModeShared);

    auto* c=g_q->commandBuffer(); auto* enc=c->computeCommandEncoder(); enc->setComputePipelineState(pso);
    enc->setBuffer(bS,0,0);enc->setBuffer(bZ,0,1);enc->setBuffer(bW,0,2);enc->setBuffer(bY,0,3);
    enc->setBuffer(bL,0,4);enc->setBuffer(bE,0,5);enc->setBuffer(bEps,0,6);
    enc->dispatchThreadgroups(MTL::Size(B,(L+3)/4,1),MTL::Size(128,1,1)); enc->endEncoding(); c->commit(); c->waitUntilCompleted();
    memcpy(y,bY->contents(),xb); c->release();pso->release();bS->release();bZ->release();bW->release();bY->release();bL->release();bE->release();bEps->release();
    return 0;
}
