#include "mamba2.h"
#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>
#include <cstring>

static MTL::Device* g_d = nullptr; static MTL::CommandQueue* g_q = nullptr; static MTL::Library* g_l = nullptr;
static MTL::ComputePipelineState* g_pso_ssd  = nullptr;
static MTL::ComputePipelineState* g_pso_conv = nullptr;
static MTL::ComputePipelineState* g_pso_gate = nullptr;
static void ensure() {
    if (g_d) return;
    g_d = MTL::CreateSystemDefaultDevice(); g_q = g_d->newCommandQueue();
    const char* env = getenv("SK_METALLIB");
    const char* p = env ? env : "build/libsk.metallib";
    NS::Error* e = nullptr;
    auto* u = NS::URL::fileURLWithPath(NS::String::string(p, NS::UTF8StringEncoding));
    g_l = g_d->newLibrary(u, &e);
}

int sk_mamba2_ssd(void* Q, void* K, void* V, void* A_log, void* y,
                  uint32_t B, uint32_t L, uint32_t Ds, uint32_t Dv, uint32_t H) {
    ensure(); if (!g_l) return -1;
    if (!g_pso_ssd) {
        auto* fn = g_l->newFunction(NS::String::string("mamba2_ssd", NS::UTF8StringEncoding)); if (!fn) return -2;
        NS::Error* e = nullptr; g_pso_ssd = g_d->newComputePipelineState(fn, &e); fn->release();
        if (!g_pso_ssd) { if (e) e->release(); return -3; }
    }

    size_t qb = (size_t)B*H*L*Ds*2, vb = (size_t)B*H*L*Dv*2, ab = (size_t)B*H*L*2;
    auto* bQ=g_d->newBuffer(qb,MTL::ResourceStorageModeShared); auto* bK=g_d->newBuffer(qb,MTL::ResourceStorageModeShared);
    auto* bV=g_d->newBuffer(vb,MTL::ResourceStorageModeShared); auto* bA=g_d->newBuffer(ab,MTL::ResourceStorageModeShared);
    auto* bY=g_d->newBuffer(vb,MTL::ResourceStorageModeShared);
    memcpy(bQ->contents(),Q,qb); memcpy(bK->contents(),K,qb); memcpy(bV->contents(),V,vb); memcpy(bA->contents(),A_log,ab);

    auto* c=g_q->commandBuffer(); auto* enc=c->computeCommandEncoder(); enc->setComputePipelineState(g_pso_ssd);
    enc->setBuffer(bQ,0,0); enc->setBuffer(bK,0,1); enc->setBuffer(bV,0,2); enc->setBuffer(bA,0,3); enc->setBuffer(bY,0,4);
    enc->setBytes(&L, sizeof(uint32_t),5); enc->setBytes(&Ds,sizeof(uint32_t),6);
    enc->setBytes(&Dv,sizeof(uint32_t),7); enc->setBytes(&H, sizeof(uint32_t),8);
    enc->dispatchThreadgroups(MTL::Size(B,H,1),MTL::Size(128,1,1)); enc->endEncoding(); c->commit(); c->waitUntilCompleted();
    memcpy(y,bY->contents(),vb); c->release();
    bQ->release();bK->release();bV->release();bA->release();bY->release();
    return 0;
}

int sk_conv1d_silu(void* x, void* weight, void* bias, void* y, uint32_t B, uint32_t L, uint32_t C) {
    ensure(); if (!g_l) return -1;
    if (!g_pso_conv) {
        auto* fn = g_l->newFunction(NS::String::string("conv1d_silu", NS::UTF8StringEncoding)); if (!fn) return -2;
        NS::Error* e = nullptr; g_pso_conv = g_d->newComputePipelineState(fn, &e); fn->release();
        if (!g_pso_conv) { if (e) e->release(); return -3; }
    }

    size_t xb=(size_t)B*L*C*2, wb=(size_t)C*4*2, bb=(size_t)C*2;
    auto* bX   =g_d->newBuffer(xb,MTL::ResourceStorageModeShared); auto* bW=g_d->newBuffer(wb,MTL::ResourceStorageModeShared);
    auto* bBias=g_d->newBuffer(bb,MTL::ResourceStorageModeShared); auto* bY=g_d->newBuffer(xb,MTL::ResourceStorageModeShared);
    memcpy(bX->contents(),x,xb); memcpy(bW->contents(),weight,wb); memcpy(bBias->contents(),bias,bb);

    auto* c=g_q->commandBuffer(); auto* enc=c->computeCommandEncoder(); enc->setComputePipelineState(g_pso_conv);
    enc->setBuffer(bX,0,0);enc->setBuffer(bW,0,1);enc->setBuffer(bBias,0,2);enc->setBuffer(bY,0,3);
    enc->setBytes(&B,sizeof(uint32_t),4);enc->setBytes(&L,sizeof(uint32_t),5);enc->setBytes(&C,sizeof(uint32_t),6);
    enc->dispatchThreadgroups(MTL::Size(B,(L+3)/4,1),MTL::Size(128,1,1)); enc->endEncoding(); c->commit(); c->waitUntilCompleted();
    memcpy(y,bY->contents(),xb); c->release();bX->release();bW->release();bBias->release();bY->release();
    return 0;
}

int sk_gate_norm(void* ssm_out, void* z, void* weight, void* y, uint32_t B, uint32_t L, uint32_t E, float eps) {
    ensure(); if (!g_l) return -1;
    if (!g_pso_gate) {
        auto* fn = g_l->newFunction(NS::String::string("gate_norm", NS::UTF8StringEncoding)); if (!fn) return -2;
        NS::Error* e = nullptr; g_pso_gate = g_d->newComputePipelineState(fn, &e); fn->release();
        if (!g_pso_gate) { if (e) e->release(); return -3; }
    }

    size_t xb=(size_t)B*L*E*2, wb=(size_t)E*2;
    auto* bS=g_d->newBuffer(xb,MTL::ResourceStorageModeShared); auto* bZ=g_d->newBuffer(xb,MTL::ResourceStorageModeShared);
    auto* bW=g_d->newBuffer(wb,MTL::ResourceStorageModeShared); auto* bY=g_d->newBuffer(xb,MTL::ResourceStorageModeShared);
    memcpy(bS->contents(),ssm_out,xb); memcpy(bZ->contents(),z,xb); memcpy(bW->contents(),weight,wb);

    auto* c=g_q->commandBuffer(); auto* enc=c->computeCommandEncoder(); enc->setComputePipelineState(g_pso_gate);
    enc->setBuffer(bS,0,0);enc->setBuffer(bZ,0,1);enc->setBuffer(bW,0,2);enc->setBuffer(bY,0,3);
    enc->setBytes(&L,  sizeof(uint32_t),4);enc->setBytes(&E,  sizeof(uint32_t),5);enc->setBytes(&eps,sizeof(float),6);
    enc->dispatchThreadgroups(MTL::Size(B,(L+3)/4,1),MTL::Size(128,1,1)); enc->endEncoding(); c->commit(); c->waitUntilCompleted();
    memcpy(y,bY->contents(),xb); c->release();bS->release();bZ->release();bW->release();bY->release();
    return 0;
}
