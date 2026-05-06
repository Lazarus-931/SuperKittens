#include "attn.h"
#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>
#include <cstring>

static MTL::Device* g_d=nullptr; static MTL::CommandQueue* g_q=nullptr; static MTL::Library* g_l=nullptr;
// indexed by (is64 ? 2 : 0) + (causal ? 1 : 0)
static MTL::ComputePipelineState* g_psos[4] = {};
static void ensure(){if(g_d)return;g_d=MTL::CreateSystemDefaultDevice();g_q=g_d->newCommandQueue();
  const char* env=getenv("SK_METALLIB"); const char* p=env?env:"build/libsk.metallib";
  NS::Error* e=nullptr;auto* u=NS::URL::fileURLWithPath(NS::String::string(p,NS::UTF8StringEncoding));g_l=g_d->newLibrary(u,&e);}

int sk_attn(void* Q, void* K, void* V, void* O, uint32_t seq, uint32_t head_dim, uint32_t nheads, int causal) {
    ensure(); if(!g_l) return -1;
    bool is64=(head_dim==64);
    int idx=(is64?2:0)+(causal?1:0);
    if(!g_psos[idx]){
        const char* kname = is64
            ? (causal?"fa_causal_64":"fa_noncausal_64")
            : (causal?"mha_causal":"mha_noncausal");
        auto* fn=g_l->newFunction(NS::String::string(kname,NS::UTF8StringEncoding)); if(!fn) return -2;
        NS::Error* e=nullptr; g_psos[idx]=g_d->newComputePipelineState(fn,&e); fn->release();
        if(!g_psos[idx]){if(e)e->release();return -3;}
    }

    size_t nb=(size_t)nheads*seq*head_dim*2;
    auto* bQ=g_d->newBuffer(nb,MTL::ResourceStorageModeShared);
    auto* bK=g_d->newBuffer(nb,MTL::ResourceStorageModeShared);
    auto* bV=g_d->newBuffer(nb,MTL::ResourceStorageModeShared);
    auto* bO=g_d->newBuffer(nb,MTL::ResourceStorageModeShared);
    memcpy(bQ->contents(),Q,nb);memcpy(bK->contents(),K,nb);memcpy(bV->contents(),V,nb);

    uint gy=is64?((seq+31)/32):((seq+3)/4);
    uint threads=is64?1024:128;

    auto* c=g_q->commandBuffer(); auto* enc=c->computeCommandEncoder(); enc->setComputePipelineState(g_psos[idx]);
    enc->setBuffer(bQ,0,0);enc->setBuffer(bK,0,1);enc->setBuffer(bV,0,2);enc->setBuffer(bO,0,3);
    enc->setBytes(&seq,sizeof(uint32_t),4);
    if(is64){ enc->setBytes(&nheads,sizeof(uint32_t),5); }
    else    { enc->setBytes(&head_dim,sizeof(uint32_t),5); enc->setBytes(&nheads,sizeof(uint32_t),6); }
    enc->dispatchThreadgroups(MTL::Size(nheads,gy,1),MTL::Size(threads,1,1));
    enc->endEncoding();c->commit();c->waitUntilCompleted();
    memcpy(O,bO->contents(),nb); c->release();
    bQ->release();bK->release();bV->release();bO->release();
    return 0;
}
