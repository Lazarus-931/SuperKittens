#include "swiglu.h"
#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>
#include <cstring>
static MTL::Device* g_d=nullptr; static MTL::CommandQueue* g_q=nullptr; static MTL::Library* g_l=nullptr;
static void ensure(){if(g_d)return;g_d=MTL::CreateSystemDefaultDevice();g_q=g_d->newCommandQueue();
  const char* p=getenv("SK_METALLIB")?getenv("SK_METALLIB"):"build/libsk.metallib";
  NS::Error* e=nullptr;auto* u=NS::URL::fileURLWithPath(NS::String::string(p,NS::UTF8StringEncoding));g_l=g_d->newLibrary(u,&e);}

int sk_swiglu(void* x, void* y, uint32_t rows, uint32_t d) {
    ensure(); if(!g_l) return -1;
    auto* fn=g_l->newFunction(NS::String::string("fused_swiglu",NS::UTF8StringEncoding)); if(!fn) return -2;
    NS::Error* e=nullptr; auto* pso=g_d->newComputePipelineState(fn,&e); fn->release(); if(!pso) return -3;

    size_t xb=(size_t)rows*d*2;
    auto* bX=g_d->newBuffer(xb,MTL::ResourceStorageModeShared);
    auto* bY=g_d->newBuffer(xb,MTL::ResourceStorageModeShared);
    memcpy(bX->contents(),x,xb);
    auto* bR=g_d->newBuffer(&rows,sizeof(uint32_t),MTL::ResourceStorageModeShared);
    auto* bD=g_d->newBuffer(&d,sizeof(uint32_t),MTL::ResourceStorageModeShared);

    auto* c=g_q->commandBuffer(); auto* enc=c->computeCommandEncoder(); enc->setComputePipelineState(pso);
    enc->setBuffer(bX,0,0);enc->setBuffer(bY,0,1);enc->setBuffer(bR,0,2);enc->setBuffer(bD,0,3);
    enc->dispatchThreadgroups(MTL::Size(1,(rows+3)/4,1),MTL::Size(128,1,1));
    enc->endEncoding();c->commit();c->waitUntilCompleted();
    memcpy(y,bY->contents(),xb); c->release();pso->release();bX->release();bY->release();bR->release();bD->release();
    return 0;
}
