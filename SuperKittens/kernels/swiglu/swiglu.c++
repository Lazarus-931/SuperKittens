#include "swiglu.h"
#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>
#include <cstring>
static MTL::Device* g_d=nullptr; static MTL::CommandQueue* g_q=nullptr; static MTL::Library* g_l=nullptr;
static MTL::ComputePipelineState* g_pso=nullptr;
static void ensure(){if(g_d)return;g_d=MTL::CreateSystemDefaultDevice();g_q=g_d->newCommandQueue();
  const char* env=getenv("SK_METALLIB"); const char* p=env?env:"build/libsk.metallib";
  NS::Error* e=nullptr;auto* u=NS::URL::fileURLWithPath(NS::String::string(p,NS::UTF8StringEncoding));g_l=g_d->newLibrary(u,&e);}

// d = output columns (half-width); input shape is (rows, 2*d), output shape is (rows, d)
int sk_swiglu(void* x, void* y, uint32_t rows, uint32_t d) {
    ensure(); if(!g_l) return -1;
    if(!g_pso){
        auto* fn=g_l->newFunction(NS::String::string("fused_swiglu",NS::UTF8StringEncoding)); if(!fn) return -2;
        NS::Error* e=nullptr; g_pso=g_d->newComputePipelineState(fn,&e); fn->release(); if(!g_pso) return -3;
    }
    size_t in_b=(size_t)rows*d*4, out_b=(size_t)rows*d*2;
    auto* bX=g_d->newBuffer(in_b,MTL::ResourceStorageModeShared);
    auto* bY=g_d->newBuffer(out_b,MTL::ResourceStorageModeShared);
    memcpy(bX->contents(),x,in_b);

    auto* c=g_q->commandBuffer(); auto* enc=c->computeCommandEncoder(); enc->setComputePipelineState(g_pso);
    enc->setBuffer(bX,0,0);enc->setBuffer(bY,0,1);
    enc->setBytes(&rows,sizeof(uint32_t),2);enc->setBytes(&d,sizeof(uint32_t),3);
    enc->dispatchThreadgroups(MTL::Size(1,(rows+3)/4,1),MTL::Size(128,1,1));
    enc->endEncoding();c->commit();c->waitUntilCompleted();
    memcpy(y,bY->contents(),out_b); c->release();bX->release();bY->release();
    return 0;
}
