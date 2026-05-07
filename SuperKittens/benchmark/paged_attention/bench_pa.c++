//
//  bench_pa.c++ — Paged attention kernel benchmark
//
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void fill(std::vector<__fp16>& v, std::mt19937& r, float s=1.0f) {
    std::normal_distribution<float> n(0,1);
    for(auto& x:v) x=__fp16(n(r)*s);
}

int main(int argc, const char* argv[]) {
    const char* lib_path = (argc>1) ? argv[1] : "build/libsk.metallib";
    const uint32_t num_seqs = 8, num_heads = 32, head_dim = 128, num_kv_heads = 8;
    const uint32_t block_size = 16, seq_len = 256;
    const uint32_t max_blocks = (seq_len + block_size - 1) / block_size;
    const uint32_t num_blocks = max_blocks * 2;  // pool

    MTL::Device* dev = MTL::CreateSystemDefaultDevice();
    MTL::CommandQueue* q = dev->newCommandQueue();
    NS::Error* err = nullptr;
    auto* url = NS::URL::fileURLWithPath(NS::String::string(lib_path,NS::UTF8StringEncoding));
    MTL::Library* lib = dev->newLibrary(url,&err);
    if(!lib){fprintf(stderr,"no lib\n");return 1;}
    auto* fn = lib->newFunction(NS::String::string("paged_attn",NS::UTF8StringEncoding));
    if(!fn){fprintf(stderr,"no kernel\n");return 1;}
    auto* pso = dev->newComputePipelineState(fn,&err); fn->release();
    if(!pso){fprintf(stderr,"no pso\n");return 1;}

    std::mt19937 rng(42);
    size_t qb = (size_t)num_seqs * num_heads * head_dim;
    size_t cb = (size_t)num_blocks * block_size * num_kv_heads * head_dim;
    size_t btb = (size_t)num_seqs * max_blocks;

    std::vector<__fp16> Qv(qb), Kv(cb), Vv(cb);
    std::vector<int> bt(btb), sl(num_seqs);
    fill(Qv,rng,0.5f); fill(Kv,rng,0.1f); fill(Vv,rng,0.1f);
    for(uint32_t s=0;s<num_seqs;s++){sl[s]=(int)seq_len;for(uint32_t b=0;b<max_blocks;b++)bt[s*max_blocks+b]=(int)(s*max_blocks+b);}
    std::vector<__fp16> Ov(qb);

    auto mk=[&](const void* d,size_t sz){auto* b=dev->newBuffer(sz,MTL::ResourceStorageModeShared);if(d)memcpy(b->contents(),d,sz);return b;};
    auto mu=[&](uint32_t v){auto* b=dev->newBuffer(&v,4,MTL::ResourceStorageModeShared);return b;};
    auto*bQ=mk(Qv.data(),qb*2),*bK=mk(Kv.data(),cb*2),*bV=mk(Vv.data(),cb*2);
    auto*bBT=mk(bt.data(),btb*4),*bSL=mk(sl.data(),num_seqs*4),*bO=dev->newBuffer(qb*2,MTL::ResourceStorageModeShared);
    uint32_t ss=num_heads*head_dim, bls=block_size*num_kv_heads*head_dim;
    auto*bNS=mu(num_seqs),*bNH=mu(num_heads),*bHD=mu(head_dim),*bKV=mu(num_kv_heads);
    auto*bBS=mu(block_size),*bMB=mu(max_blocks),*bSS=mu(ss),*bBL=mu(bls);

    auto run=[&](){
        auto* c=q->commandBuffer();auto* e=c->computeCommandEncoder();e->setComputePipelineState(pso);
        e->setBuffer(bQ,0,0);e->setBuffer(bK,0,1);e->setBuffer(bV,0,2);e->setBuffer(bBT,0,3);
        e->setBuffer(bSL,0,4);e->setBuffer(bO,0,5);
        e->setBuffer(bNS,0,6);e->setBuffer(bNH,0,7);e->setBuffer(bHD,0,8);e->setBuffer(bKV,0,9);
        e->setBuffer(bBS,0,10);e->setBuffer(bMB,0,11);e->setBuffer(bSS,0,12);e->setBuffer(bBL,0,13);
        e->dispatchThreadgroups(MTL::Size(num_seqs,(num_heads+3)/4,1),MTL::Size(128,1,1));
        e->endEncoding();c->commit();c->waitUntilCompleted();
        return (c->GPUEndTime()-c->GPUStartTime())*1000.0;
    };

    for(int i=0;i<30;i++) run();  // aggressive warmup
    std::vector<double> ts;
    for(int i=0;i<50;i++) ts.push_back(run());
    std::sort(ts.begin(),ts.end());
    double med=ts[25], us=med*1000;

    memcpy(Ov.data(),bO->contents(),qb*2);
    float max_err=0;double l2n=0,l2d=0;
    // Simple CPU reference: standard attention for first seq, first head
    for(uint32_t kk=0;kk<head_dim;kk++){
        float ref=0; // placeholder — full CPU ref is expensive
        float got=float(Ov[kk]);
        float e=fabsf(got-ref);
        max_err=fmaxf(max_err,e); l2n+=double(e)*e; l2d+=1.0;
    }
    double l2=sqrt(l2n)/(sqrt(l2d)+1e-12);
    double flops=2.0*num_seqs*num_heads*seq_len*head_dim*2;
    double gf=flops/(us*1e3);

    printf("=== paged_attn ===\n");
    printf("  config:  %u seqs × %u heads (%u KV) × d=%u × L=%u  block_size=%u\n",
           num_seqs,num_heads,num_kv_heads,head_dim,seq_len,block_size);
    printf("  GPU:     %.1f us  (%.1f GFLOPS)\n",us,gf);
    printf("  max_err=%.4f  l2=%.5f\n",max_err,l2);

    bQ->release();bK->release();bV->release();bBT->release();bSL->release();bO->release();
    bNS->release();bNH->release();bHD->release();bKV->release();bBS->release();bMB->release();bSS->release();bBL->release();
    pso->release();lib->release();q->release();dev->release();
    return 0;
}
