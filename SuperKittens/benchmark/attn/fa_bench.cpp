//
//  fa_bench.cpp — Flash Attention vs baseline MHA (causal + noncausal)
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
#include <random>
#include <vector>

static void fill_half(std::vector<__fp16>& v, std::mt19937& rng, float s=1.0f) {
    std::normal_distribution<float> n(0,1);
    for(auto& x:v)x=__fp16(n(rng)*s);
}

struct Stats { float max_abs, l2_rel; };

static Stats compare(const std::vector<__fp16>& g, const std::vector<float>& r) {
    Stats s{0,0}; double n=0,d=0;
    for(size_t i=0;i<r.size();i++){
        float e=float(g[i])-r[i];
        s.max_abs=fmaxf(s.max_abs,fabsf(e));
        n+=double(e)*e; d+=double(r[i])*r[i];
    }
    s.l2_rel=float(sqrt(n)/(sqrt(d)+1e-12));
    return s;
}

static void cpu_ref(const std::vector<__fp16>& Q, const std::vector<__fp16>& K,
    const std::vector<__fp16>& V, std::vector<float>& O,
    uint32_t seq, uint32_t d, bool causal)
{
    O.assign((size_t)seq*d,0.0f);
    std::vector<float> scores(seq);
    float scale=1.0f/sqrtf((float)d);
    for(uint32_t i=0;i<seq;i++){
        float row_max=-INFINITY;
        uint32_t limit=causal ? i+1 : seq;
        for(uint32_t j=0;j<limit;j++){
            float dot=0;for(uint32_t k=0;k<d;k++)dot+=float(Q[i*d+k])*float(K[j*d+k]);
            scores[j]=dot*scale; row_max=fmaxf(row_max,scores[j]);
        }
        float sum=0; for(uint32_t j=0;j<limit;j++){scores[j]=expf(scores[j]-row_max);sum+=scores[j];}
        float inv=1.0f/sum; for(uint32_t j=0;j<limit;j++)scores[j]*=inv;
        for(uint32_t k=0;k<d;k++){
            float acc=0; for(uint32_t j=0;j<limit;j++)acc+=scores[j]*float(V[j*d+k]);
            O[i*d+k]=acc;
        }
    }
}

struct BenchResult { double median_ms; Stats acc; };

static BenchResult run_kernel(
    MTL::Device* dev, MTL::CommandQueue* q, MTL::Library* lib,
    const char* kernel_name, bool is_fa,
    const std::vector<__fp16>& Q, const std::vector<__fp16>& K,
    const std::vector<__fp16>& V, const std::vector<float>& ref,
    uint32_t seq, uint32_t d, uint32_t heads)
{
    auto* fn=lib->newFunction(NS::String::string(kernel_name,NS::UTF8StringEncoding));
    if(!fn){printf("  FAIL kernel '%s'\n",kernel_name);return{0,{0,1}};}
    NS::Error* err=nullptr;
    auto* pso=dev->newComputePipelineState(fn,&err); fn->release();
    if(!pso){printf("  FAIL PSO '%s'\n",kernel_name);return{0,{0,1}};}

    size_t elems=(size_t)heads*seq*d;
    auto mode=MTL::ResourceStorageModeShared;
    auto *bQ=dev->newBuffer(Q.data(),elems*sizeof(__fp16),mode);
    auto *bK=dev->newBuffer(K.data(),elems*sizeof(__fp16),mode);
    auto *bV=dev->newBuffer(V.data(),elems*sizeof(__fp16),mode);
    auto *bO=dev->newBuffer(elems*sizeof(__fp16),mode);
    auto *bSeq=dev->newBuffer(&seq,sizeof(seq),mode);
    auto *bD=dev->newBuffer(&d,sizeof(d),mode);
    auto *bH=dev->newBuffer(&heads,sizeof(heads),mode);

    uint32_t grid_y = is_fa ? (seq+31)/32 : (seq+3)/4;
    uint32_t threads = is_fa ? 1024 : 128;

    auto run=[&]()->double{
        memset(bO->contents(),0,elems*sizeof(__fp16));
        auto* cmd=q->commandBuffer(); auto* enc=cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(bQ,0,0); enc->setBuffer(bK,0,1); enc->setBuffer(bV,0,2);
        enc->setBuffer(bO,0,3); enc->setBuffer(bSeq,0,4);
        enc->setBuffer(is_fa ? bH : bD,0,5);
        if(!is_fa) enc->setBuffer(bH,0,6);
        enc->dispatchThreadgroups(MTL::Size(heads,grid_y,1),MTL::Size(threads,1,1));
        enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted();
        return (cmd->GPUEndTime()-cmd->GPUStartTime())*1000.0;
    };

    for(int w=0;w<5;w++) run();
    std::vector<double> ts;
    for(int i=0;i<20;i++) ts.push_back(run());
    std::sort(ts.begin(),ts.end());

    std::vector<__fp16> gpu_out(seq*d);
    memcpy(gpu_out.data(),bO->contents(),seq*d*sizeof(__fp16));
    Stats st=compare(gpu_out,ref);

    bQ->release();bK->release();bV->release();bO->release();
    bSeq->release();bD->release();bH->release(); pso->release();
    return{ts[ts.size()/2],st};
}

int main() {
    auto* dev=MTL::CreateSystemDefaultDevice();
    auto* queue=dev->newCommandQueue();

    const char* d64_lib_path = "build/attn_d64.metallib";
    const char* d128_lib_path = "build/attn_d128.metallib";

    auto* d64_lib=dev->newLibrary(NS::URL::fileURLWithPath(
        NS::String::string(d64_lib_path,NS::UTF8StringEncoding)),nullptr);
    auto* d128_lib=dev->newLibrary(NS::URL::fileURLWithPath(
        NS::String::string(d128_lib_path,NS::UTF8StringEncoding)),nullptr);

    const uint32_t heads=1;
    printf("=== FA vs MHA  (B=1, H=1) ===\n\n");

    auto bench_pair = [&](int d, bool causal, const char* label) {
        bool is_fa = (d == 64);
        auto* lib = is_fa ? d64_lib : d128_lib;
        const char* kname = is_fa
            ? (causal ? "fa_causal_64" : "fa_noncausal_64")
            : (causal ? "mha_causal"   : "mha_noncausal");

        printf("-- %s  d=%d  kernel=%s --\n", label, d, kname);
        printf("%6s  %10s  %10s\n", "L", "time(ms)", "l2_rel");
        for(int li=0;li<4;li++){
            uint32_t Ls[]={256,512,1024,2048};
            uint32_t L=Ls[li];
            size_t n=(size_t)heads*L*d;
            std::mt19937 rng(42);
            std::vector<__fp16> Q(n),K(n),V(n);
            fill_half(Q,rng,0.5f);fill_half(K,rng,0.5f);fill_half(V,rng,0.5f);
            std::vector<float> ref;
            cpu_ref(Q,K,V,ref,L,d,causal);

            auto r=run_kernel(dev,queue,lib,kname,is_fa,Q,K,V,ref,L,d,heads);
            printf("%6u  %10.4f  %10.6f\n", L, r.median_ms, r.acc.l2_rel);
        }
        printf("\n");
    };

    bench_pair(64, true,  "causal");
    bench_pair(128, true,  "causal");
    bench_pair(64, false, "noncausal");
    bench_pair(128, false, "noncausal");

    d64_lib->release(); d128_lib->release();
    queue->release(); dev->release();
    return 0;
}
