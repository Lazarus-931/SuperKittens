//
//  fusion_bench.cpp — Fusion kernels benchmark
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
#include <functional>
#include <random>
#include <vector>

static void fill_half(std::vector<__fp16>& v, std::mt19937& rng, float s=1.0f) {
    std::normal_distribution<float> n(0,1);
    for(auto& x:v)x=__fp16(n(rng)*s);
}

static double bench_kernel(
    MTL::Device* dev, MTL::CommandQueue* q, MTL::Library* lib,
    const char* name, std::function<void(MTL::ComputeCommandEncoder*)> bind,
    uint32_t grid_x, uint32_t grid_y, uint32_t threads)
{
    auto* fn=lib->newFunction(NS::String::string(name,NS::UTF8StringEncoding));
    auto* pso=dev->newComputePipelineState(fn,(NS::Error**)nullptr);fn->release();
    if(!pso)return-1;

    auto run=[&]()->double{
        auto* cmd=q->commandBuffer();auto* enc=cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        bind(enc);
        enc->dispatchThreadgroups(MTL::Size(grid_x,grid_y,1),MTL::Size(threads,1,1));
        enc->endEncoding();cmd->commit();cmd->waitUntilCompleted();
        return (cmd->GPUEndTime()-cmd->GPUStartTime())*1e6;
    };

    for(int w=0;w<5;w++)run();
    std::vector<double> ts;
    for(int i=0;i<20;i++)ts.push_back(run());
    std::sort(ts.begin(),ts.end());
    pso->release();
    return ts[ts.size()/2];
}

int main() {
    auto* dev=MTL::CreateSystemDefaultDevice(); auto* q=dev->newCommandQueue();
    auto* lib=dev->newLibrary(NS::URL::fileURLWithPath(
        NS::String::string("build/fusion.metallib",NS::UTF8StringEncoding)),nullptr);
    auto mode=MTL::ResourceStorageModeShared;

    // ── 1. RMSNorm + Residual ──
    printf("=== 1. rms_residual ===\n%6s %6s  %10s  %10s\n","rows","cols","us","GB/s");
    for(int ci=0;ci<4;ci++){
        int configs[][2]={{512,1024},{1024,4096},{2048,4096},{4096,4096}};
        uint32_t r=configs[ci][0],c=configs[ci][1]; size_t n=(size_t)r*c;
        std::mt19937 rng(42);
        std::vector<__fp16> x(n),res(n),wt(c);
        fill_half(x,rng,0.5f);fill_half(res,rng,0.2f);fill_half(wt,rng,0.5f);
        auto *bX=dev->newBuffer(x.data(),n*sizeof(__fp16),mode);
        auto *bR=dev->newBuffer(res.data(),n*sizeof(__fp16),mode);
        auto *bW=dev->newBuffer(wt.data(),c*sizeof(__fp16),mode);
        auto *bY=dev->newBuffer(n*sizeof(__fp16),mode);
        auto *bRows=dev->newBuffer(&r,sizeof(r),mode);
        auto *bCols=dev->newBuffer(&c,sizeof(c),mode);
        float eps=1e-5f; auto *bEps=dev->newBuffer(&eps,sizeof(eps),mode);

        double us=bench_kernel(dev,q,lib,"rms_residual",[&](auto* enc){
            enc->setBuffer(bX,0,0);enc->setBuffer(bR,0,1);enc->setBuffer(bW,0,2);
            enc->setBuffer(bY,0,3);enc->setBuffer(bRows,0,4);enc->setBuffer(bCols,0,5);enc->setBuffer(bEps,0,6);
        },1,(r+7)/8,256);
        double gb=(3.0*r*c*2)/1e9;
        printf("%6u %6u  %10.1f  %10.0f\n",r,c,us,gb/(us*1e-6));

        bX->release();bR->release();bW->release();bY->release();
        bRows->release();bCols->release();bEps->release();
    }

    // ── 2. GEMM + bias + act ──
    printf("\n=== 2. gemm_bias_act (SiLU) ===\n%6s %6s %6s  %10s  %10s\n","M","N","K","us","GFLOPS");
    for(int ci=0;ci<4;ci++){
        int configs[][3]={{512,1024,512},{1024,1024,1024},{2048,1024,1024},{4096,1024,4096}};
        uint32_t M=configs[ci][0],N=configs[ci][1],K=configs[ci][2];
        size_t an=M*K,bn=K*N;
        std::mt19937 rng(42);
        std::vector<__fp16> A(an),B(bn),bias(N);
        fill_half(A,rng,0.25f);fill_half(B,rng,0.15f);fill_half(bias,rng,0.05f);
        auto *bA=dev->newBuffer(A.data(),an*sizeof(__fp16),mode);
        auto *bB=dev->newBuffer(B.data(),bn*sizeof(__fp16),mode);
        auto *bBias=dev->newBuffer(bias.data(),N*sizeof(__fp16),mode);
        auto *bC=dev->newBuffer(M*N*sizeof(__fp16),mode);
        auto *bM=dev->newBuffer(&M,sizeof(M),mode);
        auto *bN=dev->newBuffer(&N,sizeof(N),mode);
        auto *bK=dev->newBuffer(&K,sizeof(K),mode);
        uint32_t act=2; auto *bAct=dev->newBuffer(&act,sizeof(act),mode);

        double us=bench_kernel(dev,q,lib,"gemm_bias_act",[&](auto* enc){
            enc->setBuffer(bA,0,0);enc->setBuffer(bB,0,1);enc->setBuffer(bBias,0,2);
            enc->setBuffer(bC,0,3);enc->setBuffer(bM,0,4);enc->setBuffer(bN,0,5);
            enc->setBuffer(bK,0,6);enc->setBuffer(bAct,0,7);
        },(N+63)/64,(M+63)/64,128);
        double gflops=(2.0*M*N*K)/(us*1e3);
        printf("%6u %6u %6u  %10.1f  %10.0f\n",M,N,K,us,gflops);

        bA->release();bB->release();bBias->release();bC->release();
        bM->release();bN->release();bK->release();bAct->release();
    }

    // ── 3. RMSNorm + RoPE ──
    printf("\n=== 3. rms_rope ===\n%6s %6s %5s %5s  %10s  %10s\n","heads","seq","d","hd","us","GB/s");
    for(int ci=0;ci<4;ci++){
        int configs[][3]={{8,512,64},{8,1024,64},{8,2048,64},{8,2048,128}};
        uint32_t H=configs[ci][0],L=configs[ci][1],D=configs[ci][2],hd=D/2;
        size_t n=(size_t)H*L*D,n_cos=(size_t)L*hd;
        std::mt19937 rng(42);
        std::vector<__fp16> Q(n),K(n),qw(D),kw(D),cos_buf(n_cos),sin_buf(n_cos);
        fill_half(Q,rng,0.5f);fill_half(K,rng,0.5f);fill_half(qw,rng,0.5f);fill_half(kw,rng,0.5f);
        std::vector<float> freq(hd);
        for(uint32_t i=0;i<hd;i++)freq[i]=1.0f/powf(10000.0f,(2.0f*i)/D);
        for(uint32_t s=0;s<L;s++)for(uint32_t i=0;i<hd;i++){
            float th=(float)s*freq[i];
            cos_buf[(size_t)s*hd+i]=__fp16(cosf(th));sin_buf[(size_t)s*hd+i]=__fp16(sinf(th));
        }
        auto *bQ=dev->newBuffer(Q.data(),n*sizeof(__fp16),mode);
        auto *bK=dev->newBuffer(K.data(),n*sizeof(__fp16),mode);
        auto *bQW=dev->newBuffer(qw.data(),D*sizeof(__fp16),mode);
        auto *bKW=dev->newBuffer(kw.data(),D*sizeof(__fp16),mode);
        auto *bC=dev->newBuffer(cos_buf.data(),n_cos*sizeof(__fp16),mode);
        auto *bS=dev->newBuffer(sin_buf.data(),n_cos*sizeof(__fp16),mode);
        auto *bQO=dev->newBuffer(n*sizeof(__fp16),mode);
        auto *bKO=dev->newBuffer(n*sizeof(__fp16),mode);
        auto *bH=dev->newBuffer(&H,sizeof(H),mode);auto *bL=dev->newBuffer(&L,sizeof(L),mode);
        auto *bD=dev->newBuffer(&D,sizeof(D),mode);
        float eps=1e-5f;auto *bEps=dev->newBuffer(&eps,sizeof(eps),mode);

        double us=bench_kernel(dev,q,lib,"rms_rope",[&](auto* enc){
            enc->setBuffer(bQ,0,0);enc->setBuffer(bK,0,1);enc->setBuffer(bQW,0,2);
            enc->setBuffer(bKW,0,3);enc->setBuffer(bC,0,4);enc->setBuffer(bS,0,5);
            enc->setBuffer(bQO,0,6);enc->setBuffer(bKO,0,7);enc->setBuffer(bH,0,8);
            enc->setBuffer(bL,0,9);enc->setBuffer(bD,0,10);enc->setBuffer(bEps,0,11);
        },H,(L+7)/8,256);
        double total_gb=(4.0*H*L*D + 2.0*L*D)/1e9;  // Q/K read+write + cos/sin
        printf("%6u %6u %5u %5u  %10.1f  %10.0f\n",H,L,D,hd,us,total_gb/(us*1e-6));

        bQ->release();bK->release();bQW->release();bKW->release();
        bC->release();bS->release();bQO->release();bKO->release();
        bH->release();bL->release();bD->release();bEps->release();
    }

    lib->release();q->release();dev->release();
    printf("\nDone.\n");
    return 0;
}
