// bench_rms_residual.cpp — RMSNorm + Residual fusion
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static void fill_half(std::vector<__fp16>& v, std::mt19937& rng, float s=1.0f) {
    std::normal_distribution<float> n(0,1);
    for(auto& x:v)x=__fp16(n(rng)*s);
}

int main() {
    auto* dev=MTL::CreateSystemDefaultDevice(); auto* q=dev->newCommandQueue();
    auto* lib=dev->newLibrary(NS::URL::fileURLWithPath(
        NS::String::string("build/fusion.metallib",NS::UTF8StringEncoding)),nullptr);
    auto mode=MTL::ResourceStorageModeShared;
    auto* fn=lib->newFunction(NS::String::string("rms_residual",NS::UTF8StringEncoding));
    auto* pso=dev->newComputePipelineState(fn,(NS::Error**)nullptr);fn->release();

    printf("=== RMSNorm + Residual ===\n%6s %6s  %10s  %10s\n","rows","cols","us","GB/s");

    int configs[][2]={{512,1024},{1024,4096},{2048,4096},{4096,4096},{8192,4096}};
    for(int ci=0;ci<5;ci++){
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

        auto run=[&]()->double{
            auto* cmd=q->commandBuffer();auto* enc=cmd->computeCommandEncoder();
            enc->setComputePipelineState(pso);
            enc->setBuffer(bX,0,0);enc->setBuffer(bR,0,1);enc->setBuffer(bW,0,2);
            enc->setBuffer(bY,0,3);enc->setBuffer(bRows,0,4);enc->setBuffer(bCols,0,5);enc->setBuffer(bEps,0,6);
            enc->dispatchThreadgroups(MTL::Size(1,(r+7)/8,1),MTL::Size(256,1,1));
            enc->endEncoding();cmd->commit();cmd->waitUntilCompleted();
            return (cmd->GPUEndTime()-cmd->GPUStartTime())*1e6;
        };

        for(int w=0;w<5;w++)run();
        std::vector<double> ts;
        for(int i=0;i<20;i++)ts.push_back(run());
        std::sort(ts.begin(),ts.end());
        double us=ts[ts.size()/2], gb=(3.0*r*c*2)/1e9;
        printf("%6u %6u  %10.1f  %10.0f\n",r,c,us,gb/(us*1e-6));

        bX->release();bR->release();bW->release();bY->release();
        bRows->release();bCols->release();bEps->release();
    }
    pso->release();lib->release();q->release();dev->release();
    return 0;
}
