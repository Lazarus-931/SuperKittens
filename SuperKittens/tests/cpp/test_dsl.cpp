//
//  test_dsl.cpp — verify tile.h and mma.h generated kernels
//
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static float l2_rel(const std::vector<__fp16>& g, const std::vector<float>& r) {
    double n=0,d=0;
    for(size_t i=0;i<r.size();i++){float e=float(g[i])-r[i];n+=e*e;d+=r[i]*r[i];}
    return float(sqrt(n)/(sqrt(d)+1e-12));
}

static void fill(std::vector<__fp16>& v, std::mt19937& rng, float s) {
    std::normal_distribution<float> n(0,s);
    for(auto& x:v)x=__fp16(n(rng));
}

int main() {
    auto* dev=MTL::CreateSystemDefaultDevice(); auto* q=dev->newCommandQueue();
    auto* lib=dev->newLibrary(NS::URL::fileURLWithPath(
        NS::String::string("build/test_dsl.metallib",NS::UTF8StringEncoding)),nullptr);
    auto mode=MTL::ResourceStorageModeShared;

    printf("=== DSL tests ===\n\n");

    // ── Test 1: tile identity ──
    {
        printf("1. test_tile_identity: ");
        const uint32_t rows=4, cols=8; size_t n=rows*cols;
        std::mt19937 rng(42);
        std::vector<__fp16> src(n), dst(n);
        fill(src,rng,1.0f);

        auto *bS=dev->newBuffer(src.data(),n*sizeof(__fp16),mode);
        auto *bD=dev->newBuffer(n*sizeof(__fp16),mode);
        auto *bR=dev->newBuffer(&rows,sizeof(rows),mode);
        auto *bC=dev->newBuffer(&cols,sizeof(cols),mode);

        auto* fn=lib->newFunction(NS::String::string("test_tile_identity",NS::UTF8StringEncoding));
        auto* pso=dev->newComputePipelineState(fn,(NS::Error**)nullptr);fn->release();

        auto* cmd=q->commandBuffer(); auto* enc=cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(bS,0,0);enc->setBuffer(bD,0,1);enc->setBuffer(bR,0,2);enc->setBuffer(bC,0,3);
        enc->dispatchThreadgroups(MTL::Size(1,1,1),MTL::Size(64,1,1));
        enc->endEncoding();cmd->commit();cmd->waitUntilCompleted();

        std::vector<__fp16> out(n);
        memcpy(out.data(),bD->contents(),n*sizeof(__fp16));
        std::vector<float> ref(n);
        for(size_t i=0;i<n;i++)ref[i]=float(src[i]);
        float l2=l2_rel(out,ref);
        printf("l2_rel=%.6f %s\n",l2,l2<1e-6?"PASS":"FAIL");

        bS->release();bD->release();bR->release();bC->release();pso->release();
    }

    // ── Test 2: tile budget ──
    {
        printf("2. test_tile_budget: ");
        size_t n=1;
        std::vector<__fp16> dst(n);
        auto *bD=dev->newBuffer(n*sizeof(__fp16),mode);

        auto* fn=lib->newFunction(NS::String::string("test_tile_budget",NS::UTF8StringEncoding));
        auto* pso=dev->newComputePipelineState(fn,(NS::Error**)nullptr);fn->release();

        auto* cmd=q->commandBuffer(); auto* enc=cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(bD,0,0);
        enc->dispatchThreadgroups(MTL::Size(1,1,1),MTL::Size(128,1,1));
        enc->endEncoding();cmd->commit();cmd->waitUntilCompleted();

        float kb=float(*reinterpret_cast<__fp16*>(bD->contents()));
        printf("budget=%.1f KB %s\n",kb,kb>19&&kb<21?"PASS":"FAIL");

        bD->release();pso->release();
    }

    // ── Test 3: GEMM ──
    {
        printf("3. test_mma_gemm: ");
        const uint32_t M=64,N=64,K=128;
        size_t an=(size_t)M*K,bn=(size_t)K*N,cn=(size_t)M*N;
        std::mt19937 rng(42);
        std::vector<__fp16> A(an),B(bn),Cg(cn);
        fill(A,rng,0.5f);fill(B,rng,0.5f);

        // CPU reference
        std::vector<float> Cref(cn,0.0f);
        for(uint32_t i=0;i<M;i++)for(uint32_t j=0;j<N;j++){
            float acc=0;for(uint32_t k=0;k<K;k++)acc+=float(A[i*K+k])*float(B[k*N+j]);
            Cref[i*N+j]=acc;
        }

        auto *bA=dev->newBuffer(A.data(),an*sizeof(__fp16),mode);
        auto *bB=dev->newBuffer(B.data(),bn*sizeof(__fp16),mode);
        auto *bC=dev->newBuffer(cn*sizeof(__fp16),mode);
        auto *bM=dev->newBuffer(&M,sizeof(M),mode);
        auto *bN=dev->newBuffer(&N,sizeof(N),mode);
        auto *bK=dev->newBuffer(&K,sizeof(K),mode);

        auto* fn=lib->newFunction(NS::String::string("test_mma_gemm",NS::UTF8StringEncoding));
        auto* pso=dev->newComputePipelineState(fn,(NS::Error**)nullptr);fn->release();

        auto* cmd=q->commandBuffer(); auto* enc=cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(bA,0,0);enc->setBuffer(bB,0,1);enc->setBuffer(bC,0,2);
        enc->setBuffer(bM,0,3);enc->setBuffer(bN,0,4);enc->setBuffer(bK,0,5);
        enc->dispatchThreadgroups(MTL::Size(1,1,1),MTL::Size(128,1,1));
        enc->endEncoding();cmd->commit();cmd->waitUntilCompleted();

        memcpy(Cg.data(),bC->contents(),cn*sizeof(__fp16));
        float l2=l2_rel(Cg,Cref);
        printf("l2_rel=%.6f %s\n",l2,l2<0.01?"PASS":"FAIL");

        bA->release();bB->release();bC->release();bM->release();bN->release();bK->release();pso->release();
    }

    lib->release();q->release();dev->release();
    printf("\nDone.\n");
    return 0;
}
