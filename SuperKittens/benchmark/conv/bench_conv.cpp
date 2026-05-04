// bench_conv.cpp — conv1d / conv2d (tiled + im2col) kernel bench
// compile: clang++ -std=c++17 -O2 -I metal-cpp \
//   -framework Metal -framework Foundation -framework QuartzCore \
//   -o build/bench_conv bench/bench_conv.cpp
// run: ./build/bench_conv [metallib=build/conv2d.metallib]

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// ── shared helpers ────────────────────────────────────────────────────────────

static void fill_half(std::vector<__fp16>& v, std::mt19937& rng, float scale) {
    std::normal_distribution<float> n(0.f, 1.f);
    for (auto& x : v) x = __fp16(n(rng) * scale);
}

struct Shape { int N, H, W, C_in, K, C_out, H_out, W_out; };
static Shape make_shape(int H, int W, int C_in, int C_out, int K) {
    return {1, H, W, C_in, K, C_out, H-K+1, W-K+1};
}

static void cpu_conv2d(const std::vector<__fp16>& x, const std::vector<__fp16>& w,
                       const std::vector<__fp16>& bias, std::vector<float>& y, const Shape& s) {
    y.assign((size_t)s.N * s.H_out * s.W_out * s.C_out, 0.f);
    for (int n=0; n<s.N; n++) for (int h=0; h<s.H_out; h++) for (int wi=0; wi<s.W_out; wi++)
        for (int co=0; co<s.C_out; co++) {
            float acc = float(bias[co]);
            for (int kh=0; kh<s.K; kh++) for (int kw=0; kw<s.K; kw++) {
                int hi=h+kh, wj=wi+kw;
                if (hi<s.H && wj<s.W)
                    for (int ci=0; ci<s.C_in; ci++)
                        acc += float(x[((n*s.H+hi)*s.W+wj)*s.C_in+ci]) *
                               float(w[((kh*s.K+kw)*s.C_in+ci)*s.C_out+co]);
            }
            y[((n*s.H_out+h)*s.W_out+wi)*s.C_out+co] = acc;
        }
}

struct Err { float max_abs, l2_rel; };
static Err compare(const std::vector<__fp16>& gpu, const std::vector<float>& ref) {
    Err e{0,0}; double num=0, den=0;
    for (size_t i=0; i<ref.size(); i++) {
        float d = fabsf(float(gpu[i]) - ref[i]);
        e.max_abs = fmaxf(e.max_abs, d);
        num += double(d)*d; den += double(ref[i])*double(ref[i]);
    }
    e.l2_rel = float(sqrt(num) / (sqrt(den) + 1e-12));
    return e;
}

// ── conv2d tiled ──────────────────────────────────────────────────────────────

static void bench_conv2d_tiled(MTL::Device* dev, MTL::CommandQueue* queue,
                               MTL::Library* lib, const Shape& s) {
    auto* fn = lib->newFunction(NS::String::string("conv2d_tiled", NS::UTF8StringEncoding));
    if (!fn) fn = lib->newFunction(NS::String::string("conv2d", NS::UTF8StringEncoding));
    if (!fn) { printf("  [tiled] kernel not found\n"); return; }
    NS::Error* err = nullptr;
    auto* pso = dev->newComputePipelineState(fn, &err); fn->release();

    std::mt19937 rng(42);
    std::vector<__fp16> xv(s.N*s.H*s.W*s.C_in), wv(s.K*s.K*s.C_in*s.C_out), bv(s.C_out);
    fill_half(xv, rng, 0.25f); fill_half(wv, rng, 0.15f); fill_half(bv, rng, 0.05f);

    std::vector<float> ref;
    auto t0 = std::chrono::steady_clock::now();
    cpu_conv2d(xv, wv, bv, ref, s);
    double cpu_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();

    auto mode = MTL::ResourceStorageModeShared;
    auto* bX = dev->newBuffer(xv.data(), xv.size()*sizeof(__fp16), mode);
    auto* bW = dev->newBuffer(wv.data(), wv.size()*sizeof(__fp16), mode);
    auto* bB = dev->newBuffer(bv.data(), bv.size()*sizeof(__fp16), mode);
    auto* bY = dev->newBuffer((size_t)s.N*s.H_out*s.W_out*s.C_out*sizeof(__fp16), mode);

    struct Params { uint32_t N,H,W,C_in,K_H,K_W,C_out,H_out,W_out; };
    Params p = {(uint32_t)s.N,(uint32_t)s.H,(uint32_t)s.W,(uint32_t)s.C_in,
                (uint32_t)s.K,(uint32_t)s.K,(uint32_t)s.C_out,(uint32_t)s.H_out,(uint32_t)s.W_out};
    auto* bP = dev->newBuffer(&p, sizeof(p), mode);

    uint32_t gx = (s.W_out+7)/8, gy = (s.H_out+3)/4;
    auto run = [&]() -> double {
        auto* cmd = queue->commandBuffer(); auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(bX,0,0); enc->setBuffer(bW,0,1); enc->setBuffer(bB,0,2);
        enc->setBuffer(bY,0,3); enc->setBuffer(bP,0,4);
        enc->dispatchThreadgroups(MTL::Size(gx,gy,s.N), MTL::Size(64,1,1));
        enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted();
        return (cmd->GPUEndTime()-cmd->GPUStartTime())*1000.0;
    };

    for (int w=0; w<5; w++) run();
    std::vector<double> ts; for (int i=0; i<20; i++) ts.push_back(run());
    std::sort(ts.begin(), ts.end());
    double gpu_ms = ts[ts.size()/2];

    std::vector<__fp16> gpu_y(s.N*s.H_out*s.W_out*s.C_out);
    memcpy(gpu_y.data(), bY->contents(), gpu_y.size()*sizeof(__fp16));
    auto st = compare(gpu_y, ref);

    double flops = 2.0*s.N*s.H_out*s.W_out*s.K*s.K*s.C_in*s.C_out;
    printf("  tiled    N=%d H=%d W=%d C_in=%d K=%d C_out=%d  CPU=%.1fms  GPU=%.1fus  %.1f GFLOPS  l2=%.5f  %s\n",
           s.N,s.H,s.W,s.C_in,s.K,s.C_out, cpu_ms, gpu_ms*1000.0,
           flops/(gpu_ms*1e6), st.l2_rel, st.l2_rel < 0.05f ? "PASS" : "FAIL");

    bX->release(); bW->release(); bB->release(); bY->release(); bP->release();
    pso->release();
}

// ── conv2d im2col + GEMM ──────────────────────────────────────────────────────

static void bench_conv2d_im2col(MTL::Device* dev, MTL::CommandQueue* queue,
                                MTL::Library* lib, const Shape& s) {
    auto* im2col_fn = lib->newFunction(NS::String::string("im2col_fp16", NS::UTF8StringEncoding));
    if (!im2col_fn) im2col_fn = lib->newFunction(NS::String::string("conv2d_im2col", NS::UTF8StringEncoding));
    auto* gemm_fn   = lib->newFunction(NS::String::string("gemm_fp16",   NS::UTF8StringEncoding));
    if (!im2col_fn || !gemm_fn) { printf("  [im2col] kernels not found\n"); if(im2col_fn) im2col_fn->release(); if(gemm_fn) gemm_fn->release(); return; }
    NS::Error* err = nullptr;
    auto* im2col_pso = dev->newComputePipelineState(im2col_fn, &err); im2col_fn->release();
    auto* gemm_pso   = dev->newComputePipelineState(gemm_fn,   &err); gemm_fn->release();

    uint32_t M = s.N*s.H_out*s.W_out, Kt = s.K*s.K*s.C_in, N = s.C_out;

    std::mt19937 rng(42);
    std::vector<__fp16> xv(s.N*s.H*s.W*s.C_in), wv(Kt*N), bv(N);
    fill_half(xv, rng, 0.25f); fill_half(wv, rng, 0.15f); fill_half(bv, rng, 0.05f);
    std::vector<float> ref; cpu_conv2d(xv, wv, bv, ref, s);

    auto mode = MTL::ResourceStorageModeShared;
    auto* bX   = dev->newBuffer(xv.data(), xv.size()*sizeof(__fp16), mode);
    auto* bW   = dev->newBuffer(wv.data(), wv.size()*sizeof(__fp16), mode);
    auto* bB   = dev->newBuffer(bv.data(), bv.size()*sizeof(__fp16), mode);
    auto* bCol = dev->newBuffer((size_t)M*Kt*sizeof(__fp16), mode);
    auto* bY   = dev->newBuffer((size_t)M*N*sizeof(__fp16),  mode);

    auto mkU = [&](uint32_t v) { return dev->newBuffer(&v, sizeof(v), mode); };
    auto *bN_=mkU(s.N),*bH_=mkU(s.H),*bW_=mkU(s.W),*bCi=mkU(s.C_in);
    auto *bKH=mkU(s.K),*bKW=mkU(s.K),*bHo=mkU(s.H_out),*bWo=mkU(s.W_out);
    auto *bgM=mkU(M),*bgN=mkU(N),*bgK=mkU(Kt),*bgLA=mkU(Kt),*bgLB=mkU(N),*bgLC=mkU(N);
    bool f=false,t=true;
    auto* bgTA=dev->newBuffer(&f,1,mode); auto* bgTB=dev->newBuffer(&f,1,mode);
    auto* bgHB=dev->newBuffer(&t,1,mode);

    uint32_t gx_im=(s.W_out+7)/8, gy_im=(s.H_out+7)/8;
    uint32_t gx_gm=(N+63)/64, gy_gm=(M+31)/32;

    auto run = [&](double& im_ms, double& ge_ms) {
        { auto* cmd=queue->commandBuffer(); auto* enc=cmd->computeCommandEncoder();
          enc->setComputePipelineState(im2col_pso);
          enc->setBuffer(bX,0,0);enc->setBuffer(bCol,0,1);enc->setBuffer(bN_,0,2);enc->setBuffer(bH_,0,3);
          enc->setBuffer(bW_,0,4);enc->setBuffer(bCi,0,5);enc->setBuffer(bKH,0,6);enc->setBuffer(bKW,0,7);
          enc->setBuffer(bHo,0,8);enc->setBuffer(bWo,0,9);
          enc->dispatchThreadgroups(MTL::Size(gx_im,gy_im,1),MTL::Size(64,1,1));
          enc->endEncoding();cmd->commit();cmd->waitUntilCompleted();
          im_ms=(cmd->GPUEndTime()-cmd->GPUStartTime())*1000.0; }
        { auto* cmd=queue->commandBuffer(); auto* enc=cmd->computeCommandEncoder();
          enc->setComputePipelineState(gemm_pso);
          enc->setBuffer(bCol,0,0);enc->setBuffer(bW,0,1);enc->setBuffer(bY,0,2);
          enc->setBuffer(bgM,0,3);enc->setBuffer(bgN,0,4);enc->setBuffer(bgK,0,5);
          enc->setBuffer(bgLA,0,6);enc->setBuffer(bgLB,0,7);enc->setBuffer(bgLC,0,8);
          enc->setBuffer(bgTA,0,9);enc->setBuffer(bgTB,0,10);enc->setBuffer(bgHB,0,11);enc->setBuffer(bB,0,12);
          enc->dispatchThreadgroups(MTL::Size(gx_gm,gy_gm,1),MTL::Size(64,1,1));
          enc->endEncoding();cmd->commit();cmd->waitUntilCompleted();
          ge_ms=(cmd->GPUEndTime()-cmd->GPUStartTime())*1000.0; }
    };

    for (int w=0; w<5; w++) { double a,b; run(a,b); }
    std::vector<double> ti,tg,tt;
    for (int i=0; i<20; i++) { double a,b; run(a,b); ti.push_back(a); tg.push_back(b); tt.push_back(a+b); }
    std::sort(ti.begin(),ti.end()); std::sort(tg.begin(),tg.end()); std::sort(tt.begin(),tt.end());
    double mi=ti[10],mg=tg[10],mt=tt[10];

    std::vector<__fp16> gpu_y(M*N);
    memcpy(gpu_y.data(), bY->contents(), M*N*sizeof(__fp16));
    auto st = compare(gpu_y, ref);

    double flops = 2.0*s.N*s.H_out*s.W_out*s.K*s.K*s.C_in*s.C_out;
    printf("  im2col   im2col=%.1fus  gemm=%.1fus  total=%.1fus  %.1f GFLOPS  l2=%.5f  %s\n",
           mi*1000, mg*1000, mt*1000, flops/(mt*1e6), st.l2_rel, st.l2_rel < 0.05f ? "PASS" : "FAIL");

    bX->release();bW->release();bB->release();bCol->release();bY->release();
    bN_->release();bH_->release();bW_->release();bCi->release();bKH->release();bKW->release();bHo->release();bWo->release();
    bgM->release();bgN->release();bgK->release();bgLA->release();bgLB->release();bgLC->release();
    bgTA->release();bgTB->release();bgHB->release();
    im2col_pso->release(); gemm_pso->release();
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, const char* argv[]) {
    const char* lib_path = (argc > 1) ? argv[1] : "build/conv2d.metallib";

    auto* dev   = MTL::CreateSystemDefaultDevice();
    auto* queue = dev->newCommandQueue();

    NS::Error* err = nullptr;
    auto* lib = dev->newLibrary(
        NS::URL::fileURLWithPath(NS::String::string(lib_path, NS::UTF8StringEncoding)), &err);
    if (!lib) {
        fprintf(stderr, "load metallib '%s': %s\n", lib_path,
                err ? err->localizedDescription()->utf8String() : "?");
        return 1;
    }

    struct Cfg { int H, W, C_in, C_out, K; };
    Cfg cfgs[] = {
        {56,  56,  64,  64,  3},
        {28,  28, 128, 128,  3},
        {14,  14, 256, 256,  3},
        {56,  56,  64, 128,  1},
    };

    printf("=== conv2d ===\n");
    for (auto& cfg : cfgs) {
        Shape s = make_shape(cfg.H, cfg.W, cfg.C_in, cfg.C_out, cfg.K);
        printf("shape N=1 H=%d W=%d C_in=%d K=%d C_out=%d:\n",
               cfg.H, cfg.W, cfg.C_in, cfg.K, cfg.C_out);
        bench_conv2d_tiled(dev, queue, lib, s);
        bench_conv2d_im2col(dev, queue, lib, s);
    }

    lib->release(); queue->release(); dev->release();
    return 0;
}
