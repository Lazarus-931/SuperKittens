#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {
struct S { int N,H,W,CI,KH,KW,CO,HO,WO; };
static S s;
static void fh(std::vector<__fp16>& v, std::mt19937& r, float sc) {
    std::normal_distribution<float> n(0,1);
    for(auto& x:v) x=__fp16(n(r)*sc);
}
static void cpu(const std::vector<__fp16>& x, const std::vector<__fp16>& w,
    const std::vector<__fp16>& b, std::vector<float>& y) {
    y.assign(s.N*s.HO*s.WO*s.CO,0);
    for(int n=0;n<s.N;n++) for(int h=0;h<s.HO;h++) for(int wi=0;wi<s.WO;wi++)
        for(int co=0;co<s.CO;co++) { float a=float(b[co]);
            for(int kh=0;kh<s.KH;kh++) for(int kw=0;kw<s.KW;kw++) {
                int hi=h+kh,wj=wi+kw; if(hi<s.H&&wj<s.W) for(int ci=0;ci<s.CI;ci++)
                    a+=float(x[((n*s.H+hi)*s.W+wj)*s.CI+ci])*float(w[((kh*s.KW+kw)*s.CI+ci)*s.CO+co]);
            } y[((n*s.HO+h)*s.WO+wi)*s.CO+co]=a;
        }
}
struct E { float ma,me,l2; };
static E cmp(const std::vector<__fp16>& g, const std::vector<float>& r) {
    E e{0,0,0}; double sa=0,sn=0,sd=0;
    for(size_t i=0;i<r.size();i++){float d=fabsf(float(g[i])-r[i]);e.ma=fmaxf(e.ma,d);sa+=d;sn+=double(d)*d;sd+=double(r[i])*r[i];}
    e.me=float(sa/r.size()); e.l2=float(sqrt(sn)/(sqrt(sd)+1e-12)); return e;
}
}

int main(int argc, const char** argv) {
    if(argc<2){fprintf(stderr,"usage: %s <metallib>\n",argv[0]);return 1;}
    s={1,56,56,64,3,3,64,54,54};
    auto mk=[&](const void* d, size_t sz){auto* b=MTL::CreateSystemDefaultDevice()->newBuffer(sz,MTL::ResourceStorageModeShared);if(d)memcpy(b->contents(),d,sz);return b;};
    auto mu=[&](uint32_t v){return mk(&v,sizeof(uint32_t));};

    MTL::Device* dev=MTL::CreateSystemDefaultDevice();
    MTL::CommandQueue* q=dev->newCommandQueue();
    auto* url=NS::URL::fileURLWithPath(NS::String::string(argv[1],NS::UTF8StringEncoding));
    MTL::Library* lib=dev->newLibrary(url,nullptr);
    auto* fn=lib->newFunction(NS::String::string("conv2d_tiled",NS::UTF8StringEncoding));
    NS::Error* err=nullptr; auto* pso=dev->newComputePipelineState(fn,&err); fn->release();

    std::mt19937 rng(42);
    std::vector<__fp16> xv(s.N*s.H*s.W*s.CI),wv(s.KH*s.KW*s.CI*s.CO),bv(s.CO),yg(s.N*s.HO*s.WO*s.CO);
    fh(xv,rng,0.25f); fh(wv,rng,0.15f); fh(bv,rng,0.05f);
    std::vector<float> yr;
    auto t0=std::chrono::steady_clock::now(); cpu(xv,wv,bv,yr);
    auto t1=std::chrono::steady_clock::now();
    double cpu_ms=std::chrono::duration<double,std::milli>(t1-t0).count();

    auto *bX=mk(xv.data(),xv.size()*2),*bW=mk(wv.data(),wv.size()*2);
    auto *bB=mk(bv.data(),bv.size()*2),*bY=mk(0,s.N*s.HO*s.WO*s.CO*2);
    auto *bN=mu(s.N),*bH=mu(s.H),*bW1=mu(s.W),*bCi=mu(s.CI);
    auto *bKH=mu(s.KH),*bKW=mu(s.KW),*bCo=mu(s.CO),*bHo=mu(s.HO),*bWo=mu(s.WO);

    uint gx=(s.WO+7)/8,gy=(s.HO+7)/8;
    auto run=[&](){
        auto* c=q->commandBuffer(); auto* e=c->computeCommandEncoder();
        e->setComputePipelineState(pso);
        e->setBuffer(bX,0,0); e->setBuffer(bW,0,1); e->setBuffer(bB,0,2); e->setBuffer(bY,0,3);
        e->setBuffer(bN,0,4); e->setBuffer(bH,0,5); e->setBuffer(bW1,0,6); e->setBuffer(bCi,0,7);
        e->setBuffer(bKH,0,8); e->setBuffer(bKW,0,9); e->setBuffer(bCo,0,10); e->setBuffer(bHo,0,11); e->setBuffer(bWo,0,12);
        e->dispatchThreadgroups(MTL::Size(gx,gy,s.N),MTL::Size(128,1,1));
        e->endEncoding(); c->commit(); c->waitUntilCompleted();
        return (c->GPUEndTime()-c->GPUStartTime())*1000.0;
    };

    for(int i=0;i<5;i++) run();
    std::vector<double> ts; for(int i=0;i<20;i++) ts.push_back(run());
    std::sort(ts.begin(),ts.end());
    double gpu_ms=ts[10],gpu_us=gpu_ms*1000;

    memcpy(yg.data(),bY->contents(),yg.size()*2);
    E st=cmp(yg,yr);
    double flops=2.0*s.N*s.HO*s.WO*s.KH*s.KW*s.CI*s.CO;
    double gf=flops/(gpu_us*1e3);

    printf("== conv2d tiled ==\n");
    printf("  GPU:  %.1f us  (%.1f GFLOPS)\n",gpu_us,gf);
    printf("  acc:  max_abs=%.4f  l2_rel=%.5f\n",st.ma,st.l2);
    printf("  %s\n",(std::isfinite(st.l2)&&st.l2<5e-2f)?"PASS":"FAIL");
    return (std::isfinite(st.l2)&&st.l2<5e-2f)?0:1;
}
