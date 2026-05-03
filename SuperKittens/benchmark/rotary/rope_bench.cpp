//
//  rope_bench.cpp — RoPE tiled Metal vs MLX
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
#include <random>
#include <vector>

static void fill_half(std::vector<__fp16>& v, std::mt19937& rng, float s=1.0f) {
    std::normal_distribution<float> n(0,1);
    for(auto& x:v)x=__fp16(n(rng)*s);
}

struct Stats { float l2_rel, max_abs; };
static Stats compare(const std::vector<__fp16>& g, const std::vector<float>& r) {
    Stats s{0,0}; double n=0,d=0;
    for(size_t i=0;i<r.size();i++){float e=float(g[i])-r[i];s.max_abs=fmaxf(s.max_abs,fabsf(e));n+=double(e)*e;d+=double(r[i])*r[i];}
    s.l2_rel=float(sqrt(n)/(sqrt(d)+1e-12)); return s;
}

static void cpu_rope(const std::vector<__fp16>& Q, const std::vector<__fp16>& K,
    std::vector<float>& Q_out, std::vector<float>& K_out,
    uint32_t heads, uint32_t seq, uint32_t d)
{
    uint32_t hd=d/2; size_t total=(size_t)heads*seq*d;
    Q_out.assign(total,0.0f); K_out.assign(total,0.0f);
    std::vector<float> freq(hd);
    for(uint32_t i=0;i<hd;i++) freq[i]=1.0f/powf(10000.0f,(2.0f*i)/d);
    for(uint32_t h=0;h<heads;h++)for(uint32_t s=0;s<seq;s++){
        size_t off=((size_t)h*seq+s)*d;
        for(uint32_t i=0;i<hd;i++){
            float th=(float)s*freq[i];
            float c=float(__fp16(cosf(th))),sn=float(__fp16(sinf(th)));
            float q0=float(Q[off+i]),q1=float(Q[off+i+hd]);
            Q_out[off+i]=q0*c-q1*sn; Q_out[off+i+hd]=q0*sn+q1*c;
            float k0=float(K[off+i]),k1=float(K[off+i+hd]);
            K_out[off+i]=k0*c-k1*sn; K_out[off+i+hd]=k0*sn+k1*c;
        }
    }
}

int main() {
    auto* dev=MTL::CreateSystemDefaultDevice(); auto* queue=dev->newCommandQueue();
    auto* lib=dev->newLibrary(NS::URL::fileURLWithPath(
        NS::String::string("build/rotary.metallib",NS::UTF8StringEncoding)),nullptr);
    auto mode=MTL::ResourceStorageModeShared;

    printf("=== RoPE: Metal vs MLX  (B=1, fp16) ===\n\n");
    printf("%3s %3s %5s %4s  %10s %10s %8s  %10s %10s  %8s\n",
           "B","H","L","d","Metal(us)","MLX(us)","speedup","l2_Q","l2_K","GB/s");

    int configs[][4]={{1,8,512,64},{1,8,1024,64},{1,8,2048,64},
                      {1,8,512,128},{1,8,1024,128},{1,8,2048,128}};
    double mlx_times[]={541,806,1609,920,1566,2770};

    for(int ci=0;ci<6;ci++){
        uint32_t B=configs[ci][0],H=configs[ci][1],L=configs[ci][2],D=configs[ci][3];
        uint32_t hd=D/2;
        size_t n=(size_t)B*H*L*D, n_cos=(size_t)L*hd;

        std::mt19937 rng(42);
        std::vector<__fp16> Q(n),K(n),cos_buf(n_cos),sin_buf(n_cos);
        fill_half(Q,rng,0.5f);fill_half(K,rng,0.5f);
        std::vector<float> freq(hd);
        for(uint32_t i=0;i<hd;i++)freq[i]=1.0f/powf(10000.0f,(2.0f*i)/D);
        for(uint32_t s=0;s<L;s++)for(uint32_t i=0;i<hd;i++){
            float th=(float)s*freq[i];
            cos_buf[(size_t)s*hd+i]=__fp16(cosf(th));
            sin_buf[(size_t)s*hd+i]=__fp16(sinf(th));
        }

        std::vector<float> q_ref,k_ref;
        cpu_rope(Q,K,q_ref,k_ref,B*H,L,D);

        auto *bQ=dev->newBuffer(Q.data(),n*sizeof(__fp16),mode);
        auto *bK=dev->newBuffer(K.data(),n*sizeof(__fp16),mode);
        auto *bCos=dev->newBuffer(cos_buf.data(),n_cos*sizeof(__fp16),mode);
        auto *bSin=dev->newBuffer(sin_buf.data(),n_cos*sizeof(__fp16),mode);
        auto *bSeq=dev->newBuffer(&L,sizeof(L),mode);
        auto *bD=dev->newBuffer(&D,sizeof(D),mode);
        uint32_t total_heads=B*H;
        auto *bH=dev->newBuffer(&total_heads,sizeof(total_heads),mode);

        const char* kname="rope_qk";
        auto* fn=lib->newFunction(NS::String::string(kname,NS::UTF8StringEncoding));
        auto* pso=dev->newComputePipelineState(fn,(NS::Error**)nullptr);fn->release();

        auto run=[&]()->double{
            auto* cmd=queue->commandBuffer();auto* enc=cmd->computeCommandEncoder();
            enc->setComputePipelineState(pso);
            enc->setBuffer(bQ,0,0);enc->setBuffer(bK,0,1);enc->setBuffer(bCos,0,2);
            enc->setBuffer(bSin,0,3);enc->setBuffer(bSeq,0,4);enc->setBuffer(bD,0,5);
            enc->setBuffer(bH,0,6);
            enc->dispatchThreadgroups(MTL::Size(B*H,1,1),MTL::Size(1024,1,1));
            enc->endEncoding();cmd->commit();cmd->waitUntilCompleted();
            return (cmd->GPUEndTime()-cmd->GPUStartTime())*1e6;
        };

        for(int w=0;w<5;w++)run();
        std::vector<double> ts;
        for(int i=0;i<20;i++)ts.push_back(run());
        std::sort(ts.begin(),ts.end());
        double metal_us=ts[ts.size()/2];

        memcpy(bQ->contents(),Q.data(),n*sizeof(__fp16));
        memcpy(bK->contents(),K.data(),n*sizeof(__fp16));
        run();
        std::vector<__fp16> q_gpu(n),k_gpu(n);
        memcpy(q_gpu.data(),bQ->contents(),n*sizeof(__fp16));
        memcpy(k_gpu.data(),bK->contents(),n*sizeof(__fp16));
        Stats sq=compare(q_gpu,q_ref),sk=compare(k_gpu,k_ref);

        // bytes: Q+K reads (2*BHLD*half) + Q+K writes (2*BHLD*half) + cos+sin reads (2*LD*half)
        double total_gb=(8.0*B*H*L*D + 4.0*L*D)/1e9;
        double gbps=total_gb/(metal_us*1e-6);
        double sp=mlx_times[ci]/metal_us;
        printf("%3u %3u %5u %4u  %10.1f %10.0f %7.2fx  %10.6f %10.6f  %7.1f\n",
               B,H,L,D,metal_us,mlx_times[ci],sp,sq.l2_rel,sk.l2_rel,gbps);

        bQ->release();bK->release();bCos->release();bSin->release();
        bSeq->release();bD->release();bH->release();pso->release();
    }
    lib->release();queue->release();dev->release();
    return 0;
}
