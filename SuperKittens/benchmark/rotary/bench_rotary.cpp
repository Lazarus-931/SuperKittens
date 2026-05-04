// bench_rotary.cpp — RoPE kernel bench
// compile: clang++ -std=c++17 -O2 -I metal-cpp \
//   -framework Metal -framework Foundation -framework QuartzCore \
//   -o build/bench_rotary bench/bench_rotary.cpp
// run: ./build/bench_rotary [metallib=build/rotary.metallib]

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

static void fill_half(std::vector<__fp16>& v, std::mt19937& rng, float s = 1.f) {
    std::normal_distribution<float> n(0, 1);
    for (auto& x : v) x = __fp16(n(rng) * s);
}

static void cpu_rope(const std::vector<__fp16>& Q, const std::vector<__fp16>& K,
                     std::vector<float>& Q_out, std::vector<float>& K_out,
                     uint32_t heads, uint32_t seq, uint32_t d) {
    uint32_t hd = d / 2;
    size_t total = (size_t)heads * seq * d;
    Q_out.assign(total, 0.f); K_out.assign(total, 0.f);
    std::vector<float> freq(hd);
    for (uint32_t i = 0; i < hd; i++) freq[i] = 1.f / powf(10000.f, (2.f*i) / d);
    for (uint32_t h = 0; h < heads; h++) for (uint32_t s = 0; s < seq; s++) {
        size_t off = ((size_t)h * seq + s) * d;
        for (uint32_t i = 0; i < hd; i++) {
            float th = (float)s * freq[i];
            float c = float(__fp16(cosf(th))), sn = float(__fp16(sinf(th)));
            float q0 = float(Q[off+i]), q1 = float(Q[off+i+hd]);
            Q_out[off+i] = q0*c - q1*sn; Q_out[off+i+hd] = q0*sn + q1*c;
            float k0 = float(K[off+i]), k1 = float(K[off+i+hd]);
            K_out[off+i] = k0*c - k1*sn; K_out[off+i+hd] = k0*sn + k1*c;
        }
    }
}

static float l2_rel(const std::vector<__fp16>& g, const std::vector<float>& r) {
    double num = 0, den = 0;
    for (size_t i = 0; i < r.size(); i++) {
        double e = double(g[i]) - double(r[i]);
        num += e*e; den += double(r[i])*double(r[i]);
    }
    return float(sqrt(num) / (sqrt(den) + 1e-12));
}

int main(int argc, const char* argv[]) {
    const char* lib_path = (argc > 1) ? argv[1] : "build/rotary.metallib";

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
    auto* fn  = lib->newFunction(NS::String::string("rope_qk", NS::UTF8StringEncoding));
    auto* pso = dev->newComputePipelineState(fn, &err); fn->release();

    struct Cfg { uint32_t B, H, L, D; double mlx_us; };
    Cfg cfgs[] = {
        {1, 8,  512,  64,  541},
        {1, 8, 1024,  64,  806},
        {1, 8, 2048,  64, 1609},
        {1, 8,  512, 128,  920},
        {1, 8, 1024, 128, 1566},
        {1, 8, 2048, 128, 2770},
    };

    printf("%3s %3s %5s %4s  %10s %10s %8s  %10s %10s  %8s\n",
           "B","H","L","d","Metal(us)","MLX(us)","speedup","l2_Q","l2_K","GB/s");

    auto mode = MTL::ResourceStorageModeShared;
    for (auto& cfg : cfgs) {
        uint32_t hd = cfg.D / 2;
        size_t n = (size_t)cfg.B * cfg.H * cfg.L * cfg.D;
        size_t n_cos = (size_t)cfg.L * hd;

        std::mt19937 rng(42);
        std::vector<__fp16> Q(n), K(n), cos_buf(n_cos), sin_buf(n_cos);
        fill_half(Q, rng, 0.5f); fill_half(K, rng, 0.5f);

        std::vector<float> freq(hd);
        for (uint32_t i = 0; i < hd; i++) freq[i] = 1.f / powf(10000.f, (2.f*i) / cfg.D);
        for (uint32_t s = 0; s < cfg.L; s++) for (uint32_t i = 0; i < hd; i++) {
            float th = (float)s * freq[i];
            cos_buf[(size_t)s*hd+i] = __fp16(cosf(th));
            sin_buf[(size_t)s*hd+i] = __fp16(sinf(th));
        }

        std::vector<float> q_ref, k_ref;
        cpu_rope(Q, K, q_ref, k_ref, cfg.B * cfg.H, cfg.L, cfg.D);

        auto* bQ   = dev->newBuffer(Q.data(),       n*sizeof(__fp16),     mode);
        auto* bK   = dev->newBuffer(K.data(),       n*sizeof(__fp16),     mode);
        auto* bCos = dev->newBuffer(cos_buf.data(), n_cos*sizeof(__fp16), mode);
        auto* bSin = dev->newBuffer(sin_buf.data(), n_cos*sizeof(__fp16), mode);
        auto* bSeq = dev->newBuffer(&cfg.L,         sizeof(cfg.L),        mode);
        auto* bD   = dev->newBuffer(&cfg.D,         sizeof(cfg.D),        mode);
        uint32_t total_heads = cfg.B * cfg.H;
        auto* bH   = dev->newBuffer(&total_heads,   sizeof(total_heads),  mode);

        auto run = [&]() -> double {
            auto* cmd = queue->commandBuffer();
            auto* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(pso);
            enc->setBuffer(bQ,0,0); enc->setBuffer(bK,0,1); enc->setBuffer(bCos,0,2);
            enc->setBuffer(bSin,0,3); enc->setBuffer(bSeq,0,4);
            enc->setBuffer(bD,0,5); enc->setBuffer(bH,0,6);
            enc->dispatchThreadgroups(MTL::Size(cfg.B*cfg.H, 1, 1), MTL::Size(1024,1,1));
            enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted();
            return (cmd->GPUEndTime() - cmd->GPUStartTime()) * 1e6;
        };

        for (int w = 0; w < 5; w++) run();
        std::vector<double> ts;
        for (int i = 0; i < 20; i++) ts.push_back(run());
        std::sort(ts.begin(), ts.end());
        double metal_us = ts[ts.size()/2];

        // re-run to capture output for accuracy check
        memcpy(bQ->contents(), Q.data(), n*sizeof(__fp16));
        memcpy(bK->contents(), K.data(), n*sizeof(__fp16));
        run();
        std::vector<__fp16> q_gpu(n), k_gpu(n);
        memcpy(q_gpu.data(), bQ->contents(), n*sizeof(__fp16));
        memcpy(k_gpu.data(), bK->contents(), n*sizeof(__fp16));

        double gb = (8.0*cfg.B*cfg.H*cfg.L*cfg.D + 4.0*cfg.L*cfg.D) / 1e9;
        printf("%3u %3u %5u %4u  %10.1f %10.0f %7.2fx  %10.6f %10.6f  %7.1f\n",
               cfg.B, cfg.H, cfg.L, cfg.D,
               metal_us, cfg.mlx_us, cfg.mlx_us / metal_us,
               l2_rel(q_gpu, q_ref), l2_rel(k_gpu, k_ref),
               gb / (metal_us * 1e-6));

        bQ->release(); bK->release(); bCos->release(); bSin->release();
        bSeq->release(); bD->release(); bH->release();
    }

    pso->release(); lib->release(); queue->release(); dev->release();
    return 0;
}
