// bench_attn.cpp — unified attn kernel bench (d=64 and d=128, causal + noncausal)
// compile: clang++ -std=c++17 -O2 -I metal-cpp \
//   -framework Metal -framework Foundation -framework QuartzCore \
//   -o build/bench_attn bench/bench_attn.cpp
// run: ./build/bench_attn [metallib=build/attn.metallib]

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static void fill(std::vector<__fp16>& v, std::mt19937& rng) {
    std::normal_distribution<float> d(0.f, 0.5f);
    for (auto& x : v) x = __fp16(d(rng));
}

static void cpu_ref(const std::vector<__fp16>& Q, const std::vector<__fp16>& K,
                    const std::vector<__fp16>& V, std::vector<float>& O,
                    uint32_t seq, uint32_t dim, bool causal) {
    O.assign((size_t)seq * dim, 0.f);
    const float sc = 1.f / sqrtf(float(dim));
    std::vector<float> s(seq);
    for (uint32_t i = 0; i < seq; ++i) {
        float mx = -INFINITY;
        uint32_t lim = causal ? i + 1 : seq;
        for (uint32_t j = 0; j < lim; ++j) {
            float dot = 0.f;
            for (uint32_t k = 0; k < dim; ++k)
                dot += float(Q[i*dim+k]) * float(K[j*dim+k]);
            s[j] = dot * sc;
            mx = fmaxf(mx, s[j]);
        }
        float sum = 0.f;
        for (uint32_t j = 0; j < lim; ++j) { s[j] = expf(s[j]-mx); sum += s[j]; }
        float inv = 1.f / sum;
        for (uint32_t j = 0; j < lim; ++j)
            for (uint32_t k = 0; k < dim; ++k)
                O[i*dim+k] += s[j] * inv * float(V[j*dim+k]);
    }
}

struct Cfg {
    const char* name;
    uint32_t    dim;
    bool        causal;
    bool        is_fa;   // fa (d=64): buf5=nheads;  mha (d=128): buf5=dim, buf6=nheads
};

int main(int argc, const char* argv[]) {
    const char* lib_path = (argc > 1) ? argv[1] : "build/attn.metallib";

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

    const uint32_t heads  = 1;
    const uint32_t LENS[] = {256, 512, 1024, 2048};

    Cfg cfgs[] = {
        {"fa_causal_64",    64,  true,  true},
        {"fa_noncausal_64", 64,  false, true},
        {"mha_causal",      128, true,  false},
        {"mha_noncausal",   128, false, false},
    };

    printf("%-20s  %6s  %8s  %8s  %8s\n", "kernel", "seq", "ms", "TFLOPS", "l2_rel");
    printf("%-20s  %6s  %8s  %8s  %8s\n",
           "--------------------", "------", "--------", "--------", "--------");

    for (auto& cfg : cfgs) {
        auto* fn = lib->newFunction(NS::String::string(cfg.name, NS::UTF8StringEncoding));
        if (!fn) { printf("  kernel '%s' not found\n", cfg.name); continue; }
        auto* pso = dev->newComputePipelineState(fn, &err); fn->release();
        if (!pso) { printf("  PSO '%s' failed\n", cfg.name); continue; }

        for (uint32_t seq : LENS) {
            size_t n = (size_t)heads * seq * cfg.dim;
            std::mt19937 rng(42);
            std::vector<__fp16> Q(n), K(n), V(n);
            fill(Q, rng); fill(K, rng); fill(V, rng);

            std::vector<float> ref;
            cpu_ref(Q, K, V, ref, seq, cfg.dim, cfg.causal);

            auto mode = MTL::ResourceStorageModeShared;
            size_t nb = n * sizeof(__fp16);
            auto* bQ = dev->newBuffer(Q.data(), nb, mode);
            auto* bK = dev->newBuffer(K.data(), nb, mode);
            auto* bV = dev->newBuffer(V.data(), nb, mode);
            auto* bO = dev->newBuffer(nb, mode);
            auto* bS = dev->newBuffer(&seq,      sizeof(seq),      mode);
            auto* bD = dev->newBuffer(&cfg.dim,  sizeof(cfg.dim),  mode);
            auto* bH = dev->newBuffer(&heads,    sizeof(heads),    mode);

            uint32_t grid_y = cfg.is_fa ? (seq + 31) / 32 : (seq + 3) / 4;
            uint32_t tg     = cfg.is_fa ? 1024 : 128;

            auto run = [&]() -> double {
                memset(bO->contents(), 0, nb);
                auto* cmd = queue->commandBuffer();
                auto* enc = cmd->computeCommandEncoder();
                enc->setComputePipelineState(pso);
                enc->setBuffer(bQ, 0, 0); enc->setBuffer(bK, 0, 1);
                enc->setBuffer(bV, 0, 2); enc->setBuffer(bO, 0, 3);
                enc->setBuffer(bS, 0, 4);
                if (cfg.is_fa) {
                    enc->setBuffer(bH, 0, 5);
                } else {
                    enc->setBuffer(bD, 0, 5);
                    enc->setBuffer(bH, 0, 6);
                }
                enc->dispatchThreadgroups(MTL::Size(heads, grid_y, 1), MTL::Size(tg, 1, 1));
                enc->endEncoding();
                cmd->commit(); cmd->waitUntilCompleted();
                return (cmd->GPUEndTime() - cmd->GPUStartTime()) * 1e3;
            };

            for (int w = 0; w < 5; w++) run();
            std::vector<double> ts;
            for (int i = 0; i < 30; i++) ts.push_back(run());
            std::sort(ts.begin(), ts.end());
            double ms = ts[ts.size() / 2];

            auto* gpu = (const __fp16*)bO->contents();
            double num = 0, den = 0;
            for (size_t i = 0; i < ref.size(); i++) {
                double e = double(gpu[i]) - double(ref[i]);
                num += e * e; den += double(ref[i]) * double(ref[i]);
            }
            float l2 = float(sqrt(num) / (sqrt(den) + 1e-12));

            double flops = cfg.causal
                ? (2.0*cfg.dim + 2.5) * seq * seq * 0.5 * heads
                : (2.0*cfg.dim + 2.5) * seq * seq * heads;
            double tflops = flops / (ms * 1e9);

            printf("%-20s  %6u  %8.3f  %8.4f  %8.5f\n", cfg.name, seq, ms, tflops, l2);

            bQ->release(); bK->release(); bV->release(); bO->release();
            bS->release(); bD->release(); bH->release();
        }
        pso->release();
        printf("\n");
    }

    lib->release(); queue->release(); dev->release();
    return 0;
}
