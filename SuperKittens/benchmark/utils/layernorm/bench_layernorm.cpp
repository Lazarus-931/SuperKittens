// bench_layernorm.cpp — layernorm kernel bench
// compile: clang++ -std=c++17 -O2 -I metal-cpp \
//   -framework Metal -framework Foundation -framework QuartzCore \
//   -o build/bench_layernorm bench/bench_layernorm.cpp
// run: ./build/bench_layernorm [metallib=build/layernorm.metallib]

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

static void fill_half(std::vector<__fp16>& v, std::mt19937& rng, float scale) {
    std::normal_distribution<float> n(0.0f, 1.0f);
    for (auto& x : v) x = __fp16(n(rng) * scale);
}

static void cpu_layernorm(const __fp16* x, const __fp16* gamma, const __fp16* beta,
                          float* y, uint32_t rows, uint32_t d, float eps) {
    for (uint32_t r = 0; r < rows; r++) {
        const size_t off = (size_t)r * d;
        float sum = 0.f, sumSq = 0.f;
        for (uint32_t k = 0; k < d; k++) { float v = float(x[off+k]); sum += v; sumSq += v*v; }
        float mean = sum / d;
        float inv_std = 1.f / sqrtf(sumSq / d - mean * mean + eps);
        for (uint32_t k = 0; k < d; k++)
            y[off+k] = (float(x[off+k]) - mean) * inv_std * float(gamma[k]) + float(beta[k]);
    }
}

int main(int argc, const char* argv[]) {
    const char* lib_path = (argc > 1) ? argv[1] : "build/layernorm.metallib";

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
    auto* fn  = lib->newFunction(NS::String::string("layernorm", NS::UTF8StringEncoding));
    auto* pso = dev->newComputePipelineState(fn, &err); fn->release();

    const float eps = 1e-5f;
    struct Cfg { uint32_t rows, d; };
    Cfg cfgs[] = {{512,1024},{1024,1024},{2048,1024},{512,4096},{1024,4096},{2048,4096},{4096,4096}};

    printf("%-6s %-6s  %10s  %8s  %8s  %8s\n", "rows", "d", "us", "GB/s", "max_err", "status");
    printf("%-6s %-6s  %10s  %8s  %8s  %8s\n", "------","------","----------","--------","--------","--------");

    for (auto& cfg : cfgs) {
        size_t n = (size_t)cfg.rows * cfg.d;
        std::mt19937 rng(42);
        std::vector<__fp16> x(n), gamma(cfg.d), beta(cfg.d);
        fill_half(x, rng, 1.0f); fill_half(gamma, rng, 0.5f); fill_half(beta, rng, 0.1f);

        std::vector<float> ref(n);
        cpu_layernorm(x.data(), gamma.data(), beta.data(), ref.data(), cfg.rows, cfg.d, eps);

        auto mode = MTL::ResourceStorageModeShared;
        auto* bX   = dev->newBuffer(x.data(),     n*sizeof(__fp16),       mode);
        auto* bG   = dev->newBuffer(gamma.data(),  cfg.d*sizeof(__fp16),   mode);
        auto* bB   = dev->newBuffer(beta.data(),   cfg.d*sizeof(__fp16),   mode);
        auto* bY   = dev->newBuffer(n*sizeof(__fp16),                       mode);
        auto* bR   = dev->newBuffer(&cfg.rows,     sizeof(cfg.rows),        mode);
        auto* bD   = dev->newBuffer(&cfg.d,        sizeof(cfg.d),           mode);
        auto* bEps = dev->newBuffer(&eps,          sizeof(eps),             mode);

        auto run = [&]() -> double {
            auto* cmd = queue->commandBuffer();
            auto* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(pso);
            enc->setBuffer(bX, 0,0); enc->setBuffer(bG, 0,1); enc->setBuffer(bB, 0,2);
            enc->setBuffer(bY, 0,3); enc->setBuffer(bR, 0,4); enc->setBuffer(bD, 0,5);
            enc->setBuffer(bEps, 0,6);
            enc->dispatchThreadgroups(MTL::Size(1, (cfg.rows+3)/4, 1), MTL::Size(128,1,1));
            enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted();
            return (cmd->GPUEndTime() - cmd->GPUStartTime()) * 1e6;
        };

        for (int w = 0; w < 5; w++) run();
        std::vector<double> ts;
        for (int i = 0; i < 20; i++) ts.push_back(run());
        std::sort(ts.begin(), ts.end());
        double us = ts[ts.size()/2];

        std::vector<__fp16> gpu(n);
        memcpy(gpu.data(), bY->contents(), n*sizeof(__fp16));
        float max_err = 0.f;
        for (size_t i = 0; i < n; i++) max_err = fmaxf(max_err, fabsf(float(gpu[i]) - ref[i]));

        double gbps = (n * 1.5 * sizeof(__fp16)) / (us * 1e3);
        printf("%-6u %-6u  %10.1f  %8.1f  %8.5f  %s\n",
               cfg.rows, cfg.d, us, gbps, max_err, max_err < 0.05f ? "PASS" : "FAIL");

        bX->release(); bG->release(); bB->release(); bY->release();
        bR->release(); bD->release(); bEps->release();
    }

    pso->release(); lib->release(); queue->release(); dev->release();
    return 0;
}
