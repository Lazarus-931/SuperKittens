// act_bench.cpp — gelu / silu / relu kernel bench
// compile: clang++ -std=c++17 -O2 -I metal-cpp \
//   -framework Metal -framework Foundation -framework QuartzCore \
//   -o build/act_bench benchmark/activation/act_bench.cpp
// run: ./build/act_bench [metallib=build/activation.metallib]

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

static void fill_half(std::vector<__fp16>& v, std::mt19937& rng) {
    std::normal_distribution<float> d(0.f, 0.5f);
    for (auto& x : v) x = __fp16(d(rng));
}

static float l2_rel(const std::vector<__fp16>& g, const std::vector<float>& r) {
    double num = 0, den = 0;
    for (size_t i = 0; i < r.size(); i++) {
        double e = double(g[i]) - double(r[i]);
        num += e*e; den += double(r[i])*double(r[i]);
    }
    return float(sqrt(num) / (sqrt(den) + 1e-12));
}

static void ref_gelu(const std::vector<__fp16>& x, std::vector<float>& y) {
    y.resize(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        float v = float(x[i]);
        float a = 0.044715f * v * v * v;
        float c = tanhf(0.79788456f * (v + a));
        y[i] = 0.5f * v * (1.f + c);
    }
}

static void ref_silu(const std::vector<__fp16>& x, std::vector<float>& y) {
    y.resize(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        float v = float(x[i]);
        y[i] = v / (1.f + expf(-v));
    }
}

static void ref_relu(const std::vector<__fp16>& x, std::vector<float>& y) {
    y.resize(x.size());
    for (size_t i = 0; i < x.size(); i++) y[i] = fmaxf(0.f, float(x[i]));
}

int main(int argc, const char* argv[]) {
    const char* lib_path = (argc > 1) ? argv[1] : "build/activation.metallib";

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

    struct Act {
        const char* name;
        void (*ref)(const std::vector<__fp16>&, std::vector<float>&);
    };
    Act acts[] = {{"gelu", ref_gelu}, {"silu", ref_silu}, {"relu", ref_relu}};

    struct Cfg { uint32_t rows, cols; };
    Cfg cfgs[] = {
        {512,  1024}, {1024, 1024}, {2048, 1024},
        {512,  4096}, {1024, 4096}, {2048, 4096},
    };

    // THREADS=128, ROWS_PER_GROUP=4
    constexpr uint32_t ROWS_PER_TG = 4;

    printf("%-6s  %6s %6s  %8s  %8s  %8s\n", "kernel", "rows", "cols", "us", "GB/s", "l2_rel");
    printf("%-6s  %6s %6s  %8s  %8s  %8s\n", "------", "------", "------",
           "--------", "--------", "--------");

    auto mode = MTL::ResourceStorageModeShared;

    for (auto& act : acts) {
        auto* fn = lib->newFunction(NS::String::string(act.name, NS::UTF8StringEncoding));
        if (!fn) { printf("%s: kernel not found\n", act.name); continue; }
        auto* pso = dev->newComputePipelineState(fn, &err); fn->release();

        for (auto& cfg : cfgs) {
            size_t n = (size_t)cfg.rows * cfg.cols;
            std::mt19937 rng(42);
            std::vector<__fp16> x(n); fill_half(x, rng);

            std::vector<float> ref; act.ref(x, ref);

            auto* bX = dev->newBuffer(x.data(), n * sizeof(__fp16), mode);
            auto* bY = dev->newBuffer(n * sizeof(__fp16), mode);
            auto* bR = dev->newBuffer(&cfg.rows, sizeof(cfg.rows), mode);
            auto* bC = dev->newBuffer(&cfg.cols, sizeof(cfg.cols), mode);

            uint32_t tg_y = (cfg.rows + ROWS_PER_TG - 1) / ROWS_PER_TG;
            auto run = [&]() -> double {
                auto* cmd = queue->commandBuffer();
                auto* enc = cmd->computeCommandEncoder();
                enc->setComputePipelineState(pso);
                enc->setBuffer(bX, 0, 0); enc->setBuffer(bY, 0, 1);
                enc->setBuffer(bR, 0, 2); enc->setBuffer(bC, 0, 3);
                enc->dispatchThreadgroups(MTL::Size(1, tg_y, 1), MTL::Size(128, 1, 1));
                enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted();
                return (cmd->GPUEndTime() - cmd->GPUStartTime()) * 1e6;
            };

            for (int w = 0; w < 5; w++) run();
            std::vector<double> ts;
            for (int i = 0; i < 20; i++) ts.push_back(run());
            std::sort(ts.begin(), ts.end());
            double us = ts[ts.size() / 2];

            auto* gpu = (const __fp16*)bY->contents();
            std::vector<__fp16> gpu_v(n);
            memcpy(gpu_v.data(), gpu, n * sizeof(__fp16));
            float l2 = l2_rel(gpu_v, ref);

            double gb = (2.0 * n * sizeof(__fp16)) / 1e9;
            printf("%-6s  %6u %6u  %8.1f  %8.1f  %8.5f\n",
                   act.name, cfg.rows, cfg.cols, us, gb / (us * 1e-6), l2);

            bX->release(); bY->release(); bR->release(); bC->release();
        }
        pso->release();
        printf("\n");
    }

    lib->release(); queue->release(); dev->release();
    return 0;
}
