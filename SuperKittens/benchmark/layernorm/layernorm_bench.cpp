//
//  layernorm_bench.cpp
//  SuperKittens
//
//  Usage: ./layernorm_bench <metallib> [rows=2048] [d=128] [iters=20]
//

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

void fill_half(std::vector<__fp16>& v, std::mt19937& rng, float scale) {
    std::normal_distribution<float> n(0.0f, 1.0f);
    for (auto& x : v) x = __fp16(n(rng) * scale);
}

void layernorm_cpu(const __fp16* x, const __fp16* gamma, const __fp16* beta,
                   float* y, uint32_t rows, uint32_t d, float eps) {
    for (uint32_t r = 0; r < rows; r++) {
        const size_t off = (size_t)r * d;
        float sum = 0.0f, sumSq = 0.0f;
        for (uint32_t k = 0; k < d; k++) {
            float v = float(x[off + k]);
            sum += v;
            sumSq += v * v;
        }
        float mean = sum / d;
        float var = sumSq / d - mean * mean;
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (uint32_t k = 0; k < d; k++) {
            float v = float(x[off + k]);
            v = (v - mean) * inv_std;
            v = v * float(gamma[k]) + float(beta[k]);
            y[off + k] = v;
        }
    }
}

struct ErrStats {
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    bool pass = false;
};

ErrStats verify(const std::vector<__fp16>& gpu, const std::vector<float>& ref,
                float tol = 0.02f) {
    ErrStats s;
    double sum_abs = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
        float d = std::fabs(float(gpu[i]) - ref[i]);
        s.max_abs = std::max(s.max_abs, d);
        sum_abs += d;
    }
    s.mean_abs = float(sum_abs / ref.size());
    s.pass = (s.max_abs < tol);
    return s;
}

} // namespace

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <metallib> [rows=2048] [d=128] [iters=20]\n", argv[0]);
        return 1;
    }
    const char*   lib_path = argv[1];
    const uint32_t rows    = (argc > 2) ? uint32_t(std::atoi(argv[2])) : 2048;
    const uint32_t d       = (argc > 3) ? uint32_t(std::atoi(argv[3])) : 128;
    const int      iters   = (argc > 4) ? std::atoi(argv[4]) : 20;
    const float    eps     = 1e-5f;

    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    auto* queue = device->newCommandQueue();

    NS::Error* err = nullptr;
    auto* url = NS::URL::fileURLWithPath(NS::String::string(lib_path, NS::UTF8StringEncoding));
    auto* lib = device->newLibrary(url, &err);
    auto* fn = lib->newFunction(NS::String::string("layernorm", NS::UTF8StringEncoding));
    auto* pso = device->newComputePipelineState(fn, &err);
    fn->release();

    // Generate data
    std::mt19937 rng(42);
    const size_t x_elems = (size_t)rows * d;
    std::vector<__fp16> x(x_elems), gamma(d), beta(d);
    fill_half(x, rng, 1.0f);
    fill_half(gamma, rng, 0.5f);
    fill_half(beta, rng, 0.1f);

    // CPU reference
    std::vector<float> ref(x_elems);
    layernorm_cpu(x.data(), gamma.data(), beta.data(), ref.data(), rows, d, eps);

    // Metal buffers
    const auto s = MTL::ResourceStorageModeShared;
    auto* bufX   = device->newBuffer(x.data(), x_elems * sizeof(__fp16), s);
    auto* bufG   = device->newBuffer(gamma.data(), d * sizeof(__fp16), s);
    auto* bufB   = device->newBuffer(beta.data(), d * sizeof(__fp16), s);
    auto* bufY   = device->newBuffer(x_elems * sizeof(__fp16), s);
    auto* bufR   = device->newBuffer(&rows, sizeof(rows), s);
    auto* bufD   = device->newBuffer(&d, sizeof(d), s);
    auto* bufEps = device->newBuffer(&eps, sizeof(eps), s);

    auto run = [&]() -> double {
        auto* cmd = queue->commandBuffer();
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(bufX, 0, 0); enc->setBuffer(bufG, 0, 1);
        enc->setBuffer(bufB, 0, 2); enc->setBuffer(bufY, 0, 3);
        enc->setBuffer(bufR, 0, 4); enc->setBuffer(bufD, 0, 5);
        enc->setBuffer(bufEps, 0, 6);
        enc->dispatchThreadgroups(MTL::Size(1, (rows + 3) / 4, 1), MTL::Size(128, 1, 1));
        enc->endEncoding();
        cmd->commit(); cmd->waitUntilCompleted();
        return (cmd->GPUEndTime() - cmd->GPUStartTime()) * 1e6;
    };

    // Correctness
    std::memset(bufY->contents(), 0, x_elems * sizeof(__fp16));
    run();
    std::vector<__fp16> gpu(x_elems);
    std::memcpy(gpu.data(), bufY->contents(), x_elems * sizeof(__fp16));
    auto st = verify(gpu, ref, 0.05f);

    std::printf("layernorm  rows=%u d=%u  max=%.5f mean=%.5f %s\n",
                rows, d, st.max_abs, st.mean_abs, st.pass ? "PASS" : "FAIL");

    // Benchmark
    for (int i = 0; i < 5; i++) run();
    std::vector<double> times;
    for (int i = 0; i < iters; i++) times.push_back(run());
    std::sort(times.begin(), times.end());
    double us = times[times.size() / 2];
    double gbps = (x_elems * 1.5 * sizeof(__fp16)) / (us * 1e3);  // read x, gamma, beta; write y
    std::printf("  median=%.1f us  min=%.1f us  BW=%.1f GB/s\n", us, times.front(), gbps);

    bufX->release(); bufG->release(); bufB->release(); bufY->release();
    bufR->release(); bufD->release(); bufEps->release();
    pso->release(); lib->release(); queue->release(); device->release();
    return 0;
}
