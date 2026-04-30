//
//  attn_causal_smoke_test.cpp
//  SuperKittens
//

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
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct ErrStats {
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    float l2_rel = 0.0f;
};

static void fill_half(std::vector<__fp16>& v, std::mt19937& rng, float scale) {
    std::normal_distribution<float> n(0.0f, 1.0f);
    for (auto& x : v) x = __fp16(n(rng) * scale);
}

static void causal_attention_cpu(
    const std::vector<__fp16>& Q,
    const std::vector<__fp16>& K,
    const std::vector<__fp16>& V,
    std::vector<float>& O,
    uint32_t seq,
    uint32_t d)
{
    O.assign((size_t)seq * d, 0.0f);
    std::vector<float> scores(seq, 0.0f);
    const float scale = 1.0f / std::sqrt(float(d));

    for (uint32_t i = 0; i < seq; ++i) {
        float row_max = -INFINITY;
        for (uint32_t j = 0; j <= i; ++j) {
            float dot = 0.0f;
            for (uint32_t k = 0; k < d; ++k)
                dot += float(Q[i * d + k]) * float(K[j * d + k]);
            scores[j] = dot * scale;
            row_max = std::max(row_max, scores[j]);
        }
        float sum = 0.0f;
        for (uint32_t j = 0; j <= i; ++j) {
            scores[j] = std::exp(scores[j] - row_max);
            sum += scores[j];
        }
        const float inv_sum = 1.0f / sum;
        for (uint32_t j = 0; j <= i; ++j) scores[j] *= inv_sum;

        for (uint32_t k = 0; k < d; ++k) {
            float acc = 0.0f;
            for (uint32_t j = 0; j <= i; ++j)
                acc += scores[j] * float(V[j * d + k]);
            O[i * d + k] = acc;
        }
    }
}

static ErrStats compare_hf(const std::vector<__fp16>& gpu,
                           const std::vector<float>& ref) {
    ErrStats s;
    double sum_abs = 0.0;
    double sum_num = 0.0;
    double sum_den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float g = float(gpu[i]);
        const float r = ref[i];
        const float d = std::fabs(g - r);
        s.max_abs = std::max(s.max_abs, d);
        sum_abs += d;
        sum_num += double(d) * double(d);
        sum_den += double(r) * double(r);
    }
    s.mean_abs = float(sum_abs / ref.size());
    s.l2_rel = float(std::sqrt(sum_num) / (std::sqrt(sum_den) + 1e-12));
    return s;
}

} // namespace

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <metallib> [seq=512] [d=128]\n", argv[0]);
        return 1;
    }

    const char* metallib_path = argv[1];
    const uint32_t seq = (argc > 2) ? uint32_t(std::atoi(argv[2])) : 512;
    const uint32_t d = (argc > 3) ? uint32_t(std::atoi(argv[3])) : 128;

    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::fprintf(stderr, "No Metal device\n");
        return 1;
    }
    auto* queue = device->newCommandQueue();

    NS::Error* err = nullptr;
    auto* url = NS::URL::fileURLWithPath(NS::String::string(metallib_path, NS::UTF8StringEncoding));
    auto* lib = device->newLibrary(url, &err);
    if (!lib) {
        std::fprintf(stderr, "Failed to load metallib: %s\n",
                     err ? err->localizedDescription()->utf8String() : "(unknown)");
        return 1;
    }

    auto* fn = lib->newFunction(NS::String::string("mha_causal", NS::UTF8StringEncoding));
    if (!fn) {
        std::fprintf(stderr, "Kernel 'mha_causal' not found\n");
        return 1;
    }
    auto* pso = device->newComputePipelineState(fn, &err);
    fn->release();
    if (!pso) {
        std::fprintf(stderr, "PSO creation failed: %s\n",
                     err ? err->localizedDescription()->utf8String() : "(unknown)");
        return 1;
    }

    std::mt19937 rng(42);
    const size_t elems = (size_t)seq * d;
    std::vector<__fp16> Q(elems), K(elems), V(elems);
    fill_half(Q, rng, 0.5f);
    fill_half(K, rng, 0.5f);
    fill_half(V, rng, 0.5f);

    std::vector<float> ref;
    const auto cpu_t0 = std::chrono::steady_clock::now();
    causal_attention_cpu(Q, K, V, ref, seq, d);
    const auto cpu_t1 = std::chrono::steady_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();

    const auto storage = MTL::ResourceStorageModeShared;
    auto* bufQ = device->newBuffer(Q.data(), elems * sizeof(__fp16), storage);
    auto* bufK = device->newBuffer(K.data(), elems * sizeof(__fp16), storage);
    auto* bufV = device->newBuffer(V.data(), elems * sizeof(__fp16), storage);
    auto* bufO = device->newBuffer(elems * sizeof(__fp16), storage);
    uint32_t num_heads = 1;
    auto* bufSeq = device->newBuffer(&seq, sizeof(seq), storage);
    auto* bufD = device->newBuffer(&d, sizeof(d), storage);
    auto* bufH = device->newBuffer(&num_heads, sizeof(num_heads), storage);

    auto run = [&]() -> double {
        std::memset(bufO->contents(), 0, elems * sizeof(__fp16));
        auto* cmd = queue->commandBuffer();
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(bufQ, 0, 0);
        enc->setBuffer(bufK, 0, 1);
        enc->setBuffer(bufV, 0, 2);
        enc->setBuffer(bufO, 0, 3);
        enc->setBuffer(bufSeq, 0, 4);
        enc->setBuffer(bufD, 0, 5);
        enc->setBuffer(bufH, 0, 6);
        enc->dispatchThreadgroups(MTL::Size(num_heads, (seq + 3) / 4, 1), MTL::Size(128, 1, 1));
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
        return (cmd->GPUEndTime() - cmd->GPUStartTime()) * 1000.0;
    };

    for (int i = 0; i < 5; ++i) run();
    std::vector<double> ts;
    for (int i = 0; i < 20; ++i) ts.push_back(run());
    std::sort(ts.begin(), ts.end());
    const double gpu_ms = ts[ts.size() / 2];

    std::vector<__fp16> gpu(elems);
    std::memcpy(gpu.data(), bufO->contents(), elems * sizeof(__fp16));
    const ErrStats st = compare_hf(gpu, ref);

    std::printf("== causal mha baseline ==\n");
    std::printf("  seq=%u d=%u\n", seq, d);
    std::printf("  CPU ref:    %.3f ms\n", cpu_ms);
    std::printf("  GPU median: %.3f ms (%zu iters)\n", gpu_ms, ts.size());
    std::printf("  speedup:    %.2fx\n", cpu_ms / gpu_ms);
    std::printf("  accuracy:   max_abs=%.5f mean_abs=%.5f l2_rel=%.5f\n",
                st.max_abs, st.mean_abs, st.l2_rel);

    bufQ->release();
    bufK->release();
    bufV->release();
    bufO->release();
    bufSeq->release();
    bufD->release();
    bufH->release();
    pso->release();
    lib->release();
    queue->release();
    device->release();
    return 0;
}
