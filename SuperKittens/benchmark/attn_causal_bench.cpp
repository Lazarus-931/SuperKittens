//
//  attn_causal_bench.cpp
//  SuperKittens — causal MHA benchmark against MLX baseline
//
//  Build: add to Xcode target, or compile manually against metallib
//  Usage: ./attn_causal_bench <metallib> [seq=2048] [d=128] [heads=1] [iters=20]
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

struct BenchResult {
    double median_us;
    double min_us;
    double tflops;
};

// CPU reference for causal attention (float32 ground truth)
static void causal_attention_cpu(
    const __fp16* Q, const __fp16* K, const __fp16* V,
    float* O, uint32_t seq, uint32_t d)
{
    std::vector<float> scores(seq);
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

static void fill_half(std::vector<__fp16>& v, std::mt19937& rng, float scale) {
    std::normal_distribution<float> n(0.0f, 1.0f);
    for (auto& x : v) x = __fp16(n(rng) * scale);
}

struct ErrStats {
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    float l2_rel = 0.0f;
    bool pass = false;
};

static ErrStats verify(const std::vector<__fp16>& gpu, const std::vector<float>& ref,
                       float max_tol = 0.05f, float mean_tol = 0.02f) {
    ErrStats s;
    double sum_abs = 0.0, sum_num = 0.0, sum_den = 0.0;
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
    s.l2_rel  = float(std::sqrt(sum_num) / (std::sqrt(sum_den) + 1e-12));
    s.pass = (s.max_abs < max_tol) && (s.mean_abs < mean_tol);
    return s;
}

} // namespace

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <metallib> [seq=2048] [d=128] [heads=1] [iters=20]\n", argv[0]);
        return 1;
    }

    const char*       metallib_path = argv[1];
    const uint32_t    seq   = (argc > 2) ? uint32_t(std::atoi(argv[2])) : 2048;
    const uint32_t    d     = (argc > 3) ? uint32_t(std::atoi(argv[3])) : 128;
    const uint32_t    heads = (argc > 4) ? uint32_t(std::atoi(argv[4])) : 1;
    const int         iters = (argc > 5) ? std::atoi(argv[5])    : 20;

    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) { std::fprintf(stderr, "No Metal device\n"); return 1; }
    auto* queue = device->newCommandQueue();

    NS::Error* err = nullptr;
    auto* url = NS::URL::fileURLWithPath(
        NS::String::string(metallib_path, NS::UTF8StringEncoding));
    auto* lib = device->newLibrary(url, &err);
    if (!lib) {
        std::fprintf(stderr, "Failed to load metallib: %s\n",
                     err ? err->localizedDescription()->utf8String() : "(unknown)");
        return 1;
    }

    auto* fn = lib->newFunction(
        NS::String::string("mha_causal", NS::UTF8StringEncoding));
    if (!fn) {
        std::fprintf(stderr, "Kernel 'mha_causal' not found in metallib\n");
        return 1;
    }
    auto* pso = device->newComputePipelineState(fn, &err);
    fn->release();
    if (!pso) {
        std::fprintf(stderr, "PSO creation failed: %s\n",
                     err ? err->localizedDescription()->utf8String() : "(unknown)");
        return 1;
    }

    // ── Generate test data ──
    std::mt19937 rng(42);
    const size_t total_elems = (size_t)heads * seq * d;
    std::vector<__fp16> Q(total_elems), K(total_elems), V(total_elems);
    fill_half(Q, rng, 0.5f);
    fill_half(K, rng, 0.5f);
    fill_half(V, rng, 0.5f);

    // CPU reference (first head only for verification)
    std::vector<float> ref(seq * d);
    causal_attention_cpu(Q.data(), K.data(), V.data(), ref.data(), seq, d);

    // ── Metal buffers ──
    const auto storage = MTL::ResourceStorageModeShared;
    const size_t buf_bytes = total_elems * sizeof(__fp16);
    auto* bufQ   = device->newBuffer(Q.data(), buf_bytes, storage);
    auto* bufK   = device->newBuffer(K.data(), buf_bytes, storage);
    auto* bufV   = device->newBuffer(V.data(), buf_bytes, storage);
    auto* bufO   = device->newBuffer(buf_bytes, storage);
    auto* bufSeq = device->newBuffer(&seq, sizeof(seq), storage);
    auto* bufD   = device->newBuffer(&d, sizeof(d), storage);
    auto* bufH   = device->newBuffer(&heads, sizeof(heads), storage);

    const uint32_t grid_y = (seq + 3) / 4;  // ceil_div(seq, ATTN_ROWS_PER_GRP)

    auto run = [&]() -> double {
        std::memset(bufO->contents(), 0, buf_bytes);
        auto* cmd = queue->commandBuffer();
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(bufQ,   0, 0);
        enc->setBuffer(bufK,   0, 1);
        enc->setBuffer(bufV,   0, 2);
        enc->setBuffer(bufO,   0, 3);
        enc->setBuffer(bufSeq, 0, 4);
        enc->setBuffer(bufD,   0, 5);
        enc->setBuffer(bufH,   0, 6);
        enc->dispatchThreadgroups(MTL::Size(heads, grid_y, 1), MTL::Size(128, 1, 1));
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
        return (cmd->GPUEndTime() - cmd->GPUStartTime()) * 1e6;  // us
    };

    // ── Correctness check (first head) ──
    std::printf("== causal mha  kernel=mha_causal  seq=%u d=%u heads=%u ==\n", seq, d, heads);
    run();  // one warmup to allocate
    std::vector<__fp16> gpu_out(seq * d);
    std::memcpy(gpu_out.data(), bufO->contents(), seq * d * sizeof(__fp16));
    const ErrStats st = verify(gpu_out, ref);

    std::printf("  accuracy: max_abs=%.5f  mean_abs=%.5f  l2_rel=%.5f  %s\n",
                st.max_abs, st.mean_abs, st.l2_rel, st.pass ? "PASS" : "FAIL");
    if (!st.pass) {
        std::printf("  Sample mismatches:\n");
        int shown = 0;
        for (uint32_t i = 0; i < seq && shown < 5; i++) {
            for (uint32_t k = 0; k < d && shown < 5; k++) {
                float g = float(gpu_out[i * d + k]);
                float c = ref[i * d + k];
                if (std::fabs(g - c) > 0.05f) {
                    std::printf("    [%u,%u] gpu=%.4f cpu=%.4f diff=%.4f\n", i, k, g, c, g - c);
                    shown++;
                }
            }
        }
    }

    // ── Benchmark ──
    for (int i = 0; i < 5; ++i) run();  // warmup
    std::vector<double> times;
    for (int i = 0; i < iters; ++i) times.push_back(run());
    std::sort(times.begin(), times.end());

    const double median_us = times[times.size() / 2];
    const double min_us    = times.front();

    // Causal FLOPs: half of non-causal + mask overhead
    // 2 * d * seq * (seq+1)/2 + seq*(seq+1)/2 = d * seq * (seq+1) + 0.5*seq*(seq+1)
    const double flops = (2.0 * d + 2.5) * double(seq) * double(seq) * double(heads);
    const double tflops = flops / (median_us * 1e6);  // TFLOPs

    std::printf("  time (median): %.1f us\n", median_us);
    std::printf("  time (min):    %.1f us\n", min_us);
    std::printf("  TFLOPS:        %.2f\n", tflops);

    // Cleanup
    bufQ->release(); bufK->release(); bufV->release(); bufO->release();
    bufSeq->release(); bufD->release(); bufH->release();
    pso->release(); lib->release(); queue->release(); device->release();
    return 0;
}
