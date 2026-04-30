//
//  gemm_bias_silu_smoke_test.cpp
//  SuperKittens
//
//  CPU vs GPU smoke test for the first fused GEMM path:
//  C = silu(alpha * (A @ B) + beta * C + bias)
//
//  Build:
//    clang++ -std=gnu++20 -O2 -arch arm64 -I metal-cpp \
//      -framework Metal -framework Foundation -framework QuartzCore \
//      SuperKittens/benchmark/gemm/gemm_bias_silu_smoke_test.cpp -o build/gemm_bias_silu_smoke_test
//
//  Run:
//    ./build/gemm_bias_silu_smoke_test <metallib> [M=256] [N=256] [K=256]
//

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "../../meow.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

template <typename T>
static void dump_bin(const std::string& path, const std::vector<T>& v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
}

static float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

static void cpu_reference(
    const std::vector<__fp16>& A,
    const std::vector<__fp16>& B,
    const std::vector<__fp16>& Cin,
    const std::vector<__fp16>& bias,
    std::vector<float>& Cout,
    const meow::gemm::GemmParams& p)
{
    Cout.assign(static_cast<size_t>(p.m) * p.n, 0.0f);
    for (int row = 0; row < p.m; ++row) {
        for (int col = 0; col < p.n; ++col) {
            float acc = 0.0f;
            for (int kk = 0; kk < p.k; ++kk) {
                const float a = float(A[static_cast<size_t>(row) * p.lda + kk]);
                const float b = float(B[static_cast<size_t>(kk) * p.ldb + col]);
                acc += a * b;
            }
            const float prior = float(Cin[static_cast<size_t>(row) * p.ldc + col]);
            float out = p.alpha * acc + p.beta * prior + float(bias[col]);
            Cout[static_cast<size_t>(row) * p.ldc + col] = silu(out);
        }
    }
}

static ErrStats compare_hf(const std::vector<__fp16>& gpu, const std::vector<float>& ref) {
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
        std::fprintf(stderr, "usage: %s <metallib> [M] [N] [K] [--dump <dir>] [--no-verify] [--iters N] [--force-generic]\n", argv[0]);
        return 1;
    }

    const char* metallib_path = argv[1];
    const int M = (argc > 2) ? std::atoi(argv[2]) : 256;
    const int N = (argc > 3) ? std::atoi(argv[3]) : 256;
    const int K = (argc > 4) ? std::atoi(argv[4]) : 256;
    std::string dump_dir;
    bool verify = true;
    int iters = 20;
    bool force_generic = false;
    for (int i = 5; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--dump") dump_dir = argv[i + 1];
        if (std::string(argv[i]) == "--iters") iters = std::atoi(argv[i + 1]);
    }
    for (int i = 5; i < argc; ++i) {
        if (std::string(argv[i]) == "--no-verify") verify = false;
        if (std::string(argv[i]) == "--force-generic") force_generic = true;
    }

    meow::gemm::GemmParams p = meow::gemm::make_gemm_params(
        M, N, K, meow::gemm::GEMM_OP_N, meow::gemm::GEMM_OP_N,
        1.0f, 0.0f, meow::gemm::GEMM_SPEC_GENERIC, meow::gemm::GEMM_EPILOGUE_BIAS_SILU);
    if (!force_generic) {
        if (M == 2048 && N == 3072 && K == 4096) p.specialization = meow::gemm::GEMM_SPEC_2048_3072_4096;
        if (M == 3072 && N == 2048 && K == 4096) p.specialization = meow::gemm::GEMM_SPEC_3072_2048_4096;
        if (M == 4096 && N == 4096 && K == 4096) p.specialization = meow::gemm::GEMM_SPEC_4096_4096_4096;
    }

    const char* kernel_name = meow::gemm::kernel_name_for(p);
    if (!kernel_name) {
        std::fprintf(stderr, "unsupported GEMM variant\n");
        return 1;
    }

    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::fprintf(stderr, "No Metal device\n");
        return 1;
    }
    MTL::CommandQueue* queue = device->newCommandQueue();

    NS::Error* err = nullptr;
    auto* url = NS::URL::fileURLWithPath(NS::String::string(metallib_path, NS::UTF8StringEncoding));
    MTL::Library* lib = device->newLibrary(url, &err);
    if (!lib) {
        std::fprintf(stderr, "Failed to load metallib at %s: %s\n",
                     metallib_path,
                     err ? err->localizedDescription()->utf8String() : "(unknown)");
        return 1;
    }

    auto* fn = lib->newFunction(NS::String::string(kernel_name, NS::UTF8StringEncoding));
    if (!fn) {
        std::fprintf(stderr, "Kernel '%s' not found\n", kernel_name);
        return 1;
    }
    auto* pso = device->newComputePipelineState(fn, &err);
    fn->release();
    if (!pso) {
        std::fprintf(stderr, "PSO creation failed: %s\n",
                     err ? err->localizedDescription()->utf8String() : "(no error)");
        return 1;
    }

    const size_t a_elems = static_cast<size_t>(M) * K;
    const size_t b_elems = static_cast<size_t>(K) * N;
    const size_t c_elems = static_cast<size_t>(M) * N;
    const size_t bias_elems = static_cast<size_t>(N);

    std::mt19937 rng(42);
    std::vector<__fp16> A(a_elems), B(b_elems), C0(c_elems), Cgpu(c_elems), bias(bias_elems);
    fill_half(A, rng, 0.25f);
    fill_half(B, rng, 0.25f);
    fill_half(C0, rng, 0.05f);
    fill_half(bias, rng, 0.1f);

    std::vector<float> Cref;
    double cpu_ms = 0.0;
    if (verify) {
        auto cpu_t0 = std::chrono::steady_clock::now();
        cpu_reference(A, B, C0, bias, Cref, p);
        auto cpu_t1 = std::chrono::steady_clock::now();
        cpu_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();
    }

    auto* bA = device->newBuffer(a_elems * sizeof(__fp16), MTL::ResourceStorageModeShared);
    auto* bB = device->newBuffer(b_elems * sizeof(__fp16), MTL::ResourceStorageModeShared);
    auto* bC = device->newBuffer(c_elems * sizeof(__fp16), MTL::ResourceStorageModeShared);
    auto* bP = device->newBuffer(&p, sizeof(p), MTL::ResourceStorageModeShared);
    auto* bBias = device->newBuffer(bias_elems * sizeof(__fp16), MTL::ResourceStorageModeShared);

    std::memcpy(bA->contents(), A.data(), a_elems * sizeof(__fp16));
    std::memcpy(bB->contents(), B.data(), b_elems * sizeof(__fp16));
    std::memcpy(bC->contents(), C0.data(), c_elems * sizeof(__fp16));
    std::memcpy(bBias->contents(), bias.data(), bias_elems * sizeof(__fp16));

    const meow::gemm::GemmLaunchConfig launch = meow::gemm::launch_config_for(p);

    auto run = [&]() -> double {
        std::memcpy(bC->contents(), C0.data(), c_elems * sizeof(__fp16));
        auto* cmd = queue->commandBuffer();
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(bA, 0, 0);
        enc->setBuffer(bB, 0, 1);
        enc->setBuffer(bC, 0, 2);
        enc->setBuffer(bP, 0, 3);
        enc->setBuffer(bBias, 0, 4);
        enc->dispatchThreadgroups(
            MTL::Size(launch.grid_x, launch.grid_y, 1),
            MTL::Size(launch.threads_x, launch.threads_y, 1));
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
        return (cmd->GPUEndTime() - cmd->GPUStartTime()) * 1000.0;
    };

    for (int i = 0; i < 5; ++i) run();
    std::vector<double> ts;
    for (int i = 0; i < iters; ++i) ts.push_back(run());
    std::sort(ts.begin(), ts.end());
    const double gpu_ms = ts[ts.size() / 2];

    std::memcpy(Cgpu.data(), bC->contents(), c_elems * sizeof(__fp16));
    ErrStats stats = {};
    if (verify) stats = compare_hf(Cgpu, Cref);

    std::printf("== gemm bias+silu smoke test ==\n");
    std::printf("  kernel:     %s\n", kernel_name);
    std::printf("  shape:      M=%d N=%d K=%d\n", M, N, K);
    if (verify) std::printf("  CPU ref:    %.3f ms\n", cpu_ms);
    std::printf("  GPU median: %.3f ms (%zu iters)\n", gpu_ms, ts.size());
    if (verify) {
        std::printf("  accuracy:   max_abs=%.5f mean_abs=%.5f l2_rel=%.5f\n",
                    stats.max_abs, stats.mean_abs, stats.l2_rel);
    }
    const bool pass = !verify || (std::isfinite(stats.l2_rel) && stats.l2_rel < 5e-2f);
    std::printf("  %s\n", pass ? "PASS" : "FAIL");

    if (!dump_dir.empty() && verify) {
        dump_bin(dump_dir + "/a.bin", A);
        dump_bin(dump_dir + "/b.bin", B);
        dump_bin(dump_dir + "/c_in.bin", C0);
        dump_bin(dump_dir + "/bias.bin", bias);
        dump_bin(dump_dir + "/y_gpu.bin", Cgpu);
        dump_bin(dump_dir + "/y_cpu.bin", Cref);
        std::ofstream meta(dump_dir + "/meta.txt");
        meta << "mode=bias_silu\n";
        meta << "M=" << M << " N=" << N << " K=" << K << "\n";
        meta << "alpha=" << p.alpha << " beta=" << p.beta << "\n";
        meta << "lda=" << p.lda << " ldb=" << p.ldb << " ldc=" << p.ldc << "\n";
    }

    bA->release();
    bB->release();
    bC->release();
    bP->release();
    bBias->release();
    pso->release();
    lib->release();
    queue->release();
    device->release();
    return pass ? 0 : 1;
}
