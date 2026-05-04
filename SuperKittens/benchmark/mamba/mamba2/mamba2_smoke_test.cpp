//
//  mamba2_smoke_test.cpp — v1
//  SuperKittens — CLI smoke test for mamba2 SSD forward.
//
//  Build:
//    clang++ -std=gnu++20 -O2 -arch arm64 -I metal-cpp \
//      -framework Metal -framework Foundation -framework QuartzCore \
//      SuperKittens/benchmark/mamba/mamba2_smoke_test.cpp -o build/mamba2_smoke_test
//
//  Run:
//    ./build/mamba2_smoke_test <metallib> [L=256] [B=1] [H=2] [--dump <dir>]
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
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

struct Mamba2FwdArgsHost {
    uint32_t batch;
    uint32_t nheads;
    uint32_t seq_len;
    uint32_t n_chunks;
};

template <typename T>
static void dump_bin(const std::string& path, const std::vector<T>& v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
}

static void fill_inputs(std::vector<__fp16>& Q, std::vector<__fp16>& K,
                        std::vector<__fp16>& V, std::vector<float>& A,
                        uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> n(0.0f, 1.0f);
    for (auto& x : Q) x = __fp16(n(rng) * 0.5f);
    for (auto& x : K) x = __fp16(n(rng) * 0.5f);
    for (auto& x : V) x = __fp16(n(rng) * 0.5f);
    std::normal_distribution<float> a_dist(-0.1f, 0.2f);
    for (auto& x : A) x = a_dist(rng);
}

static void cpu_reference(const std::vector<__fp16>& Q,
                          const std::vector<__fp16>& K,
                          const std::vector<__fp16>& V,
                          const std::vector<float>& A,
                          std::vector<float>& Yout,
                          int B, int H, int L, int D_qk, int D_v) {
    Yout.assign((size_t)B * H * L * D_v, 0.0f);
    std::vector<float> state((size_t)D_qk * D_v, 0.0f);

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            std::fill(state.begin(), state.end(), 0.0f);
            const size_t qk_base = ((size_t)b * H + h) * L * D_qk;
            const size_t v_base = ((size_t)b * H + h) * L * D_v;
            const size_t a_base = ((size_t)b * H + h) * L;

            for (int t = 0; t < L; ++t) {
                const float decay = std::exp(A[a_base + t]);
                for (int i = 0; i < D_qk; ++i) {
                    const float k_i = float(K[qk_base + (size_t)t * D_qk + i]);
                    for (int j = 0; j < D_v; ++j) {
                        const float v_j = float(V[v_base + (size_t)t * D_v + j]);
                        float& s = state[(size_t)i * D_v + j];
                        s = decay * s + k_i * v_j;
                    }
                }
                for (int j = 0; j < D_v; ++j) {
                    float acc = 0.0f;
                    for (int i = 0; i < D_qk; ++i) {
                        acc += float(Q[qk_base + (size_t)t * D_qk + i]) *
                               state[(size_t)i * D_v + j];
                    }
                    Yout[v_base + (size_t)t * D_v + j] = acc;
                }
            }
        }
    }
}

struct ErrStats {
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    float l2_rel = 0.0f;
};

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
        if (d > s.max_abs) s.max_abs = d;
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
        std::fprintf(stderr, "usage: %s <metallib> [L] [B] [H] [--dump <dir>]\n", argv[0]);
        return 1;
    }
    const char* metallib_path = argv[1];
    const int L = (argc > 2) ? std::atoi(argv[2]) : 256;
    const int B = (argc > 3) ? std::atoi(argv[3]) : 1;
    const int H = (argc > 4) ? std::atoi(argv[4]) : 2;
    std::string dump_dir;
    for (int i = 5; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--dump") dump_dir = argv[i + 1];
    }

    constexpr int CHUNK = 32;
    constexpr int D_QK = 64;
    constexpr int D_V = 64;
    if (L % CHUNK != 0) {
        std::fprintf(stderr, "seq_len must be a multiple of %d\n", CHUNK);
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
        std::fprintf(stderr, "Failed to load metallib at %s: %s\n", metallib_path,
                     err ? err->localizedDescription()->utf8String() : "(unknown)");
        return 1;
    }

    const char* name = "mamba2_siso_fwd_32_64_64";
    auto* fn = lib->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
    if (!fn) {
        std::fprintf(stderr, "Kernel '%s' not found\n", name);
        return 1;
    }
    auto* pso = device->newComputePipelineState(fn, &err);
    fn->release();
    if (!pso) {
        std::fprintf(stderr, "PSO creation failed: %s\n",
                     err ? err->localizedDescription()->utf8String() : "(no error)");
        return 1;
    }

    const size_t qk_elems = (size_t)B * H * L * D_QK;
    const size_t v_elems = (size_t)B * H * L * D_V;
    const size_t a_elems = (size_t)B * H * L;

    std::vector<__fp16> Q(qk_elems), K(qk_elems), V(v_elems);
    std::vector<float> A(a_elems);
    fill_inputs(Q, K, V, A, 42);

    std::printf("== mamba2 v1 smoke test ==\n");
    std::printf("  B=%d H=%d L=%d D_qk=%d D_v=%d CHUNK=%d\n", B, H, L, D_QK, D_V, CHUNK);

    auto cpu_t0 = std::chrono::steady_clock::now();
    std::vector<float> ref;
    cpu_reference(Q, K, V, A, ref, B, H, L, D_QK, D_V);
    auto cpu_t1 = std::chrono::steady_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();

    auto mode = MTL::ResourceStorageModeShared;
    auto* bQ = device->newBuffer(qk_elems * sizeof(__fp16), mode);
    auto* bK = device->newBuffer(qk_elems * sizeof(__fp16), mode);
    auto* bV = device->newBuffer(v_elems * sizeof(__fp16), mode);
    auto* bA = device->newBuffer(a_elems * sizeof(float), mode);
    auto* bO = device->newBuffer(v_elems * sizeof(__fp16), mode);
    std::memcpy(bQ->contents(), Q.data(), qk_elems * sizeof(__fp16));
    std::memcpy(bK->contents(), K.data(), qk_elems * sizeof(__fp16));
    std::memcpy(bV->contents(), V.data(), v_elems * sizeof(__fp16));
    std::memcpy(bA->contents(), A.data(), a_elems * sizeof(float));

    Mamba2FwdArgsHost args{(uint32_t)B, (uint32_t)H, (uint32_t)L, (uint32_t)(L / CHUNK)};
    auto* bArgs = device->newBuffer(&args, sizeof(args), mode);

    auto run = [&]() -> double {
        std::memset(bO->contents(), 0, v_elems * sizeof(__fp16));
        auto* cmd = queue->commandBuffer();
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(bQ, 0, 0);
        enc->setBuffer(bK, 0, 1);
        enc->setBuffer(bV, 0, 2);
        enc->setBuffer(bA, 0, 3);
        enc->setBuffer(bO, 0, 4);
        enc->setBuffer(bArgs, 0, 5);
        enc->dispatchThreadgroups(MTL::Size(B, H, 1), MTL::Size(128, 1, 1));
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
        return (cmd->GPUEndTime() - cmd->GPUStartTime()) * 1000.0;
    };

    for (int i = 0; i < 10; ++i) run();
    std::vector<double> ts;
    for (int i = 0; i < 50; ++i) ts.push_back(run());
    std::sort(ts.begin(), ts.end());
    const double gpu_ms = ts[ts.size() / 2];

    std::vector<__fp16> gpu(v_elems);
    std::memcpy(gpu.data(), bO->contents(), v_elems * sizeof(__fp16));

    auto st = compare_hf(gpu, ref);
    std::printf("  CPU ref:    %.3f ms\n", cpu_ms);
    std::printf("  GPU median: %.3f ms (%zu iters)\n", gpu_ms, ts.size());
    std::printf("  speedup:    %.2fx\n", cpu_ms / gpu_ms);
    std::printf("  accuracy:   max_abs=%.5f mean_abs=%.5f l2_rel=%.5f\n",
                st.max_abs, st.mean_abs, st.l2_rel);
    const bool pass = (st.l2_rel < 5e-2f);
    std::printf("  %s\n", pass ? "PASS" : "FAIL");

    if (!dump_dir.empty()) {
        dump_bin(dump_dir + "/q.bin", Q);
        dump_bin(dump_dir + "/k.bin", K);
        dump_bin(dump_dir + "/v.bin", V);
        dump_bin(dump_dir + "/a.bin", A);
        dump_bin(dump_dir + "/y_gpu.bin", gpu);
        dump_bin(dump_dir + "/y_cpu.bin", ref);
        std::ofstream meta(dump_dir + "/meta.txt");
        meta << "B=" << B << " H=" << H << " L=" << L
             << " D_qk=" << D_QK << " D_v=" << D_V << " CHUNK=" << CHUNK << "\n";
    }

    bQ->release();
    bK->release();
    bV->release();
    bA->release();
    bO->release();
    bArgs->release();
    pso->release();
    lib->release();
    queue->release();
    device->release();
    return pass ? 0 : 1;
}
