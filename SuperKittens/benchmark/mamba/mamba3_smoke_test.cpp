//
//  mamba3_smoke_test.cpp
//  SuperKittens — CPU vs GPU smoke benchmark for Mamba-3 SISO/MIMO.
//
//  Build:
//    clang++ -std=gnu++20 -O2 -arch arm64 -I metal-cpp \
//      -framework Metal -framework Foundation -framework QuartzCore \
//      SuperKittens/benchmark/mamba/mamba3_smoke_test.cpp -o build/mamba3_smoke_test
//
//  Run:
//    ./build/mamba3_smoke_test <metallib> <siso|mimo> [L=256] [B=1] [H=2]
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

struct Mamba3FwdArgsHost {
    uint32_t batch;
    uint32_t nheads;
    uint32_t seq_len;
    uint32_t n_chunks;
};

struct ErrStats {
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    float l2_rel = 0.0f;
};

static void fill_half(std::vector<__fp16>& v, std::mt19937& rng, float scale) {
    std::normal_distribution<float> n(0.0f, 1.0f);
    for (auto& x : v) x = __fp16(n(rng) * scale);
}

static void fill_float(std::vector<float>& v, std::mt19937& rng, float mean, float stddev) {
    std::normal_distribution<float> n(mean, stddev);
    for (auto& x : v) x = n(rng);
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
        if (d > s.max_abs) s.max_abs = d;
        sum_abs += d;
        sum_num += double(d) * double(d);
        sum_den += double(r) * double(r);
    }
    s.mean_abs = float(sum_abs / ref.size());
    s.l2_rel = float(std::sqrt(sum_num) / (std::sqrt(sum_den) + 1e-12));
    return s;
}

static void cpu_reference_mamba3(
    const std::vector<__fp16>& Q,
    const std::vector<__fp16>& K,
    const std::vector<__fp16>& V,
    const std::vector<float>& A,
    const std::vector<float>& B,
    const std::vector<__fp16>& angle,
    std::vector<float>& Yout,
    int batch, int heads, int length,
    int base_dim_qk, int rank_blocks, int dim_v, int chunk)
{
    const int dim_qk = base_dim_qk * rank_blocks;
    Yout.assign((size_t)batch * heads * length * dim_v, 0.0f);
    std::vector<float> state((size_t)dim_qk * dim_v, 0.0f);
    std::vector<float> q_rot((size_t)chunk * dim_qk, 0.0f);
    std::vector<float> k_rot((size_t)chunk * dim_qk, 0.0f);
    std::vector<float> a_cs(chunk, 0.0f);
    std::vector<float> b_scale(chunk, 0.0f);

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            std::fill(state.begin(), state.end(), 0.0f);
            const size_t qk_base = ((size_t)b * heads + h) * length * dim_qk;
            const size_t v_base = ((size_t)b * heads + h) * length * dim_v;
            const size_t a_base = ((size_t)b * heads + h) * length;
            const size_t angle_base = ((size_t)b * heads + h) * length * (base_dim_qk / 2);

            for (int chunk_start = 0; chunk_start < length; chunk_start += chunk) {
                const int chunk_len = std::min(chunk, length - chunk_start);

                float running = 0.0f;
                for (int t = 0; t < chunk_len; ++t) {
                    running += A[a_base + chunk_start + t];
                    a_cs[t] = running;
                    b_scale[t] = 1.0f + B[a_base + chunk_start + t] * std::exp(-a_cs[t]);
                }

                for (int t = 0; t < chunk_len; ++t) {
                    for (int r = 0; r < rank_blocks; ++r) {
                        const int base = r * base_dim_qk;
                        const int half = base_dim_qk / 2;
                        for (int p = 0; p < half; ++p) {
                            const float raw_angle = float(
                                angle[angle_base + (size_t)(chunk_start + t) * half + p]);
                            const float a = a_cs[t] * raw_angle * 3.141592653589793f;
                            const float c = std::cos(a);
                            const float s = std::sin(a);

                            const size_t q_idx0 = qk_base + (size_t)(chunk_start + t) * dim_qk + base + p;
                            const size_t q_idx1 = q_idx0 + half;
                            const float q0 = float(Q[q_idx0]);
                            const float q1 = float(Q[q_idx1]);
                            q_rot[(size_t)t * dim_qk + base + p] = q0 * c - q1 * s;
                            q_rot[(size_t)t * dim_qk + base + p + half] = q0 * s + q1 * c;

                            const float k0 = float(K[q_idx0]);
                            const float k1 = float(K[q_idx1]);
                            k_rot[(size_t)t * dim_qk + base + p] = k0 * c - k1 * s;
                            k_rot[(size_t)t * dim_qk + base + p + half] = k0 * s + k1 * c;
                        }
                    }
                }

                const float chunk_decay = std::exp(a_cs[chunk_len - 1]) * b_scale[chunk_len - 1];
                for (size_t i = 0; i < state.size(); ++i)
                    state[i] *= chunk_decay;

                for (int t = 0; t < chunk_len; ++t) {
                    for (int i = 0; i < dim_qk; ++i) {
                        const float k_i = k_rot[(size_t)t * dim_qk + i];
                        for (int j = 0; j < dim_v; ++j) {
                            state[(size_t)i * dim_v + j] +=
                                k_i * float(V[v_base + (size_t)(chunk_start + t) * dim_v + j]);
                        }
                    }
                }

                for (int r = 0; r < chunk_len; ++r) {
                    std::vector<float> y(dim_v, 0.0f);

                    for (int c = 0; c <= r; ++c) {
                        float score = 0.0f;
                        for (int i = 0; i < dim_qk; ++i)
                            score += q_rot[(size_t)r * dim_qk + i] * k_rot[(size_t)c * dim_qk + i];
                        const float decay = std::exp(a_cs[r] - a_cs[c]);
                        for (int j = 0; j < dim_v; ++j) {
                            y[j] += score * decay * float(V[v_base + (size_t)(chunk_start + c) * dim_v + j]);
                        }
                    }

                    const float q_decay = std::exp(a_cs[r]) * b_scale[r];
                    for (int j = 0; j < dim_v; ++j) {
                        float acc = 0.0f;
                        for (int i = 0; i < dim_qk; ++i)
                            acc += q_rot[(size_t)r * dim_qk + i] * state[(size_t)i * dim_v + j];
                        y[j] += q_decay * acc;
                    }

                    for (int j = 0; j < dim_v; ++j)
                        Yout[v_base + (size_t)(chunk_start + r) * dim_v + j] = y[j];
                }
            }
        }
    }
}

} // namespace

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <metallib> <siso|mimo> [L] [B] [H]\n", argv[0]);
        return 1;
    }

    const char* metallib_path = argv[1];
    const std::string mode = argv[2];
    const int L = (argc > 3) ? std::atoi(argv[3]) : 256;
    const int B = (argc > 4) ? std::atoi(argv[4]) : 1;
    const int H = (argc > 5) ? std::atoi(argv[5]) : 2;

    const bool is_mimo = (mode == "mimo");
    const int CHUNK = 32;
    const int BASE_D_QK = 64;
    const int RANK = is_mimo ? 2 : 1;
    const int D_QK = BASE_D_QK * RANK;
    const int D_V = 64;

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

    const char* kernel = is_mimo ? "mamba3_mimo_fwd_32_64_64_r2" : "mamba3_siso_fwd_32_64_64";
    auto* fn = lib->newFunction(NS::String::string(kernel, NS::UTF8StringEncoding));
    if (!fn) {
        std::fprintf(stderr, "Kernel '%s' not found\n", kernel);
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
    const size_t angle_elems = (size_t)B * H * L * (BASE_D_QK / 2);

    std::mt19937 rng(42);
    std::vector<__fp16> Q(qk_elems), K(qk_elems), V(v_elems), angle(angle_elems);
    std::vector<float> A(a_elems), Btrap(a_elems);
    fill_half(Q, rng, 0.5f);
    fill_half(K, rng, 0.5f);
    fill_half(V, rng, 0.5f);
    fill_half(angle, rng, 0.1f);
    fill_float(A, rng, -0.1f, 0.2f);
    fill_float(Btrap, rng, 0.0f, 0.2f);

    std::printf("== mamba3 smoke test (%s) ==\n", is_mimo ? "mimo" : "siso");
    std::printf("  B=%d H=%d L=%d D_qk=%d D_v=%d CHUNK=%d RANK=%d\n", B, H, L, D_QK, D_V, CHUNK, RANK);

    const auto cpu_t0 = std::chrono::steady_clock::now();
    std::vector<float> ref;
    cpu_reference_mamba3(Q, K, V, A, Btrap, angle, ref, B, H, L, BASE_D_QK, RANK, D_V, CHUNK);
    const auto cpu_t1 = std::chrono::steady_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();

    const auto storage = MTL::ResourceStorageModeShared;
    auto* bQ = device->newBuffer(qk_elems * sizeof(__fp16), storage);
    auto* bK = device->newBuffer(qk_elems * sizeof(__fp16), storage);
    auto* bV = device->newBuffer(v_elems * sizeof(__fp16), storage);
    auto* bA = device->newBuffer(a_elems * sizeof(float), storage);
    auto* bB = device->newBuffer(a_elems * sizeof(float), storage);
    auto* bAngle = device->newBuffer(angle_elems * sizeof(__fp16), storage);
    auto* bO = device->newBuffer(v_elems * sizeof(__fp16), storage);
    std::memcpy(bQ->contents(), Q.data(), qk_elems * sizeof(__fp16));
    std::memcpy(bK->contents(), K.data(), qk_elems * sizeof(__fp16));
    std::memcpy(bV->contents(), V.data(), v_elems * sizeof(__fp16));
    std::memcpy(bA->contents(), A.data(), a_elems * sizeof(float));
    std::memcpy(bB->contents(), Btrap.data(), a_elems * sizeof(float));
    std::memcpy(bAngle->contents(), angle.data(), angle_elems * sizeof(__fp16));

    Mamba3FwdArgsHost args{(uint32_t)B, (uint32_t)H, (uint32_t)L, (uint32_t)(L / CHUNK)};
    auto* bArgs = device->newBuffer(&args, sizeof(args), storage);

    auto run = [&]() -> double {
        std::memset(bO->contents(), 0, v_elems * sizeof(__fp16));
        auto* cmd = queue->commandBuffer();
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(bQ, 0, 0);
        enc->setBuffer(bK, 0, 1);
        enc->setBuffer(bV, 0, 2);
        enc->setBuffer(bA, 0, 3);
        enc->setBuffer(bB, 0, 4);
        enc->setBuffer(bAngle, 0, 5);
        enc->setBuffer(bO, 0, 6);
        enc->setBuffer(bArgs, 0, 7);
        enc->dispatchThreadgroups(MTL::Size(B, H, 1), MTL::Size(256, 1, 1));
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

    std::vector<__fp16> gpu(v_elems);
    std::memcpy(gpu.data(), bO->contents(), v_elems * sizeof(__fp16));
    const ErrStats st = compare_hf(gpu, ref);

    std::printf("  CPU ref:    %.3f ms\n", cpu_ms);
    std::printf("  GPU median: %.3f ms (%zu iters)\n", gpu_ms, ts.size());
    std::printf("  speedup:    %.2fx\n", cpu_ms / gpu_ms);
    std::printf("  accuracy:   max_abs=%.5f mean_abs=%.5f l2_rel=%.5f\n",
                st.max_abs, st.mean_abs, st.l2_rel);
    const bool pass = std::isfinite(st.l2_rel) && st.l2_rel < 5e-2f;
    std::printf("  %s\n", pass ? "PASS" : "FAIL");

    bQ->release();
    bK->release();
    bV->release();
    bA->release();
    bB->release();
    bAngle->release();
    bO->release();
    bArgs->release();
    pso->release();
    lib->release();
    queue->release();
    device->release();
    return pass ? 0 : 1;
}
