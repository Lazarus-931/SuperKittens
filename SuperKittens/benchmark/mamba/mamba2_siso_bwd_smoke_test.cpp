//
//  mamba2_siso_bwd_smoke_test.cpp
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

struct Mamba2BwdArgsHost {
    uint32_t batch;
    uint32_t nheads;
    uint32_t seq_len;
    uint32_t n_chunks;
};

struct ForwardCache {
    std::vector<float> out;
    std::vector<float> states;
};

struct BackwardGrads {
    std::vector<float> dQ;
    std::vector<float> dK;
    std::vector<float> dV;
    std::vector<float> dA;
};

struct ErrStats {
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    float l2_rel = 0.0f;
};

static size_t qk_index(int b, int h, int t, int d, int H, int L, int D) {
    return ((((size_t)b * H + h) * L) + t) * D + d;
}

static size_t v_index(int b, int h, int t, int d, int H, int L, int D) {
    return ((((size_t)b * H + h) * L) + t) * D + d;
}

static size_t scalar_index(int b, int h, int t, int H, int L) {
    return (((size_t)b * H + h) * L) + t;
}

static size_t state_index(int b, int h, int t, int i, int j, int H, int L, int DQ, int DV) {
    return ((((((size_t)b * H + h) * L) + t) * DQ) + i) * DV + j;
}

static void fill_half(std::vector<__fp16>& v, std::mt19937& rng, float scale) {
    std::normal_distribution<float> n(0.0f, 1.0f);
    for (auto& x : v) x = __fp16(n(rng) * scale);
}

static void fill_float(std::vector<float>& v, std::mt19937& rng, float mean, float stddev) {
    std::normal_distribution<float> n(mean, stddev);
    for (auto& x : v) x = n(rng);
}

static ErrStats compare_float(const std::vector<float>& gpu,
                              const std::vector<float>& ref) {
    ErrStats s;
    double sum_abs = 0.0;
    double sum_num = 0.0;
    double sum_den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float d = std::fabs(gpu[i] - ref[i]);
        s.max_abs = std::max(s.max_abs, d);
        sum_abs += d;
        sum_num += double(d) * double(d);
        sum_den += double(ref[i]) * double(ref[i]);
    }
    s.mean_abs = float(sum_abs / ref.size());
    s.l2_rel = float(std::sqrt(sum_num) / (std::sqrt(sum_den) + 1e-12));
    return s;
}

static ForwardCache cpu_forward(
    const std::vector<__fp16>& Q,
    const std::vector<__fp16>& K,
    const std::vector<__fp16>& V,
    const std::vector<float>& A,
    int batch, int heads, int length, int DQ, int DV) {
    ForwardCache cache;
    cache.out.assign((size_t)batch * heads * length * DV, 0.0f);
    cache.states.assign((size_t)batch * heads * length * DQ * DV, 0.0f);
    std::vector<float> state((size_t)DQ * DV, 0.0f);

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            std::fill(state.begin(), state.end(), 0.0f);
            for (int t = 0; t < length; ++t) {
                const float decay = std::exp(A[scalar_index(b, h, t, heads, length)]);
                for (int i = 0; i < DQ; ++i) {
                    const float kval = float(K[qk_index(b, h, t, i, heads, length, DQ)]);
                    for (int j = 0; j < DV; ++j) {
                        const float vval = float(V[v_index(b, h, t, j, heads, length, DV)]);
                        float& s = state[(size_t)i * DV + j];
                        s = decay * s + kval * vval;
                        cache.states[state_index(b, h, t, i, j, heads, length, DQ, DV)] = s;
                    }
                }

                for (int j = 0; j < DV; ++j) {
                    float acc = 0.0f;
                    for (int i = 0; i < DQ; ++i) {
                        acc += float(Q[qk_index(b, h, t, i, heads, length, DQ)]) * state[(size_t)i * DV + j];
                    }
                    cache.out[v_index(b, h, t, j, heads, length, DV)] = acc;
                }
            }
        }
    }
    return cache;
}

static BackwardGrads cpu_backward(
    const std::vector<__fp16>& Q,
    const std::vector<__fp16>& K,
    const std::vector<__fp16>& V,
    const std::vector<float>& A,
    const std::vector<float>& states,
    const std::vector<float>& dO,
    int batch, int heads, int length, int DQ, int DV) {
    BackwardGrads g;
    g.dQ.assign((size_t)batch * heads * length * DQ, 0.0f);
    g.dK.assign((size_t)batch * heads * length * DQ, 0.0f);
    g.dV.assign((size_t)batch * heads * length * DV, 0.0f);
    g.dA.assign((size_t)batch * heads * length, 0.0f);

    std::vector<float> ds((size_t)DQ * DV, 0.0f);
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            std::fill(ds.begin(), ds.end(), 0.0f);
            for (int t = length - 1; t >= 0; --t) {
                for (int i = 0; i < DQ; ++i) {
                    float dq_acc = 0.0f;
                    const float qval = float(Q[qk_index(b, h, t, i, heads, length, DQ)]);
                    for (int j = 0; j < DV; ++j) {
                        const float dout = dO[v_index(b, h, t, j, heads, length, DV)];
                        dq_acc += dout * states[state_index(b, h, t, i, j, heads, length, DQ, DV)];
                        ds[(size_t)i * DV + j] += qval * dout;
                    }
                    g.dQ[qk_index(b, h, t, i, heads, length, DQ)] = dq_acc;
                }

                for (int i = 0; i < DQ; ++i) {
                    float dk_acc = 0.0f;
                    const float kval = float(K[qk_index(b, h, t, i, heads, length, DQ)]);
                    for (int j = 0; j < DV; ++j) {
                        const float vval = float(V[v_index(b, h, t, j, heads, length, DV)]);
                        dk_acc += ds[(size_t)i * DV + j] * vval;
                        g.dV[v_index(b, h, t, j, heads, length, DV)] += kval * ds[(size_t)i * DV + j];
                    }
                    g.dK[qk_index(b, h, t, i, heads, length, DQ)] = dk_acc;
                }

                const float decay = std::exp(A[scalar_index(b, h, t, heads, length)]);
                if (t > 0) {
                    float da_acc = 0.0f;
                    for (int i = 0; i < DQ; ++i) {
                        for (int j = 0; j < DV; ++j) {
                            const float s_prev = states[state_index(b, h, t - 1, i, j, heads, length, DQ, DV)];
                            da_acc += ds[(size_t)i * DV + j] * s_prev;
                            ds[(size_t)i * DV + j] *= decay;
                        }
                    }
                    g.dA[scalar_index(b, h, t, heads, length)] = decay * da_acc;
                } else {
                    g.dA[scalar_index(b, h, t, heads, length)] = 0.0f;
                }
            }
        }
    }
    return g;
}

static double run_encoder(MTL::CommandQueue* queue,
                          MTL::ComputePipelineState* pso,
                          std::initializer_list<MTL::Buffer*> buffers,
                          MTL::Size tg,
                          MTL::Size threads) {
    auto* cmd = queue->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    uint32_t idx = 0;
    for (auto* buf : buffers) enc->setBuffer(buf, 0, idx++);
    enc->dispatchThreadgroups(tg, threads);
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    return (cmd->GPUEndTime() - cmd->GPUStartTime()) * 1000.0;
}

static double run_three_stage(MTL::CommandQueue* queue,
                              MTL::ComputePipelineState* pso_pre,
                              MTL::ComputePipelineState* pso_scan,
                              MTL::ComputePipelineState* pso_fin,
                              std::initializer_list<MTL::Buffer*> pre_buffers,
                              std::initializer_list<MTL::Buffer*> scan_buffers,
                              std::initializer_list<MTL::Buffer*> fin_buffers,
                              MTL::Size tg_chunks,
                              MTL::Size tg_heads,
                              MTL::Size threads) {
    auto* cmd = queue->commandBuffer();
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso_pre);
        uint32_t idx = 0;
        for (auto* buf : pre_buffers) enc->setBuffer(buf, 0, idx++);
        enc->dispatchThreadgroups(tg_chunks, threads);
        enc->endEncoding();
    }
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso_scan);
        uint32_t idx = 0;
        for (auto* buf : scan_buffers) enc->setBuffer(buf, 0, idx++);
        enc->dispatchThreadgroups(tg_heads, threads);
        enc->endEncoding();
    }
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso_fin);
        uint32_t idx = 0;
        for (auto* buf : fin_buffers) enc->setBuffer(buf, 0, idx++);
        enc->dispatchThreadgroups(tg_chunks, threads);
        enc->endEncoding();
    }
    cmd->commit();
    cmd->waitUntilCompleted();
    return (cmd->GPUEndTime() - cmd->GPUStartTime()) * 1000.0;
}

} // namespace

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <metallib> [L] [B] [H]\n", argv[0]);
        return 1;
    }
    const char* metallib_path = argv[1];
    const int L = (argc > 2) ? std::atoi(argv[2]) : 256;
    const int B = (argc > 3) ? std::atoi(argv[3]) : 1;
    const int H = (argc > 4) ? std::atoi(argv[4]) : 2;

    constexpr int DQ = 64;
    constexpr int DV = 64;

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

    auto* fn_pre = lib->newFunction(NS::String::string("mamba2_siso_bwd_precompute_32_64_64", NS::UTF8StringEncoding));
    auto* fn_scan = lib->newFunction(NS::String::string("mamba2_siso_bwd_scan_32_64_64", NS::UTF8StringEncoding));
    auto* fn_fin = lib->newFunction(NS::String::string("mamba2_siso_bwd_finalize_32_64_64", NS::UTF8StringEncoding));
    if (!fn_pre || !fn_scan || !fn_fin) {
        std::fprintf(stderr, "One or more M2 backward kernels are missing\n");
        return 1;
    }
    auto* pso_pre = device->newComputePipelineState(fn_pre, &err);
    auto* pso_scan = device->newComputePipelineState(fn_scan, &err);
    auto* pso_fin = device->newComputePipelineState(fn_fin, &err);
    fn_pre->release();
    fn_scan->release();
    fn_fin->release();
    if (!pso_pre || !pso_scan || !pso_fin) {
        std::fprintf(stderr, "PSO creation failed: %s\n",
                     err ? err->localizedDescription()->utf8String() : "(no error)");
        return 1;
    }

    const size_t qk_elems = (size_t)B * H * L * DQ;
    const size_t v_elems = (size_t)B * H * L * DV;
    const size_t a_elems = (size_t)B * H * L;
    const size_t state_elems = (size_t)B * H * L * DQ * DV;

    std::mt19937 rng(42);
    std::vector<__fp16> Q(qk_elems), K(qk_elems), V(v_elems);
    std::vector<float> A(a_elems), dO(v_elems);
    fill_half(Q, rng, 0.5f);
    fill_half(K, rng, 0.5f);
    fill_half(V, rng, 0.5f);
    fill_float(A, rng, -0.1f, 0.2f);
    fill_float(dO, rng, 0.0f, 0.5f);

    ForwardCache ref_fwd = cpu_forward(Q, K, V, A, B, H, L, DQ, DV);
    auto cpu_t0 = std::chrono::steady_clock::now();
    BackwardGrads ref_bwd = cpu_backward(Q, K, V, A, ref_fwd.states, dO, B, H, L, DQ, DV);
    auto cpu_t1 = std::chrono::steady_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();

    auto mode = MTL::ResourceStorageModeShared;
    auto* bufQ = device->newBuffer(qk_elems * sizeof(__fp16), mode);
    auto* bufK = device->newBuffer(qk_elems * sizeof(__fp16), mode);
    auto* bufV = device->newBuffer(v_elems * sizeof(__fp16), mode);
    auto* bufA = device->newBuffer(a_elems * sizeof(float), mode);
    auto* bufStates = device->newBuffer(state_elems * sizeof(float), mode);
    auto* bufDO = device->newBuffer(v_elems * sizeof(float), mode);
    auto* bufDQ = device->newBuffer(qk_elems * sizeof(float), mode);
    auto* bufDK = device->newBuffer(qk_elems * sizeof(float), mode);
    auto* bufDV = device->newBuffer(v_elems * sizeof(float), mode);
    auto* bufDA = device->newBuffer(a_elems * sizeof(float), mode);
    auto* bufLocalDK = device->newBuffer(qk_elems * sizeof(float), mode);
    auto* bufLocalDV = device->newBuffer(v_elems * sizeof(float), mode);
    auto* bufLocalDA = device->newBuffer(a_elems * sizeof(float), mode);
    auto* bufSuffix = device->newBuffer(a_elems * sizeof(float), mode);
    auto* bufChunkDecay = device->newBuffer((size_t)B * H * (L / 32) * sizeof(float), mode);
    auto* bufChunkU = device->newBuffer((size_t)B * H * (L / 32) * DQ * DV * sizeof(float), mode);
    auto* bufChunkCarry = device->newBuffer((size_t)B * H * (L / 32) * DQ * DV * sizeof(float), mode);

    std::memcpy(bufQ->contents(), Q.data(), qk_elems * sizeof(__fp16));
    std::memcpy(bufK->contents(), K.data(), qk_elems * sizeof(__fp16));
    std::memcpy(bufV->contents(), V.data(), v_elems * sizeof(__fp16));
    std::memcpy(bufA->contents(), A.data(), a_elems * sizeof(float));
    std::memcpy(bufStates->contents(), ref_fwd.states.data(), state_elems * sizeof(float));
    std::memcpy(bufDO->contents(), dO.data(), v_elems * sizeof(float));

    Mamba2BwdArgsHost args{(uint32_t)B, (uint32_t)H, (uint32_t)L, (uint32_t)(L / 32)};
    auto* bufArgs = device->newBuffer(&args, sizeof(args), mode);

    auto zero_grads = [&]() {
        std::memset(bufDQ->contents(), 0, qk_elems * sizeof(float));
        std::memset(bufDK->contents(), 0, qk_elems * sizeof(float));
        std::memset(bufDV->contents(), 0, v_elems * sizeof(float));
        std::memset(bufDA->contents(), 0, a_elems * sizeof(float));
        std::memset(bufLocalDK->contents(), 0, qk_elems * sizeof(float));
        std::memset(bufLocalDV->contents(), 0, v_elems * sizeof(float));
        std::memset(bufLocalDA->contents(), 0, a_elems * sizeof(float));
        std::memset(bufSuffix->contents(), 0, a_elems * sizeof(float));
        std::memset(bufChunkDecay->contents(), 0, (size_t)B * H * (L / 32) * sizeof(float));
        std::memset(bufChunkU->contents(), 0, (size_t)B * H * (L / 32) * DQ * DV * sizeof(float));
        std::memset(bufChunkCarry->contents(), 0, (size_t)B * H * (L / 32) * DQ * DV * sizeof(float));
    };

    struct StageTimes { double pre=0.0, scan=0.0, fin=0.0; };
    auto run = [&](StageTimes* stages = nullptr) -> double {
        zero_grads();
        const auto pre_buffers = std::initializer_list<MTL::Buffer*>{
            bufQ, bufK, bufV, bufA, bufStates, bufDO, bufDQ,
            bufLocalDK, bufLocalDV, bufLocalDA, bufSuffix,
            bufChunkDecay, bufChunkU, bufArgs
        };
        const auto scan_buffers = std::initializer_list<MTL::Buffer*>{
            bufChunkDecay, bufChunkU, bufChunkCarry, bufArgs
        };
        const auto fin_buffers = std::initializer_list<MTL::Buffer*>{
            bufK, bufV, bufA, bufStates, bufLocalDK, bufLocalDV,
            bufLocalDA, bufSuffix, bufChunkCarry, bufDK, bufDV,
            bufDA, bufArgs
        };
        const MTL::Size tg_chunks(B, H, L / 32);
        const MTL::Size tg_heads(B, H, 1);
        const MTL::Size threads(128, 1, 1);
        const double total = run_three_stage(queue, pso_pre, pso_scan, pso_fin,
                                             pre_buffers, scan_buffers, fin_buffers,
                                             tg_chunks, tg_heads, threads);
        if (stages) {
            stages->pre = run_encoder(queue, pso_pre, pre_buffers, tg_chunks, threads);
            stages->scan = run_encoder(queue, pso_scan, scan_buffers, tg_heads, threads);
            stages->fin = run_encoder(queue, pso_fin, fin_buffers, tg_chunks, threads);
        }
        return total;
    };

    for (int i = 0; i < 10; ++i) run();
    std::vector<double> ts;
    for (int i = 0; i < 50; ++i) ts.push_back(run());
    std::sort(ts.begin(), ts.end());
    const double gpu_ms = ts[ts.size() / 2];
    StageTimes one_run;
    run(&one_run);

    std::vector<float> gpuDQ(qk_elems), gpuDK(qk_elems), gpuDV(v_elems), gpuDA(a_elems);
    std::memcpy(gpuDQ.data(), bufDQ->contents(), qk_elems * sizeof(float));
    std::memcpy(gpuDK.data(), bufDK->contents(), qk_elems * sizeof(float));
    std::memcpy(gpuDV.data(), bufDV->contents(), v_elems * sizeof(float));
    std::memcpy(gpuDA.data(), bufDA->contents(), a_elems * sizeof(float));

    ErrStats dq_err = compare_float(gpuDQ, ref_bwd.dQ);
    ErrStats dk_err = compare_float(gpuDK, ref_bwd.dK);
    ErrStats dv_err = compare_float(gpuDV, ref_bwd.dV);
    ErrStats da_err = compare_float(gpuDA, ref_bwd.dA);

    std::printf("== mamba2 siso backward ==\n");
    std::printf("  B=%d H=%d L=%d DQ=%d DV=%d\n", B, H, L, DQ, DV);
    std::printf("  CPU bwd:      %.3f ms\n", cpu_ms);
    std::printf("  Metal bwd:    %.3f ms\n", gpu_ms);
    std::printf("    precompute: %.3f ms\n", one_run.pre);
    std::printf("    scan:       %.3f ms\n", one_run.scan);
    std::printf("    finalize:   %.3f ms\n", one_run.fin);
    std::printf("  speedup:      %.2fx\n", cpu_ms / gpu_ms);
    std::printf("  dQ l2_rel=%.6f max_abs=%.6f\n", dq_err.l2_rel, dq_err.max_abs);
    std::printf("  dK l2_rel=%.6f max_abs=%.6f\n", dk_err.l2_rel, dk_err.max_abs);
    std::printf("  dV l2_rel=%.6f max_abs=%.6f\n", dv_err.l2_rel, dv_err.max_abs);
    std::printf("  dA l2_rel=%.6f max_abs=%.6f\n", da_err.l2_rel, da_err.max_abs);

    bufQ->release();
    bufK->release();
    bufV->release();
    bufA->release();
    bufStates->release();
    bufDO->release();
    bufDQ->release();
    bufDK->release();
    bufDV->release();
    bufDA->release();
    bufLocalDK->release();
    bufLocalDV->release();
    bufLocalDA->release();
    bufSuffix->release();
    bufChunkDecay->release();
    bufChunkU->release();
    bufChunkCarry->release();
    bufArgs->release();
    pso_pre->release();
    pso_scan->release();
    pso_fin->release();
    lib->release();
    queue->release();
    device->release();
    return 0;
}
