//
//  mamba2_parallel_bench.cpp
//  SuperKittens
//
//  Compare baseline mamba2.metal against the experimental chunked-parallel path.
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

constexpr int CHUNK = 32;
constexpr int D_QK = 64;
constexpr int D_V = 64;
constexpr int STATE_SIZE = D_QK * D_V;

struct Mamba2ArgsHost {
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
                          int B, int H, int L) {
    Yout.assign((size_t)B * H * L * D_V, 0.0f);
    std::vector<float> state((size_t)STATE_SIZE, 0.0f);

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            std::fill(state.begin(), state.end(), 0.0f);
            const size_t qk_base = ((size_t)b * H + h) * L * D_QK;
            const size_t v_base = ((size_t)b * H + h) * L * D_V;
            const size_t a_base = ((size_t)b * H + h) * L;

            for (int t = 0; t < L; ++t) {
                const float decay = std::exp(A[a_base + t]);
                for (int i = 0; i < D_QK; ++i) {
                    const float k_i = float(K[qk_base + (size_t)t * D_QK + i]);
                    for (int j = 0; j < D_V; ++j) {
                        float& s = state[(size_t)i * D_V + j];
                        s = decay * s + k_i * float(V[v_base + (size_t)t * D_V + j]);
                    }
                }
                for (int j = 0; j < D_V; ++j) {
                    float acc = 0.0f;
                    for (int i = 0; i < D_QK; ++i)
                        acc += float(Q[qk_base + (size_t)t * D_QK + i]) * state[(size_t)i * D_V + j];
                    Yout[v_base + (size_t)t * D_V + j] = acc;
                }
            }
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
        if (d > s.max_abs) s.max_abs = d;
        sum_abs += d;
        sum_num += double(d) * double(d);
        sum_den += double(r) * double(r);
    }
    s.mean_abs = float(sum_abs / ref.size());
    s.l2_rel = float(std::sqrt(sum_num) / (std::sqrt(sum_den) + 1e-12));
    return s;
}

static MTL::ComputePipelineState* make_pso(MTL::Device* device, MTL::Library* lib, const char* name) {
    NS::Error* err = nullptr;
    auto* fn = lib->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
    if (!fn) return nullptr;
    auto* pso = device->newComputePipelineState(fn, &err);
    fn->release();
    return pso;
}

struct SharedBuffers {
    MTL::Buffer* Q = nullptr;
    MTL::Buffer* K = nullptr;
    MTL::Buffer* V = nullptr;
    MTL::Buffer* A = nullptr;
    MTL::Buffer* O = nullptr;
    MTL::Buffer* args = nullptr;

    void release() {
        if (Q) Q->release();
        if (K) K->release();
        if (V) V->release();
        if (A) A->release();
        if (O) O->release();
        if (args) args->release();
    }
};

struct ParallelBuffers {
    MTL::Buffer* local_out = nullptr;
    MTL::Buffer* q_scaled = nullptr;
    MTL::Buffer* chunk_decay = nullptr;
    MTL::Buffer* chunk_update = nullptr;

    void release() {
        if (local_out) local_out->release();
        if (q_scaled) q_scaled->release();
        if (chunk_decay) chunk_decay->release();
        if (chunk_update) chunk_update->release();
    }
};

static SharedBuffers allocate_shared(MTL::Device* device, int B, int H, int L,
                                     const std::vector<__fp16>& Qv,
                                     const std::vector<__fp16>& Kv,
                                     const std::vector<__fp16>& Vv,
                                     const std::vector<float>& Av) {
    SharedBuffers bufs;
    const size_t qk_elems = (size_t)B * H * L * D_QK;
    const size_t v_elems = (size_t)B * H * L * D_V;
    const size_t a_elems = (size_t)B * H * L;
    const auto mode = MTL::ResourceStorageModeShared;

    bufs.Q = device->newBuffer(qk_elems * sizeof(__fp16), mode);
    bufs.K = device->newBuffer(qk_elems * sizeof(__fp16), mode);
    bufs.V = device->newBuffer(v_elems * sizeof(__fp16), mode);
    bufs.A = device->newBuffer(a_elems * sizeof(float), mode);
    bufs.O = device->newBuffer(v_elems * sizeof(__fp16), mode);
    std::memcpy(bufs.Q->contents(), Qv.data(), qk_elems * sizeof(__fp16));
    std::memcpy(bufs.K->contents(), Kv.data(), qk_elems * sizeof(__fp16));
    std::memcpy(bufs.V->contents(), Vv.data(), v_elems * sizeof(__fp16));
    std::memcpy(bufs.A->contents(), Av.data(), a_elems * sizeof(float));

    Mamba2ArgsHost args{(uint32_t)B, (uint32_t)H, (uint32_t)L, (uint32_t)(L / CHUNK)};
    bufs.args = device->newBuffer(&args, sizeof(args), mode);
    return bufs;
}

static ParallelBuffers allocate_parallel(MTL::Device* device, int B, int H, int L) {
    ParallelBuffers bufs;
    const size_t seq = (size_t)B * H * L;
    const size_t chunks = (size_t)B * H * (L / CHUNK);
    const auto mode = MTL::ResourceStorageModeShared;

    bufs.local_out = device->newBuffer(seq * D_V * sizeof(__fp16), mode);
    bufs.q_scaled = device->newBuffer(seq * D_QK * sizeof(__fp16), mode);
    bufs.chunk_decay = device->newBuffer(chunks * sizeof(float), mode);
    bufs.chunk_update = device->newBuffer(chunks * STATE_SIZE * sizeof(float), mode);
    return bufs;
}

static double run_baseline(MTL::CommandQueue* queue, MTL::ComputePipelineState* pso,
                           const SharedBuffers& bufs, int B, int H) {
    const size_t v_elems = (size_t)B * H;
    (void)v_elems;
    auto* cmd = queue->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bufs.Q, 0, 0);
    enc->setBuffer(bufs.K, 0, 1);
    enc->setBuffer(bufs.V, 0, 2);
    enc->setBuffer(bufs.A, 0, 3);
    enc->setBuffer(bufs.O, 0, 4);
    enc->setBuffer(bufs.args, 0, 5);
    enc->dispatchThreadgroups(MTL::Size(B, H, 1), MTL::Size(128, 1, 1));
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    return (cmd->GPUEndTime() - cmd->GPUStartTime()) * 1000.0;
}

static double run_parallel(MTL::CommandQueue* queue,
                           MTL::ComputePipelineState* pre_pso,
                           MTL::ComputePipelineState* scan_pso,
                           const SharedBuffers& shared,
                           const ParallelBuffers& parallel,
                           int B, int H, int L) {
    const int chunks = L / CHUNK;
    std::memset(shared.O->contents(), 0, (size_t)B * H * L * D_V * sizeof(__fp16));

    auto* cmd = queue->commandBuffer();

    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pre_pso);
        enc->setBuffer(shared.Q, 0, 0);
        enc->setBuffer(shared.K, 0, 1);
        enc->setBuffer(shared.V, 0, 2);
        enc->setBuffer(shared.A, 0, 3);
        enc->setBuffer(parallel.local_out, 0, 4);
        enc->setBuffer(parallel.q_scaled, 0, 5);
        enc->setBuffer(parallel.chunk_decay, 0, 6);
        enc->setBuffer(parallel.chunk_update, 0, 7);
        enc->setBuffer(shared.args, 0, 8);
        enc->dispatchThreadgroups(MTL::Size(B, H, chunks), MTL::Size(128, 1, 1));
        enc->endEncoding();
    }

    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(scan_pso);
        enc->setBuffer(parallel.chunk_decay, 0, 0);
        enc->setBuffer(parallel.chunk_update, 0, 1);
        enc->setBuffer(parallel.local_out, 0, 2);
        enc->setBuffer(parallel.q_scaled, 0, 3);
        enc->setBuffer(shared.O, 0, 4);
        enc->setBuffer(shared.args, 0, 5);
        enc->dispatchThreadgroups(MTL::Size(B, H, 1), MTL::Size(128, 1, 1));
        enc->endEncoding();
    }

    cmd->commit();
    cmd->waitUntilCompleted();
    return (cmd->GPUEndTime() - cmd->GPUStartTime()) * 1000.0;
}

static std::vector<__fp16> read_half_buffer(MTL::Buffer* buf, size_t elems) {
    std::vector<__fp16> out(elems);
    std::memcpy(out.data(), buf->contents(), elems * sizeof(__fp16));
    return out;
}

} // namespace

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <baseline.metallib> <parallel.metallib> [L=256] [B=1] [H=2]\n", argv[0]);
        return 1;
    }

    const char* baseline_path = argv[1];
    const char* parallel_path = argv[2];
    const int L = (argc > 3) ? std::atoi(argv[3]) : 256;
    const int B = (argc > 4) ? std::atoi(argv[4]) : 1;
    const int H = (argc > 5) ? std::atoi(argv[5]) : 2;
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
    auto* base_url = NS::URL::fileURLWithPath(NS::String::string(baseline_path, NS::UTF8StringEncoding));
    auto* para_url = NS::URL::fileURLWithPath(NS::String::string(parallel_path, NS::UTF8StringEncoding));
    MTL::Library* base_lib = device->newLibrary(base_url, &err);
    if (!base_lib) {
        std::fprintf(stderr, "Failed to load baseline metallib: %s\n",
                     err ? err->localizedDescription()->utf8String() : "(unknown)");
        return 1;
    }
    err = nullptr;
    MTL::Library* para_lib = device->newLibrary(para_url, &err);
    if (!para_lib) {
        std::fprintf(stderr, "Failed to load parallel metallib: %s\n",
                     err ? err->localizedDescription()->utf8String() : "(unknown)");
        return 1;
    }

    auto* base_pso = make_pso(device, base_lib, "mamba2_siso_fwd_32_64_64");
    auto* pre_pso = make_pso(device, para_lib, "mamba2_parallel_precompute_32_64_64");
    auto* scan_pso = make_pso(device, para_lib, "mamba2_parallel_scan_32_64_64");
    if (!base_pso || !pre_pso || !scan_pso) {
        std::fprintf(stderr, "Failed to create one or more PSOs\n");
        return 1;
    }

    const size_t qk_elems = (size_t)B * H * L * D_QK;
    const size_t v_elems = (size_t)B * H * L * D_V;
    std::vector<__fp16> Q(qk_elems), K(qk_elems), V(v_elems);
    std::vector<float> A((size_t)B * H * L);
    fill_inputs(Q, K, V, A, 42);

    std::vector<float> ref;
    auto cpu_t0 = std::chrono::steady_clock::now();
    cpu_reference(Q, K, V, A, ref, B, H, L);
    auto cpu_t1 = std::chrono::steady_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();

    SharedBuffers base_shared = allocate_shared(device, B, H, L, Q, K, V, A);
    SharedBuffers para_shared = allocate_shared(device, B, H, L, Q, K, V, A);
    ParallelBuffers parallel = allocate_parallel(device, B, H, L);

    auto warm_baseline = [&]() { return run_baseline(queue, base_pso, base_shared, B, H); };
    auto warm_parallel = [&]() { return run_parallel(queue, pre_pso, scan_pso, para_shared, parallel, B, H, L); };

    for (int i = 0; i < 5; ++i) {
        warm_baseline();
        warm_parallel();
    }

    std::vector<double> base_times;
    std::vector<double> para_times;
    for (int i = 0; i < 20; ++i) {
        base_times.push_back(run_baseline(queue, base_pso, base_shared, B, H));
        para_times.push_back(run_parallel(queue, pre_pso, scan_pso, para_shared, parallel, B, H, L));
    }
    std::sort(base_times.begin(), base_times.end());
    std::sort(para_times.begin(), para_times.end());
    const double base_ms = base_times[base_times.size() / 2];
    const double para_ms = para_times[para_times.size() / 2];

    auto base_out = read_half_buffer(base_shared.O, v_elems);
    auto para_out = read_half_buffer(para_shared.O, v_elems);
    const auto base_err = compare_hf(base_out, ref);
    const auto para_err = compare_hf(para_out, ref);

    std::printf("== mamba2 baseline vs chunked-parallel ==\n");
    std::printf("  B=%d H=%d L=%d D_qk=%d D_v=%d CHUNK=%d\n", B, H, L, D_QK, D_V, CHUNK);
    std::printf("  CPU ref:      %.3f ms\n", cpu_ms);
    std::printf("  baseline GPU: %.3f ms  max_abs=%.5f mean_abs=%.5f l2_rel=%.5f\n",
                base_ms, base_err.max_abs, base_err.mean_abs, base_err.l2_rel);
    std::printf("  parallel GPU: %.3f ms  max_abs=%.5f mean_abs=%.5f l2_rel=%.5f\n",
                para_ms, para_err.max_abs, para_err.mean_abs, para_err.l2_rel);
    std::printf("  parallel vs baseline speedup: %.2fx\n", base_ms / para_ms);

    parallel.release();
    base_shared.release();
    para_shared.release();
    base_pso->release();
    pre_pso->release();
    scan_pso->release();
    base_lib->release();
    para_lib->release();
    queue->release();
    device->release();
    return 0;
}
