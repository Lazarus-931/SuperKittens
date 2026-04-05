//
//  main.cpp
//  SuperKittens
//
//  Usage: ./SuperKittens <kernel_name> <seq> <d>
//  Example: ./SuperKittens attn_2048_128 2048 128

#include "../metal-cpp/Foundation/Foundation.hpp"
#include "../metal-cpp/Metal/Metal.hpp"
#include "../metal-cpp/QuartzCore/QuartzCore.hpp"
#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>
#include <functional>
#include "attention_ref.h"

int main(int argc, const char* argv[]) {

    const char* kernelName = "attn_2048_128";
    uint32_t seq = 2048;
    uint32_t d = 128;

    if (argc > 1) kernelName = argv[1];
    if (argc > 2) seq = atoi(argv[2]);
    if (argc > 3) d = atoi(argv[3]);

    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) { std::cerr << "No Metal device\n"; return 1; }

    MTL::CommandQueue* queue = device->newCommandQueue();
    MTL::Library* lib = device->newDefaultLibrary();
    if (!lib) { std::cerr << "No default Metal library\n"; return 1; }

    std::cout << "SuperKittens — " << device->name()->utf8String() << std::endl;

    NS::Error* err = nullptr;
    auto* fn = lib->newFunction(NS::String::string(kernelName, NS::UTF8StringEncoding));
    if (!fn) { std::cerr << "Kernel not found: " << kernelName << std::endl; return 1; }
    auto* pso = device->newComputePipelineState(fn, &err);
    fn->release();
    if (!pso) { std::cerr << "PSO failed\n"; return 1; }

    int iters = (seq <= 512) ? 20 : 10;

    size_t qkv_bytes = seq * d * sizeof(__fp16);
    auto* bufQ = device->newBuffer(qkv_bytes, MTL::ResourceStorageModeShared);
    auto* bufK = device->newBuffer(qkv_bytes, MTL::ResourceStorageModeShared);
    auto* bufV = device->newBuffer(qkv_bytes, MTL::ResourceStorageModeShared);
    auto* bufO = device->newBuffer(qkv_bytes, MTL::ResourceStorageModeShared);
    __fp16* pQ = (__fp16*)bufQ->contents();
    __fp16* pK = (__fp16*)bufK->contents();
    __fp16* pV = (__fp16*)bufV->contents();
    srand(42);
    for (size_t i = 0; i < seq * d; i++) {
        pQ[i] = (__fp16)((float)rand() / RAND_MAX * 0.5f);
        pK[i] = (__fp16)((float)rand() / RAND_MAX * 0.5f);
        pV[i] = (__fp16)((float)rand() / RAND_MAX * 0.5f);
    }

    struct { uint32_t seq; uint32_t head_dim; uint32_t num_heads; uint32_t causal; } params;
    params.seq = seq;
    params.head_dim = d;
    params.num_heads = 32;
    params.causal = 0;
    auto* bufParams = device->newBuffer(&params, sizeof(params), MTL::ResourceStorageModeShared);

    auto* bufSeq = device->newBuffer(&seq, sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* bufD = device->newBuffer(&d, sizeof(uint32_t), MTL::ResourceStorageModeShared);

    uint32_t gy = (seq + 15) / 16;
    bool uses_params = (strstr(kernelName, "attn_") != nullptr);

    auto run = [&]() -> double {
        auto* cmd = queue->commandBuffer();
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(bufQ, 0, 0); enc->setBuffer(bufK, 0, 1);
        enc->setBuffer(bufV, 0, 2); enc->setBuffer(bufO, 0, 3);
        if (uses_params) {
            enc->setBuffer(bufParams, 0, 4);
        } else {
            enc->setBuffer(bufSeq, 0, 4); enc->setBuffer(bufD, 0, 5);
        }
        enc->dispatchThreadgroups(MTL::Size(1, gy, 1), MTL::Size(128, 1, 1));
        enc->endEncoding();
        cmd->commit(); cmd->waitUntilCompleted();
        return cmd->GPUEndTime() - cmd->GPUStartTime();
    };

    // ── Correctness check ──
    printf("\n%-20s seq=%-6u d=%-6u\n", kernelName, seq, d);
    printf("  Verifying against CPU reference...\n");

    std::vector<float> cpu_out(seq * d);
    attention_cpu(pQ, pK, pV, cpu_out.data(), seq, d);

    memset(bufO->contents(), 0, qkv_bytes);
    run();

    __fp16* gpu_out = (__fp16*)bufO->contents();
    auto vr = verify_attention(gpu_out, cpu_out.data(), seq, d);

    printf("  max_abs_err:  %.6f  (row %u, col %u)\n", vr.max_abs_err, vr.worst_row, vr.worst_col);
    printf("  mean_abs_err: %.6f\n", vr.mean_abs_err);
    printf("  l2_rel_err:   %.6f\n", vr.l2_rel_err);

    if (!vr.pass) {
        printf("  FAIL — output diverges from CPU reference. Skipping benchmark.\n");
        printf("\n  Sample mismatches (first 5):\n");
        int shown = 0;
        for (uint32_t i = 0; i < seq && shown < 5; i++) {
            for (uint32_t k = 0; k < d && shown < 5; k++) {
                float g = (float)gpu_out[i * d + k];
                float c = cpu_out[i * d + k];
                if (fabsf(g - c) > 0.05f) {
                    printf("    [%u,%u] gpu=%.4f cpu=%.4f diff=%.4f\n", i, k, g, c, g - c);
                    shown++;
                }
            }
        }
        bufQ->release(); bufK->release(); bufV->release(); bufO->release();
        bufSeq->release(); bufD->release(); bufParams->release(); pso->release();
        queue->release(); lib->release(); device->release();
        return 1;
    }

    printf("  PASS\n");

    // ── Benchmark ──
    for (int i = 0; i < 3; i++) run();
    std::vector<double> times;
    for (int i = 0; i < iters; i++) times.push_back(run());
    std::sort(times.begin(), times.end());
    double t = times[iters / 2];

    double us = t * 1e6;
    double flops = 4.0 * seq * seq * d + 5.0 * seq * seq;
    double gflops = flops / (us * 1e3);

    printf("  Time:     %.0f us\n", us);
    printf("  GFLOPS:   %.1f\n", gflops);
    printf("  Eff:      %.1f%%\n", gflops / 5170.0 * 100);

    // ── Baselines ──
    printf("\n  Baselines:\n");
    std::string python = "/opt/miniconda3/bin/python3";
    std::string baseDir = std::string(getenv("HOME")) + "/SuperKittens/SuperKittens/kernels/attn/baseline/";
    const char* scripts[] = {"torch/bench.py", "mlx/bench.py"};
    const char* names[] = {"torch", "mlx"};
    for (int i = 0; i < 2; i++) {
        std::string cmd = python + " " + baseDir + scripts[i] + " " +
                          std::to_string(seq) + " " + std::to_string(d);
        FILE* pipe = popen(cmd.c_str(), "r");
        if (pipe) {
            char buf[256];
            if (fgets(buf, sizeof(buf), pipe)) {
                float bt, bgf;
                if (sscanf(buf, "%f,%f", &bt, &bgf) == 2) {
                    printf("  %-10s %.0f us, %.1f GFLOPS", names[i], bt, bgf);
                    if (bt > 0) printf(" (%.1fx vs yours)", us / bt);
                    printf("\n");
                }
            }
            pclose(pipe);
        }
    }

    bufQ->release(); bufK->release(); bufV->release(); bufO->release();
    bufSeq->release(); bufD->release(); bufParams->release(); pso->release();
    queue->release(); lib->release(); device->release();
    return 0;
}
