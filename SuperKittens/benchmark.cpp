//
//  benchmark.cpp
//  SuperKittens
//
//  Benchmarks a single Metal attention kernel against baselines.
//  Called by main.cpp with: runBenchmark(device, queue, lib, seq, d, kernelName)

#include "../metal-cpp/Foundation/Foundation.hpp"
#include "../metal-cpp/Metal/Metal.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <cstdlib>
#include <cstring>

#include "benchmark.h"

static double dispatch(
    MTL::CommandQueue* queue,
    MTL::ComputePipelineState* pso,
    std::function<void(MTL::ComputeCommandEncoder*)> encode,
    MTL::Size grid, MTL::Size tg, bool threadgroups) {
    auto* cmd = queue->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    encode(enc);
    if (threadgroups) enc->dispatchThreadgroups(grid, tg);
    else enc->dispatchThreads(grid, tg);
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    return cmd->GPUEndTime() - cmd->GPUStartTime();
}

static double benchMedian(int warmup, int trials, std::function<double()> run) {
    for (int i = 0; i < warmup; i++) run();
    std::vector<double> times;
    for (int i = 0; i < trials; i++) times.push_back(run());
    std::sort(times.begin(), times.end());
    return times[trials / 2];
}

void runBenchmark(
    MTL::Device* device,
    MTL::CommandQueue* queue,
    MTL::Library* lib,
    uint32_t seq,
    uint32_t d,
    const char* kernelName) {

    NS::Error* err = nullptr;
    auto* fn = lib->newFunction(NS::String::string(kernelName, NS::UTF8StringEncoding));
    if (!fn) { std::cerr << "Kernel not found: " << kernelName << std::endl; return; }
    auto* pso = device->newComputePipelineState(fn, &err);
    fn->release();
    if (!pso) { std::cerr << "PSO failed for: " << kernelName << std::endl; return; }

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

    auto* bufSeq = device->newBuffer(&seq, sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* bufD = device->newBuffer(&d, sizeof(uint32_t), MTL::ResourceStorageModeShared);

    uint32_t gy = (seq + 15) / 16;

    double t = benchMedian(3, iters, [&]() -> double {
        return dispatch(queue, pso, [&](MTL::ComputeCommandEncoder* enc) {
            enc->setBuffer(bufQ, 0, 0); enc->setBuffer(bufK, 0, 1);
            enc->setBuffer(bufV, 0, 2); enc->setBuffer(bufO, 0, 3);
            enc->setBuffer(bufSeq, 0, 4); enc->setBuffer(bufD, 0, 5);
        }, MTL::Size(1, gy, 1), MTL::Size(128, 1, 1), true);
    });

    double us = t * 1e6;
    double flops = 4.0 * seq * seq * d + 5.0 * seq * seq;
    double gflops = flops / (us * 1e3);
    double eff = gflops / 5170.0 * 100;

    printf("\n%-20s seq=%-6u d=%-6u\n", kernelName, seq, d);
    printf("  Time:     %.0f us\n", us);
    printf("  GFLOPS:   %.1f\n", gflops);
    printf("  Eff:      %.1f%%\n", eff);

    // run baselines
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
    bufSeq->release(); bufD->release(); pso->release();
}
