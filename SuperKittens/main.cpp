//
//  main.cpp
//  SuperKittens
//
//  Created by Alazar Manakelew on 3/31/26.
//

#include "../metal-cpp/Foundation/Foundation.hpp"
#include "../metal-cpp/Metal/Metal.hpp"
#include "../metal-cpp/QuartzCore/QuartzCore.hpp"
#include <iostream>
#include <cstring>
#include <chrono>
#include <vector>

int main(int argc, const char* argv[]) {
    // Defaults
    uint32_t M = 512, N = 512, K = 512;
    const char* kernel_name = "fp16_gemm_baseline";
    int iters = 10;

    // Parse args: ./SuperKittens [kernel] [M] [N] [K] [iters]
    if (argc > 1) kernel_name = argv[1];
    if (argc > 2) M = atoi(argv[2]);
    if (argc > 3) N = atoi(argv[3]);
    if (argc > 4) K = atoi(argv[4]);
    if (argc > 5) iters = atoi(argv[5]);
    


    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) { std::cerr << "No Metal device\n"; return 1; }
    std::cout << "GPU: " << device->name()->utf8String() << std::endl;

    NS::Error* error = nullptr;
    MTL::Library* lib = device->newDefaultLibrary();
    if (!lib) { std::cerr << "No default Metal library\n"; return 1; }

    auto* fn = lib->newFunction(NS::String::string(kernel_name, NS::UTF8StringEncoding));
    if (!fn) { std::cerr << "Kernel '" << kernel_name << "' not found\n"; return 1; }
    auto* pso = device->newComputePipelineState(fn, &error);
    if (!pso) { std::cerr << "Pipeline failed\n"; return 1; }

    std::cout << "Kernel: " << kernel_name << std::endl;
    std::cout << "M=" << M << " N=" << N << " K=" << K << " iters=" << iters << std::endl;

    // Buffers
    uint32_t sizeA = M * K, sizeB = K * N, sizeC = M * N;
    std::vector<__fp16> A(sizeA), B(sizeB);
    srand(42);
    for (uint32_t i = 0; i < sizeA; i++) A[i] = (__fp16)(((float)rand() / RAND_MAX) * 0.5f);
    for (uint32_t i = 0; i < sizeB; i++) B[i] = (__fp16)(((float)rand() / RAND_MAX) * 0.5f);

    auto* bufA = device->newBuffer(A.data(), sizeA * sizeof(__fp16), MTL::ResourceStorageModeShared);
    auto* bufB = device->newBuffer(B.data(), sizeB * sizeof(__fp16), MTL::ResourceStorageModeShared);
    auto* bufC = device->newBuffer(sizeC * sizeof(__fp16), MTL::ResourceStorageModeShared);
    auto* bufM = device->newBuffer(&M, sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* bufN = device->newBuffer(&N, sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* bufK = device->newBuffer(&K, sizeof(uint32_t), MTL::ResourceStorageModeShared);

    MTL::CommandQueue* queue = device->newCommandQueue();

    // GPU capture — save to tmp/run_N.gputrace
    std::string traceDir = std::string(getenv("HOME")) + "/SuperKittens/tmp/";
    int runNum = 1;
    while (true) {
        std::string p = traceDir + "run_" + std::to_string(runNum) + ".gputrace";
        FILE* f = fopen(p.c_str(), "r");
        if (!f) break;
        fclose(f);
        runNum++;
    }
    std::string tracePath = traceDir + "run_" + std::to_string(runNum) + ".gputrace";

    auto* captureMgr = MTL::CaptureManager::sharedCaptureManager();
    auto* capDesc = MTL::CaptureDescriptor::alloc()->init();
    capDesc->setCaptureObject(device);
    capDesc->setDestination(MTL::CaptureDestinationGPUTraceDocument);
    capDesc->setOutputURL(NS::URL::fileURLWithPath(
        NS::String::string(tracePath.c_str(), NS::UTF8StringEncoding)));
    NS::Error* captureErr = nullptr;
    bool capturing = false;
    if (!captureMgr->supportsDestination(MTL::CaptureDestinationGPUTraceDocument)) {
        std::cerr << "GPU trace not supported — enable GPU Frame Capture in Xcode scheme (Edit Scheme > Run > Options > GPU Frame Capture > Metal)" << std::endl;
    } else {
        capturing = captureMgr->startCapture(capDesc, &captureErr);
        if (!capturing && captureErr) {
            std::cerr << "Capture failed: " << captureErr->localizedDescription()->utf8String() << std::endl;
        }
    }
    capDesc->release();

    // Threadgroup size: 128 for tiled kernels, 256 for baseline
    bool is_baseline = (strstr(kernel_name, "baseline") != nullptr);
    MTL::Size threads = is_baseline ? MTL::Size(16, 16, 1) : MTL::Size(16, 8, 1);
    MTL::Size grid(N, M, 1);

    // Warmup
    for (int w = 0; w < 3; w++) {
        auto* cmd = queue->commandBuffer();
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(bufA, 0, 0); enc->setBuffer(bufB, 0, 1);
        enc->setBuffer(bufC, 0, 2); enc->setBuffer(bufM, 0, 3);
        enc->setBuffer(bufN, 0, 4); enc->setBuffer(bufK, 0, 5);
        enc->dispatchThreads(grid, threads);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        auto* cmd = queue->commandBuffer();
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(bufA, 0, 0); enc->setBuffer(bufB, 0, 1);
        enc->setBuffer(bufC, 0, 2); enc->setBuffer(bufM, 0, 3);
        enc->setBuffer(bufN, 0, 4); enc->setBuffer(bufK, 0, 5);
        enc->dispatchThreads(grid, threads);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (double)iters;
    double gflops = (2.0 * M * N * K) / (us * 1e3);

    if (capturing) {
        captureMgr->stopCapture();
        std::cout << "\nGPU trace: " << tracePath << std::endl;
    }

    std::cout << "Time:   " << us << " us" << std::endl;
    std::cout << "GFLOPS: " << gflops << std::endl;

    // Cleanup
    bufA->release(); bufB->release(); bufC->release();
    bufM->release(); bufN->release(); bufK->release();
    fn->release(); pso->release();
    queue->release(); lib->release(); device->release();
    return 0;
}
