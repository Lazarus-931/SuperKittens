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
#include <vector>
#include <cmath>
#include <cassert>

// CPU reference for verification
void cpu_attention(const std::vector<float>& Q,
                   const std::vector<float>& K,
                   const std::vector<float>& V,
                   std::vector<float>& O,
                   uint32_t N, uint32_t d) {
    float scale = 1.0f / sqrtf((float)d);
    for (uint32_t i = 0; i < N; i++) {
        std::vector<float> scores(N);
        float max_s = -1e9f;
        for (uint32_t j = 0; j < N; j++) {
            float dot = 0;
            for (uint32_t k = 0; k < d; k++)
                dot += Q[i*d+k] * K[j*d+k];
            scores[j] = dot * scale;
            max_s = std::max(max_s, scores[j]);
        }
        float sum = 0;
        for (uint32_t j = 0; j < N; j++) { scores[j] = expf(scores[j] - max_s); sum += scores[j]; }
        for (uint32_t j = 0; j < N; j++) scores[j] /= sum;
        for (uint32_t col = 0; col < d; col++) {
            float out = 0;
            for (uint32_t j = 0; j < N; j++) out += scores[j] * V[j*d+col];
            O[i*d+col] = out;
        }
    }
}

int main() {
    const uint32_t N = 64;   // sequence length
    const uint32_t d = 64;   // head dimension

    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    std::cout << "GPU: " << device->name()->utf8String() << "\n";

    NS::Error* error = nullptr;
    MTL::Library* lib = device->newDefaultLibrary();
    assert(lib && "No default library");

    MTL::Function* fn = lib->newFunction(
        NS::String::string("attention_forward", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pso =
        device->newComputePipelineState(fn, &error);
    assert(pso);

    // Init Q, K, V with random values
    uint32_t size = N * d;
    std::vector<float> Q(size), K(size), V(size), O_gpu(size), O_cpu(size);
    for (uint32_t i = 0; i < size; i++) {
        Q[i] = ((float)rand() / RAND_MAX) * 0.1f;
        K[i] = ((float)rand() / RAND_MAX) * 0.1f;
        V[i] = ((float)rand() / RAND_MAX) * 0.1f;
    }

    // GPU buffers
    auto* bufQ = device->newBuffer(Q.data(), size*sizeof(float), MTL::ResourceStorageModeShared);
    auto* bufK = device->newBuffer(K.data(), size*sizeof(float), MTL::ResourceStorageModeShared);
    auto* bufV = device->newBuffer(V.data(), size*sizeof(float), MTL::ResourceStorageModeShared);
    auto* bufO = device->newBuffer(size*sizeof(float), MTL::ResourceStorageModeShared);
    auto* bufN = device->newBuffer(&N, sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* bufD = device->newBuffer(&d, sizeof(uint32_t), MTL::ResourceStorageModeShared);

    // Dispatch
    MTL::CommandQueue* queue = device->newCommandQueue();
    MTL::CommandBuffer* cmd  = queue->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();

    enc->setComputePipelineState(pso);
    enc->setBuffer(bufQ, 0, 0);
    enc->setBuffer(bufK, 0, 1);
    enc->setBuffer(bufV, 0, 2);
    enc->setBuffer(bufO, 0, 3);
    enc->setBuffer(bufN, 0, 4);
    enc->setBuffer(bufD, 0, 5);

    MTL::Size grid(d, N, 1);      
    MTL::Size threads(8, 8, 1);
    enc->dispatchThreads(grid, threads);
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();

    // CPU reference
    cpu_attention(Q, K, V, O_cpu, N, d);

    // Compare
    float* gpu_out = (float*)bufO->contents();
    float max_err = 0.0f;
    for (uint32_t i = 0; i < size; i++) {
        max_err = std::max(max_err, std::abs(gpu_out[i] - O_cpu[i]));
    }
    std::cout << "Max error GPU vs CPU: " << max_err << "\n";
    std::cout << (max_err < 1e-4f ? "ATTENTION CORRECT!\n" : "ATTENTION WRONG\n");

    // Cleanup
    bufQ->release(); bufK->release(); bufV->release();
    bufO->release(); bufN->release(); bufD->release();
    pso->release(); fn->release(); lib->release();
    queue->release(); device->release();
    return 0;
}
