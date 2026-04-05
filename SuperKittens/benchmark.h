#ifndef SUPERKITTENS_BENCHMARK_H
#define SUPERKITTENS_BENCHMARK_H

#include "../metal-cpp/Metal/Metal.hpp"

void runBenchmark(
    MTL::Device* device,
    MTL::CommandQueue* queue,
    MTL::Library* lib,
    uint32_t seq,
    uint32_t d,
    const char* kernelName);

#endif
