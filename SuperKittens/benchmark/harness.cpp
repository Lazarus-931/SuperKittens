//
//  harness.cpp
//  SuperKittens
//
//  Created by Alazar Manakelew on 4/16/26.
//

/////////////////////////// General Benchmarking Harness ////////////////////////////////////////////


#include "../meow.h"


namespace meow::bench {

struct results {
    float median, min, max;
    float gflops, gbps;
    int iters;
};


inline MTL::Buffer* rand_buffer(MTL::Device* dev, size_t count) {
    auto* buf = dev->newBuffer(count * sizeof(__fp16), MTL::ResourceStorageModeShared);
    auto* ptr = static_cast<__fp16*>(buf->contents());
    for (size_t i = 0; i < count; i++)
        ptr[i] = (__fp16)((float)arc4random() / UINT32_MAX - 0.5f);
    return buf;
}

inline float median(float* v, int n) {
     std::sort(v, v + n);
     return (n % 2) ? v[n/2] : (v[n/2 - 1] + v[n/2]) * 0.5f;
 }

 // Core timing loop
 inline BenchResult run(MTL::CommandQueue* queue,
                        MTL::ComputePipelineState* pipeline,
                        std::function<void(MTL::ComputeCommandEncoder*)> encode,
                        MTL::Size grid, MTL::Size group,
                        int warmup = 5, int iters = 20) {
     // warmup
     for (int i = 0; i < warmup; i++) {
         auto* cmd = queue->commandBuffer();
         auto* enc = cmd->computeCommandEncoder();
         enc->setComputePipelineState(pipeline);
         encode(enc);
         enc->dispatchThreadgroups(grid, group);
         enc->endEncoding();
         cmd->commit();
         cmd->waitUntilCompleted();
     }

     float times[iters];
     for (int i = 0; i < iters; i++) {
         auto* cmd = queue->commandBuffer();
         auto* enc = cmd->computeCommandEncoder();
         enc->setComputePipelineState(pipeline);
         encode(enc);
         enc->dispatchThreadgroups(grid, group);
         enc->endEncoding();
         cmd->commit();
         cmd->waitUntilCompleted();
         times[i] = (float)(cmd->GPUEndTime() - cmd->GPUStartTime()) * 1000.0f;
     }

     BenchResult r;
     r.iters = iters;
     r.median_ms = median(times, iters);
     r.min_ms = *std::min_element(times, times + iters);
     r.max_ms = *std::max_element(times, times + iters);
     r.gflops = 0;
     r.gbps = 0;
     return r;
 }


}





