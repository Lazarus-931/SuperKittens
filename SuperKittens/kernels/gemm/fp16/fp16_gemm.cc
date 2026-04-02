//
//  fp16_m2_gemm.c++
//  SuperKittens
//
//  Created by Alazar Manakelew on 4/1/26.
//

#include "../../../../metal-cpp/Foundation/Foundation.hpp"
#include "../../../../metal-cpp/Metal/Metal.hpp"
#include <cstdint>
#include "types.h"
namespace superkittens {










MTL::ComputePipelineState* create_fp16_gemm_pipeline(MTL::Device* device, MTL::Library* lib) {
    auto* fn = lib->newFunction(
        NS::String::string("fp16_gemm", NS::UTF8StringEncoding));
    if (!fn) return nullptr;

    NS::Error* error = nullptr;
    auto* pso = device->newComputePipelineState(fn, &error);
    fn->release();
    return pso;
}

} // namespace superkittens
