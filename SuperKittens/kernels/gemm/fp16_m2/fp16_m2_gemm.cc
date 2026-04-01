//
//  fp16_m2_gemm.c++
//  SuperKittens
//
//  Created by Alazar Manakelew on 4/1/26.
//

#include "../../../../metal-cpp/Foundation/Foundation.hpp"
#include "../../../../metal-cpp/Metal/Metal.hpp"
#include <cstdint>

namespace superkittens {

struct fp16_m2_gemm_config {
    static constexpr uint32_t MB = 256;
    static constexpr uint32_t NB = 128;
    static constexpr uint32_t KB = 32;

    static_assert(MB == 256, "MB needs to be 256");
    static_assert(NB >= 16 && NB <= 256 && NB % 16 == 0, "NB needs to be 16, 32...256");
    static_assert(KB >= 16 && KB % 16 == 0, "KB needs to be 16,...128,...512...");
};

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
