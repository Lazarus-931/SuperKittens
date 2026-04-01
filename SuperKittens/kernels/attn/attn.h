//
//  attn.h
//  SuperKittens
//
//  Created by Alazar Manakelew on 4/1/26.
//

#ifndef SUPERKITTENS_ATTN_H
#define SUPERKITTENS_ATTN_H

#include "../../../metal-cpp/Foundation/Foundation.hpp"
#include "../../../metal-cpp/Metal/Metal.hpp"

namespace superkittens {

MTL::ComputePipelineState* create_attention_pipeline(MTL::Device* device, MTL::Library* lib) {
    auto* fn = lib->newFunction(
        NS::String::string("attention_forward", NS::UTF8StringEncoding));
    if (!fn) return nullptr;

    NS::Error* error = nullptr;
    auto* pso = device->newComputePipelineState(fn, &error);
    fn->release();
    return pso;
}

} // namespace superkittens

#endif // SUPERKITTENS_ATTN_H
